import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import Optional, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_ibm_amlsim(filepath: str) -> pd.DataFrame:
    """
    Maps raw IBM HI-Small columns to project schema.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Normalized DataFrame with project schema.
    """
    logger.info(f"Normalizing IBM AMLSim data from {filepath}")
    
    # Read CSV
    # Note: IBM HI-Small is large, but we assume it fits in memory for this function
    # unless using the pipeline chunking.
    df = pd.read_csv(filepath)
    
    # Mapping logic
    mapping = {
        'Timestamp': 'timestamp',
        'Account': 'sender_id',
        'Account.1': 'receiver_id',
        'Amount Paid': 'amount',
        'Payment Format': 'transaction_type',
        'Payment Currency': 'sender_country',
        'Receiving Currency': 'receiver_country',
        'Is Laundering': 'is_laundering'
    }
    
    # Keep only relevant columns and rename
    df = df[list(mapping.keys())].rename(columns=mapping).copy()
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M')
    
    # Cast IDs to string
    df['sender_id'] = df['sender_id'].astype(str)
    df['receiver_id'] = df['receiver_id'].astype(str)
    
    # Map transaction types
    type_map = {
        'Cheque': 'ACH',
        'Credit Card': 'ACH',
        'Reinvestment': 'WIRE',
        'Bills': 'CASH',
        'WIRE': 'WIRE',
        'ACH': 'ACH',
        'CASH': 'CASH'
    }
    df['transaction_type'] = df['transaction_type'].map(lambda x: type_map.get(x, x.upper()))
    
    # Generate transaction_id: TXN_ + zero-padded row index
    df['transaction_id'] = [f"TXN_{i:06d}" for i in range(len(df))]
    
    # Reorder columns to project schema
    schema_cols = [
        'transaction_id', 'timestamp', 'sender_id', 'receiver_id', 
        'amount', 'transaction_type', 'sender_country', 'receiver_country', 'is_laundering'
    ]
    
    return df[schema_cols]

def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the normalized DataFrame and applies cleaning/feature engineering.
    
    Args:
        df: Normalized DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with additional features.
        
    Raises:
        ValueError: If validation fails for critical rows.
    """
    logger.info("Cleaning and engineering features for normalized data")
    
    initial_count = len(df)
    
    # 1. Drop rows where critical fields are null
    df = df.dropna(subset=['sender_id', 'receiver_id', 'amount', 'timestamp'])
    null_dropped = initial_count - len(df)
    if null_dropped > 0:
        logger.warning(f"Dropped {null_dropped} rows with null values")
    
    # 2. Add transaction_id if missing (should be present from normalization)
    if 'transaction_id' not in df.columns:
        df['transaction_id'] = [f"TXN_{i:06d}" for i in range(len(df))]
        
    # 3. Validation: amount > 0
    neg_zero_amount = df[df['amount'] <= 0]
    if not neg_zero_amount.empty:
        logger.warning(f"Dropped {len(neg_zero_amount)} rows with amount <= 0")
        df = df[df['amount'] > 0]
        
    # 4. Validation: sender_id != receiver_id
    self_transfers = df[df['sender_id'] == df['receiver_id']]
    if not self_transfers.empty:
        logger.warning(f"Dropped {len(self_transfers)} self-transfers")
        df = df[df['sender_id'] != df['receiver_id']]
        
    # 5. Standardize transaction_type
    df = df.copy()
    df['transaction_type'] = df['transaction_type'].astype(str).str.strip().str.upper()
    
    # 6. Validation: transaction_type whitelist
    allowed_types = {"WIRE", "ACH", "CASH", "CRYPTO", "INTERNAL"}
    unknown_types = df[~df['transaction_type'].isin(allowed_types)]
    if not unknown_types.empty:
        logger.warning(f"Dropped {len(unknown_types)} rows with unknown transaction types")
        df = df[df['transaction_type'].isin(allowed_types)]

    # 7. Feature Engineering
    df = df.copy()
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_cross_border'] = (df['sender_country'] != df['receiver_country']).astype(bool)
    df['amount_log'] = np.log1p(df['amount'])
    
    return df.reset_index(drop=True)

def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns summary statistics for the processed DataFrame.
    """
    total_transactions = len(df)
    if total_transactions == 0:
        return {}
        
    laundering_count = int(df['is_laundering'].sum())
    laundering_rate = float(laundering_count / total_transactions)
    
    type_breakdown = df['transaction_type'].value_counts().to_dict()
    # Convert numpy types to native python types for JSON/Dict compatibility
    type_breakdown = {str(k): int(v) for k, v in type_breakdown.items()}
    
    cross_border_count = int(df['is_cross_border'].sum())
    cross_border_rate = float(cross_border_count / total_transactions)
    
    return {
        'total_transactions': total_transactions,
        'laundering_count': laundering_count,
        'laundering_rate': laundering_rate,
        'transaction_type_breakdown': type_breakdown,
        'cross_border_count': cross_border_count,
        'cross_border_rate': cross_border_rate,
        'amount_mean': float(df['amount'].mean()),
        'amount_std': float(df['amount'].std()),
        'amount_median': float(df['amount'].median())
    }

def load_ibm_pipeline(filepath: str, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    Main entry point that chains normalize_ibm_amlsim and load_and_clean.
    Supports chunking for large files.
    """
    if chunksize is None:
        df_norm = normalize_ibm_amlsim(filepath)
        return load_and_clean(df_norm)
    else:
        logger.info(f"Processing {filepath} in chunks of {chunksize}")
        chunks = []
        # We need a custom reader for chunking because normalize_ibm_amlsim expects a filename
        # and reads the whole thing. For chunking, we map the columns first.
        mapping = {
            'Timestamp': 'timestamp',
            'Account': 'sender_id',
            'Account.1': 'receiver_id',
            'Amount Paid': 'amount',
            'Payment Format': 'transaction_type',
            'Payment Currency': 'sender_country',
            'Receiving Currency': 'receiver_country',
            'Is Laundering': 'is_laundering'
        }
        type_map = {
            'Cheque': 'ACH', 'Credit Card': 'ACH', 'Reinvestment': 'WIRE', 'Bills': 'CASH',
            'WIRE': 'WIRE', 'ACH': 'ACH', 'CASH': 'CASH'
        }
        
        row_offset = 0
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            # Normalize chunk
            chunk = chunk[list(mapping.keys())].rename(columns=mapping).copy()
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='%Y/%m/%d %H:%M')
            chunk['sender_id'] = chunk['sender_id'].astype(str)
            chunk['receiver_id'] = chunk['receiver_id'].astype(str)
            chunk['transaction_type'] = chunk['transaction_type'].map(lambda x: type_map.get(x, x.upper()))
            chunk['transaction_id'] = [f"TXN_{i+row_offset:06d}" for i in range(len(chunk))]
            row_offset += len(chunk)
            
            # Clean chunk
            cleaned_chunk = load_and_clean(chunk)
            chunks.append(cleaned_chunk)
            
        return pd.concat(chunks, ignore_index=True)

def generate_synthetic_data(n_rows: int = 500) -> pd.DataFrame:
    """
    Generates synthetic data matching the IBM Hi-Small schema.
    """
    logger.info(f"Generating {n_rows} rows of synthetic IBM-schema data")
    
    np.random.seed(42)
    
    timestamps = [
        datetime(2022, 9, 1, np.random.randint(0,24), np.random.randint(0,60)).strftime("%Y/%m/%d %H:%M")
        for _ in range(n_rows)
    ]
    
    formats = ["WIRE", "Cheque", "ACH", "Cash", "Credit Card", "Reinvestment", "Bills"]
    currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
    
    data = {
        'Timestamp': timestamps,
        'From Bank': np.random.randint(1, 100, n_rows),
        'Account': [f"ACC_{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'To Bank': np.random.randint(1, 100, n_rows),
        'Account.1': [f"ACC_{np.random.randint(1000, 9999)}" for _ in range(n_rows)],
        'Amount Received': np.random.lognormal(mean=8, sigma=2, size=n_rows),
        'Receiving Currency': np.random.choice(currencies, n_rows),
        'Amount Paid': np.random.lognormal(mean=8, sigma=2, size=n_rows),
        'Payment Currency': np.random.choice(currencies, n_rows),
        'Payment Format': np.random.choice(formats, n_rows),
        'Is Laundering': np.random.choice([0, 1], n_rows, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    raw_path = "data/raw/HI-Small_Trans.csv"
    
    if os.path.exists(raw_path):
        logger.info(f"Data found at {raw_path}. Processing...")
        processed_df = load_ibm_pipeline(raw_path)
    else:
        logger.warning(f"Data not found at {raw_path}. Generating synthetic data...")
        synthetic_raw = generate_synthetic_data(500)
        # To simulate the pipeline, we save it temporarily or just process the DF
        # Given load_ibm_pipeline expects a filepath, let's save synthetic to /tmp
        tmp_path = "data/raw/synthetic_temp.csv"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        synthetic_raw.to_csv(tmp_path, index=False)
        processed_df = load_ibm_pipeline(tmp_path)
        os.remove(tmp_path)
        
    stats = get_summary_stats(processed_df)
    print("\nProject Schema Summary Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\nFirst 5 rows of processed data:")
    print(processed_df.head())
