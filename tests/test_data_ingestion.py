import pytest
import pandas as pd
import io
import os
from src.pipeline.data_ingestion import (
    normalize_ibm_amlsim, load_and_clean, get_summary_stats, load_ibm_pipeline, generate_synthetic_data
)

@pytest.fixture
def sample_ibm_csv():
    """Returns a stringIO object with synthetic IBM-schema CSV data."""
    df = generate_synthetic_data(50)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

def test_normalize_ibm_columns(sample_ibm_csv):
    """Output has all 9 project schema columns."""
    # Write to a temp file because normalize_ibm_amlsim expects a path
    with open("temp_test.csv", "w") as f:
        f.write(sample_ibm_csv.getvalue())
    
    df = normalize_ibm_amlsim("temp_test.csv")
    os.remove("temp_test.csv")
    
    expected_cols = [
        'transaction_id', 'timestamp', 'sender_id', 'receiver_id', 
        'amount', 'transaction_type', 'sender_country', 'receiver_country', 'is_laundering'
    ]
    assert all(col in df.columns for col in expected_cols)
    assert len(df.columns) == 9

def test_load_and_clean_adds_features():
    """Output has hour_of_day, day_of_week, is_cross_border, amount_log."""
    df_raw = generate_synthetic_data(10)
    # We need to normalize first to get the right columns
    with open("temp_test_2.csv", "w") as f:
        df_raw.to_csv(f, index=False)
    df_norm = normalize_ibm_amlsim("temp_test_2.csv")
    os.remove("temp_test_2.csv")
    
    df_clean = load_and_clean(df_norm)
    
    assert 'hour_of_day' in df_clean.columns
    assert 'day_of_week' in df_clean.columns
    assert 'is_cross_border' in df_clean.columns
    assert 'amount_log' in df_clean.columns
    assert df_clean['amount_log'].dtype == float

def test_null_amount_rows_dropped():
    """Rows with null amount are removed."""
    df = pd.DataFrame({
        'transaction_id': ['T1', 'T2'],
        'timestamp': [pd.Timestamp.now(), pd.Timestamp.now()],
        'sender_id': ['S1', 'S2'],
        'receiver_id': ['R1', 'R2'],
        'amount': [100.0, None],
        'transaction_type': ['WIRE', 'WIRE'],
        'sender_country': ['US', 'US'],
        'receiver_country': ['UK', 'UK'],
        'is_laundering': [0, 0]
    })
    cleaned_df = load_and_clean(df)
    assert len(cleaned_df) == 1
    assert cleaned_df['transaction_id'].iloc[0] == 'T1'

def test_self_transfers_dropped():
    """Rows where sender_id == receiver_id are removed."""
    df = pd.DataFrame({
        'transaction_id': ['T1', 'T2'],
        'timestamp': [pd.Timestamp.now(), pd.Timestamp.now()],
        'sender_id': ['S1', 'S1'],
        'receiver_id': ['R1', 'S1'],
        'amount': [100.0, 100.0],
        'transaction_type': ['WIRE', 'WIRE'],
        'sender_country': ['US', 'US'],
        'receiver_country': ['UK', 'US'],
        'is_laundering': [0, 0]
    })
    cleaned_df = load_and_clean(df)
    assert len(cleaned_df) == 1

def test_negative_amount_dropped():
    """Rows with amount <= 0 are removed."""
    df = pd.DataFrame({
        'transaction_id': ['T1', 'T2', 'T3'],
        'timestamp': [pd.Timestamp.now()] * 3,
        'sender_id': ['S1', 'S2', 'S3'],
        'receiver_id': ['R1', 'R2', 'R3'],
        'amount': [100.0, 0.0, -50.0],
        'transaction_type': ['WIRE'] * 3,
        'sender_country': ['US'] * 3,
        'receiver_country': ['UK'] * 3,
        'is_laundering': [0] * 3
    })
    cleaned_df = load_and_clean(df)
    assert len(cleaned_df) == 1

def test_transaction_type_mapping():
    """'Cheque' maps to 'ACH', 'Bills' maps to 'CASH'."""
    data = {
        'Timestamp': ["2022/09/01 00:00"] * 2,
        'Account': ["S1", "S2"],
        'Account.1': ["R1", "R2"],
        'Amount Paid': [100.0, 200.0],
        'Payment Format': ["Cheque", "Bills"],
        'Payment Currency': ["USD", "USD"],
        'Receiving Currency': ["USD", "USD"],
        'Is Laundering': [0, 0]
    }
    df_raw = pd.DataFrame(data)
    with open("temp_test_3.csv", "w") as f:
        df_raw.to_csv(f, index=False)
    df_norm = normalize_ibm_amlsim("temp_test_3.csv")
    os.remove("temp_test_3.csv")
    
    assert list(df_norm['transaction_type']) == ["ACH", "CASH"]

def test_summary_stats_keys():
    """get_summary_stats returns all required keys."""
    df = generate_synthetic_data(10)
    with open("temp_test_4.csv", "w") as f:
        df.to_csv(f, index=False)
    processed_df = load_ibm_pipeline("temp_test_4.csv")
    os.remove("temp_test_4.csv")
    
    stats = get_summary_stats(processed_df)
    required_keys = {
        'total_transactions', 'laundering_count', 'laundering_rate',
        'transaction_type_breakdown', 'cross_border_count', 'cross_border_rate',
        'amount_mean', 'amount_std', 'amount_median'
    }
    assert required_keys.issubset(stats.keys())

def test_load_ibm_pipeline_end_to_end():
    """Full pipeline runs on 50-row synthetic IBM-schema CSV."""
    df = generate_synthetic_data(50)
    with open("temp_test_5.csv", "w") as f:
        df.to_csv(f, index=False)
    processed_df = load_ibm_pipeline("temp_test_5.csv")
    os.remove("temp_test_5.csv")
    
    assert len(processed_df) > 0
    assert 'amount_log' in processed_df.columns
