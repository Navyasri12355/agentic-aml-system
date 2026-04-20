"""
data_ingestion.py
-----------------
Phase 1: Data loading, validation, cleaning, and feature engineering.

Input  : Path to raw CSV file with transaction data
Output : Cleaned pandas DataFrame with engineered features

Expected raw CSV columns:
    transaction_id   : str
    timestamp        : str (ISO 8601)
    sender_id        : str
    receiver_id      : str
    amount           : float
    transaction_type : str  (WIRE | ACH | CASH | CRYPTO | INTERNAL)
    sender_country   : str  (ISO 3166-1 alpha-2)
    receiver_country : str  (ISO 3166-1 alpha-2)
    is_laundering    : int  (0 = clean, 1 = suspicious) — ground truth label
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_TRANSACTION_TYPES = {"WIRE", "ACH", "CASH", "CRYPTO", "INTERNAL"}

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "sender_id",
    "receiver_id",
    "amount",
    "transaction_type",
    "sender_country",
    "receiver_country",
]


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load and clean a raw transaction CSV.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Cleaned DataFrame with engineered features added.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or data fails validation.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {filepath}")

    logger.info(f"Loading transaction data from {filepath}")
    df = pd.read_csv(filepath)

    df = _validate_columns(df)
    df = _drop_nulls(df)
    df = _validate_rows(df)
    df = _engineer_features(df)

    logger.info(f"Loaded {len(df):,} transactions after cleaning.")
    return df


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["sender_id", "receiver_id", "amount", "timestamp"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with null values in key fields.")
    return df


def _validate_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Amount must be positive
    invalid_amount = df["amount"] <= 0
    if invalid_amount.any():
        logger.warning(f"Dropping {invalid_amount.sum()} rows with amount <= 0.")
        df = df[~invalid_amount]

    # No self-transfers
    self_transfer = df["sender_id"] == df["receiver_id"]
    if self_transfer.any():
        logger.warning(f"Dropping {self_transfer.sum()} self-transfer rows.")
        df = df[~self_transfer]

    # Deduplicate on transaction_id
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_id"])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} duplicate transaction_id rows.")

    # Normalize transaction_type
    df["transaction_type"] = df["transaction_type"].str.upper().str.strip()
    unknown_types = ~df["transaction_type"].isin(ALLOWED_TRANSACTION_TYPES)
    if unknown_types.any():
        logger.warning(
            f"{unknown_types.sum()} rows have unknown transaction_type — "
            "setting to 'OTHER'."
        )
        df.loc[unknown_types, "transaction_type"] = "OTHER"

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Time features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday

    # Cross-border flag
    df["is_cross_border"] = (
        df["sender_country"].str.upper() != df["receiver_country"].str.upper()
    ).astype(int)

    # Log-transform amount (handles skew)
    df["amount_log"] = np.log1p(df["amount"])

    return df.reset_index(drop=True)