"""
detection_agent.py
------------------
Phase 1: Anomaly detection using Isolation Forest.

Input  : Cleaned DataFrame from data_ingestion.py
Output : DataFrame of flagged transactions with anomaly scores

Each flagged row includes:
    transaction_id : str
    anomaly_score  : float  (raw IF decision score; more negative = more anomalous)
    is_flagged     : bool
    flag_reason    : str    (human-readable explanation of why it was flagged)
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Features used for Isolation Forest training
NUMERIC_FEATURES = ["amount_log", "hour_of_day", "day_of_week", "is_cross_border"]
CATEGORICAL_FEATURES = ["transaction_type"]

DEFAULT_CONTAMINATION = 0.01
MODEL_PATH = "models/isolation_forest.joblib"
ENCODER_PATH = "models/ohe_encoder.joblib"


def train_detection_model(
    df: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    save_path: str = MODEL_PATH,
) -> IsolationForest:
    """
    Train and save an Isolation Forest on the provided cleaned DataFrame.

    Args:
        df:            Cleaned transaction DataFrame.
        contamination: Expected fraction of anomalies (0.001 – 0.5).
        save_path:     Path to save the trained model artifact.

    Returns:
        Trained IsolationForest instance.
    """
    X, encoder = _build_feature_matrix(df, fit_encoder=True)

    logger.info(
        f"Training Isolation Forest on {len(X):,} transactions "
        f"(contamination={contamination})."
    )
    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    joblib.dump(encoder, ENCODER_PATH)
    logger.info(f"Model saved to {save_path}")

    return model


def run_detection(
    df: pd.DataFrame,
    model_path: Optional[str] = MODEL_PATH,
    contamination: float = DEFAULT_CONTAMINATION,
) -> pd.DataFrame:
    """
    Run anomaly detection on a cleaned transaction DataFrame.

    Loads a saved model if available, otherwise trains a new one.

    Args:
        df:            Cleaned transaction DataFrame.
        model_path:    Path to a saved IsolationForest model (optional).
        contamination: Used only if training a new model.

    Returns:
        DataFrame containing only flagged (suspicious) transactions,
        with columns: transaction_id, anomaly_score, is_flagged, flag_reason.
    """
    if model_path and Path(model_path).exists():
        logger.info(f"Loading detection model from {model_path}")
        model = joblib.load(model_path)
        encoder = joblib.load(ENCODER_PATH)
        X, _ = _build_feature_matrix(df, fit_encoder=False, encoder=encoder)
    else:
        logger.info("No saved model found — training new Isolation Forest.")
        model = train_detection_model(df, contamination=contamination)
        encoder = joblib.load(ENCODER_PATH)
        X, _ = _build_feature_matrix(df, fit_encoder=False, encoder=encoder)

    raw_scores = model.decision_function(X)   # more negative = more anomalous
    predictions = model.predict(X)            # -1 = anomaly, 1 = normal

    result_df = df[["transaction_id", "sender_id", "receiver_id",
                     "amount", "timestamp", "transaction_type",
                     "is_cross_border"]].copy()

    result_df["anomaly_score"] = raw_scores
    result_df["is_flagged"] = predictions == -1
    result_df["flag_reason"] = result_df.apply(
        lambda row: _build_flag_reason(row, df), axis=1
    )

    flagged = result_df[result_df["is_flagged"]].copy().reset_index(drop=True)
    logger.info(
        f"Flagged {len(flagged):,} of {len(df):,} transactions as suspicious."
    )
    return flagged


def _build_feature_matrix(
    df: pd.DataFrame,
    fit_encoder: bool = True,
    encoder: Optional[OneHotEncoder] = None,
):
    """Build the numeric feature matrix used for Isolation Forest."""
    numeric = df[NUMERIC_FEATURES].values

    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        categorical = encoder.fit_transform(df[CATEGORICAL_FEATURES])
    else:
        categorical = encoder.transform(df[CATEGORICAL_FEATURES])

    X = np.hstack([numeric, categorical])
    return X, encoder


def _build_flag_reason(row: pd.Series, original_df: pd.DataFrame) -> str:
    """
    Produce a human-readable string explaining the primary flag trigger.
    Used to populate the flag_reason column.
    """
    reasons = []

    # High amount relative to dataset median
    median_amount = original_df["amount"].median()
    if row["amount"] > median_amount * 10:
        reasons.append("Unusually high transaction amount")

    # Cross-border
    if row["is_cross_border"]:
        reasons.append("Cross-border transfer")

    # Unusual hour (midnight to 5am)
    # NOTE: hour_of_day not in flagged row — re-derive from timestamp if needed
    # This is a simplified version for initial commit
    if not reasons:
        reasons.append("Statistical outlier detected by Isolation Forest")

    return "; ".join(reasons)