"""
risk_agent.py
-------------
Phase 2: Compute a weighted risk score and assign a risk tier.

Input:
    features        : dict (from feature_agent.py)
    pattern_result  : dict (from pattern_agent.py)
    anomaly_score   : float (raw Isolation Forest score for this account's
                      most anomalous transaction)

Output:
    {
      "account_id"      : str,
      "risk_score"      : float,   # 0.0 – 1.0
      "risk_tier"       : str,     # "LOW" | "MEDIUM" | "HIGH"
      "routing_decision": str,     # "EXIT" | "INVESTIGATE"
      "score_components": dict,    # breakdown of each weighted term
    }

Weights (must sum to 1.0):
    anomaly    : 0.25
    pattern    : 0.30
    velocity   : 0.15
    cross_border: 0.15
    structuring: 0.15
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Weight configuration
WEIGHTS = {
    "anomaly": 0.25,
    "pattern": 0.30,
    "velocity": 0.15,
    "cross_border": 0.15,
    "structuring": 0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Risk tier thresholds (configurable via env vars)
RISK_LOW_THRESHOLD = float(os.getenv("RISK_LOW_THRESHOLD", "0.35"))
RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", "0.65"))


def compute_risk_score(
    features: dict,
    pattern_result: dict,
    anomaly_score: float,
) -> dict:
    """
    Compute a composite risk score and determine routing.

    Args:
        features:      Feature dict from feature_agent.extract_features()
        pattern_result: Pattern dict from pattern_agent.detect_patterns()
        anomaly_score: Raw Isolation Forest decision score (typically negative).
                       Will be rescaled to [0, 1] where 1 = most anomalous.

    Returns:
        Risk scoring result dict.
    """
    account_id = features.get("account_id", "UNKNOWN")

    components = _compute_components(features, pattern_result, anomaly_score)

    risk_score = sum(WEIGHTS[k] * v for k, v in components.items())
    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    risk_tier = _assign_tier(risk_score)
    routing_decision = "EXIT" if risk_tier == "LOW" else "INVESTIGATE"

    logger.info(
        f"Account {account_id}: risk_score={risk_score:.3f}, "
        f"tier={risk_tier}, routing={routing_decision}"
    )

    return {
        "account_id": account_id,
        "risk_score": round(risk_score, 4),
        "risk_tier": risk_tier,
        "routing_decision": routing_decision,
        "score_components": {k: round(v, 4) for k, v in components.items()},
    }


def _compute_components(
    features: dict,
    pattern_result: dict,
    anomaly_score: float,
) -> dict[str, float]:
    """
    Compute each normalised component score (each in [0, 1]).
    """
    # 1. Anomaly score — Isolation Forest scores are typically in [-0.5, 0.5]
    #    More negative = more anomalous. Rescale so -0.5 → 1.0, 0.5 → 0.0
    anomaly_norm = float(np.clip((-anomaly_score + 0.5) / 1.0, 0.0, 1.0))

    # 2. Pattern score — 0.3 per unique non-ISOLATED pattern, capped at 1.0
    non_isolated = [
        p for p in pattern_result.get("detected_patterns", [])
        if p != "ISOLATED"
    ]
    pattern_norm = min(len(non_isolated) * 0.3, 1.0)

    # 3. Transaction velocity — normalise against a ceiling of 20 txns/day
    velocity = features.get("txn_velocity", 0.0)
    velocity_norm = float(np.clip(velocity / 20.0, 0.0, 1.0))

    # 4. Cross-border ratio — already in [0, 1]
    cross_border_norm = float(np.clip(
        features.get("cross_border_ratio", 0.0), 0.0, 1.0
    ))

    # 5. Structuring (smurfing) — binary: SMURFING detected → 1.0
    structuring_norm = (
        1.0 if "SMURFING" in pattern_result.get("detected_patterns", [])
        else 0.0
    )

    return {
        "anomaly": anomaly_norm,
        "pattern": pattern_norm,
        "velocity": velocity_norm,
        "cross_border": cross_border_norm,
        "structuring": structuring_norm,
    }


def _assign_tier(score: float) -> str:
    if score < RISK_LOW_THRESHOLD:
        return "LOW"
    elif score < RISK_HIGH_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"