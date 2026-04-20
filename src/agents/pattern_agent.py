"""
pattern_agent.py
----------------
Phase 2: Classify laundering patterns from extracted features.

Input : Feature dict (from feature_agent.py)
Output: List of detected patterns with per-pattern confidence scores

Patterns detected:
    FUNNELING   — many senders, one receiver
    SCATTERING  — one sender, many receivers
    CIRCULAR    — cycles in transaction graph (money returns to origin)
    SMURFING    — structuring: low-variance amounts just below reporting limits
    LAYERING    — many intermediaries, long transaction chains
    ISOLATED    — low connectivity, low risk
"""

import logging

logger = logging.getLogger(__name__)

# Reporting limit threshold (USA: $10,000 CTR threshold)
STRUCTURING_LIMIT = 10_000.0
STRUCTURING_BUFFER = 0.15     # Flag if avg_amount is within 15% below limit


def detect_patterns(features: dict) -> dict:
    """
    Detect laundering patterns from a feature dict.

    Args:
        features: Output from feature_agent.extract_features()

    Returns:
        {
          "account_id"         : str,
          "detected_patterns"  : list[str],
          "pattern_confidence" : dict[str, float],
          "is_isolated"        : bool,
        }
    """
    account_id = features.get("account_id", "UNKNOWN")
    scores: dict[str, float] = {}

    scores["FUNNELING"] = _score_funneling(features)
    scores["SCATTERING"] = _score_scattering(features)
    scores["CIRCULAR"] = _score_circular(features)
    scores["SMURFING"] = _score_smurfing(features)
    scores["LAYERING"] = _score_layering(features)

    # A pattern is "detected" if its confidence score exceeds 0.5
    DETECTION_THRESHOLD = 0.5
    detected = [p for p, s in scores.items() if s >= DETECTION_THRESHOLD]

    is_isolated = _is_isolated(features)
    if is_isolated and not detected:
        detected = ["ISOLATED"]
        scores["ISOLATED"] = 1.0

    logger.info(
        f"Account {account_id}: detected patterns = {detected}"
    )

    return {
        "account_id": account_id,
        "detected_patterns": detected,
        "pattern_confidence": {k: round(v, 4) for k, v in scores.items()},
        "is_isolated": is_isolated,
    }


# ---------------------------------------------------------------------------
# Pattern scoring functions — each returns a float in [0.0, 1.0]
# ---------------------------------------------------------------------------

def _score_funneling(f: dict) -> float:
    """
    Funneling: many senders → one receiver.
    High in_degree, low out_degree, high in_out_ratio.
    """
    score = 0.0
    if f.get("in_degree", 0) >= 5:
        score += 0.4
    if f.get("in_out_ratio", 0) >= 3.0:
        score += 0.4
    if f.get("betweenness", 0) > 0.3:
        score += 0.2
    return min(score, 1.0)


def _score_scattering(f: dict) -> float:
    """
    Scattering: one sender → many receivers.
    High out_degree, low in_degree, low in_out_ratio.
    """
    score = 0.0
    if f.get("out_degree", 0) >= 5:
        score += 0.4
    if f.get("in_out_ratio", 1.0) <= 0.33:
        score += 0.4
    if f.get("betweenness", 0) > 0.3:
        score += 0.2
    return min(score, 1.0)


def _score_circular(f: dict) -> float:
    """
    Circular flow: money returns to origin account.
    Direct indicator: has_cycle.
    """
    if f.get("has_cycle", False):
        return 1.0
    return 0.0


def _score_smurfing(f: dict) -> float:
    """
    Smurfing / Structuring: many small transactions just below reporting limit.
    Low amount variance + avg_amount close to but below STRUCTURING_LIMIT.
    """
    avg = f.get("avg_amount", 0.0)
    std = f.get("amount_std", 1.0)

    if avg <= 0:
        return 0.0

    lower_bound = STRUCTURING_LIMIT * (1 - STRUCTURING_BUFFER)
    is_near_limit = lower_bound <= avg < STRUCTURING_LIMIT
    is_low_variance = std < avg * 0.15  # std < 15% of mean

    score = 0.0
    if is_near_limit:
        score += 0.6
    if is_low_variance:
        score += 0.4
    return min(score, 1.0)


def _score_layering(f: dict) -> float:
    """
    Layering: funds passed through many intermediaries across long chains.
    """
    score = 0.0
    if f.get("num_intermediaries", 0) >= 3:
        score += 0.5
    if f.get("max_path_length", 0) >= 3:
        score += 0.3
    if f.get("txn_velocity", 0) >= 5:
        score += 0.2
    return min(score, 1.0)


def _is_isolated(f: dict) -> bool:
    """
    True if the subgraph is very small and sparsely connected —
    likely a single suspicious transaction with no broader network.
    """
    return (
        f.get("subgraph_node_count", 0) < 3
        and f.get("in_degree", 0) <= 1
        and f.get("out_degree", 0) <= 1
    )