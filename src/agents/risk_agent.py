# File: src/agents/risk_agent.py

class RiskAgent:
    """
    Phase 2.4 Risk Scoring Agent

    Inputs:
        1. flagged_row      -> one row from flagged_transactions.csv
        2. feature_result   -> output of FeatureAgent
        3. pattern_result   -> output of PatternAgent

    Output:
        {
          "account_id": str,
          "risk_score": float,
          "risk_tier": "LOW" | "MEDIUM" | "HIGH",
          "score_components": {...},
          "routing_decision": "EXIT" | "INVESTIGATE"
        }
    """

    def __init__(self):
        # Recommended weights
        self.w1 = 0.25   # anomaly
        self.w2 = 0.30   # pattern
        self.w3 = 0.15   # velocity
        self.w4 = 0.15   # cross-border
        self.w5 = 0.15   # structuring

    # ---------------------------------------------------------
    # Main scoring function
    # ---------------------------------------------------------
    def compute_risk(
        self,
        flagged_row,
        feature_result: dict,
        pattern_result: dict
    ):
        account_id = feature_result["account_id"]

        features = feature_result["features"]
        patterns = pattern_result["detected_patterns"]

        # -------------------------------------------------
        # 1. anomaly_score_normalized
        # -------------------------------------------------
        # If your phase1 file has anomaly_score column use it.
        # Else fallback = 1.0 because already flagged.
        raw_anomaly = self.safe_get(flagged_row, "anomaly_score", 1.0)

        anomaly_score = self.normalize_anomaly(raw_anomaly)

        # -------------------------------------------------
        # 2. pattern_score
        # 0.3 per unique pattern max 1.0
        # ignore UNCLASSIFIED / LOW_RISK labels
        # -------------------------------------------------
        useful_patterns = [
            p for p in patterns
            if p not in ["UNCLASSIFIED", "ISOLATED_LOW_RISK"]
        ]

        pattern_score = min(len(set(useful_patterns)) * 0.30, 1.0)

        # -------------------------------------------------
        # 3. velocity_score
        # -------------------------------------------------
        txn_velocity = features["txn_velocity"]
        velocity_score = min(txn_velocity / 10.0, 1.0)

        # -------------------------------------------------
        # 4. cross_border_score
        # use phase1 column if available
        # -------------------------------------------------
        cross_border_score = self.safe_get(
            flagged_row,
            "is_cross_border",
            0
        )

        # -------------------------------------------------
        # 5. structuring_score
        # -------------------------------------------------
        structuring_score = 1.0 if "SMURFING" in patterns else 0.0

        # -------------------------------------------------
        # Weighted total
        # -------------------------------------------------
        risk_score = (
            self.w1 * anomaly_score +
            self.w2 * pattern_score +
            self.w3 * velocity_score +
            self.w4 * cross_border_score +
            self.w5 * structuring_score
        )

        risk_score = round(min(max(risk_score, 0.0), 1.0), 4)

        # -------------------------------------------------
        # Tiering
        # -------------------------------------------------
        if risk_score < 0.35:
            risk_tier = "LOW"
            routing = "EXIT"

        elif risk_score < 0.65:
            risk_tier = "MEDIUM"
            routing = "EXIT"

        else:
            risk_tier = "HIGH"
            routing = "INVESTIGATE"

        # -------------------------------------------------
        # Output
        # -------------------------------------------------
        return {
            "account_id": account_id,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "score_components": {
                "anomaly_score": round(anomaly_score, 4),
                "pattern_score": round(pattern_score, 4),
                "velocity_score": round(velocity_score, 4),
                "cross_border_score": round(cross_border_score, 4),
                "structuring_score": round(structuring_score, 4)
            },
            "routing_decision": routing
        }

    # ---------------------------------------------------------
    # Normalize anomaly score to 0–1
    # ---------------------------------------------------------
    def normalize_anomaly(self, x):
        """
        Handles different formats:
        If already 0..1 -> keep.
        If negative/positive score -> squash.
        """
        try:
            x = float(x)
        except:
            return 1.0

        # already probability-like
        if 0 <= x <= 1:
            return x

        # convert arbitrary value to bounded score
        return 1 / (1 + abs(x))

    # ---------------------------------------------------------
    # Safe getter for pandas row / dict
    # ---------------------------------------------------------
    def safe_get(self, row, key, default):
        try:
            return row[key]
        except:
            try:
                return getattr(row, key)
            except:
                return default