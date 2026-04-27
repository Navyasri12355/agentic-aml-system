# File: src/agents/risk_agent.py

import math


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
        pattern_score = 0.0
        if patterns == ["UNCLASSIFIED"]:
            pattern_score = 0.1
        else:
            pattern_score = sum(severity_map.get(p, 0.3) for p in valid_patterns)
            pattern_score = min(pattern_score, 1.0)

        # ----------------------------
        #velocity
        # ----------------------------
        v = features["txn_velocity"]
        v_mean = self.global_stats.get("velocity_mean", 1)
        v_std = self.global_stats.get("velocity_std", 1)
        v_p95 = self.global_stats.get("velocity_p95", v_mean + 2 * v_std)
        velocity_score = min(1.0, v / (v_p95 + 1e-6))

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

        # ----------------------------
        #high amount detection
        # ----------------------------
        p99_amount = self.global_stats.get("p99_amount", 1e6)
        high_amount_flag = math.log1p(amount) / math.log1p(p99_amount + 1e-6)
        high_amount_flag = min(1.0, high_amount_flag)

        # ----------------------------
        #time anomaly
        # ----------------------------
        if hour is not None and hour in [0,1,2,3,4]:
            time_risk = 1.0
        else:
            time_risk = 0.0
        # ----------------------------
        #risk score
        # ----------------------------
        risk_score = (
            0.30 * max(anomaly_score, 0.001) +
            0.40 * pattern_score +
            0.10 * velocity_score +
            0.05 * cross_border_score +
            0.05 * structuring_score +
            0.05 * high_amount_flag +
            0.05 * time_risk
        )
        risk_score = round(min(max(risk_score, 0), 1), 2)

        if "CIRCULAR_FLOW" in patterns and "LARGE_VALUE" in patterns:
            risk_score += 0.10

        if anomaly_score > 0.8 and pattern_score > 0.7:
            risk_score += 0.10
        
        risk_score = min(1.0, risk_score)

        # 🔥 FIXED routing (IMPORTANT)
        if risk_score < 0.35:
            tier = "LOW"
            routing = "EXIT"

        elif risk_score < 0.55:
            tier = "MEDIUM"
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
            return 0.0
        p1 = self.global_stats.get("anomaly_p1", 0.0)
        p95 = self.global_stats.get("anomaly_p95", 1.0)
        if p95 - p1 == 0:
            return 0.0
        # robust scaling (winsorized min-max)
        x = max(min(x, p95), p1)
        norm = (x - p1) / (p95 - p1)
        return round(norm, 4)
    
    def safe_get(self, row, key, default):
        try:
            return row[key]
        except:
            try:
                return getattr(row, key)
            except:
                return default