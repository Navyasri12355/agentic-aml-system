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


    def __init__(self, global_stats: dict):
        self.global_stats = global_stats
        self.w1 = 0.20
        self.w2 = 0.40
        self.w3 = 0.15
        self.w4 = 0.10
        self.w5 = 0.15

    def compute_risk(self, flagged_row, feature_result, pattern_result):

        account_id = feature_result["account_id"]
        features = feature_result["features"]
        patterns = pattern_result["detected_patterns"]

        severity_map = pattern_result.get("pattern_severity", {})

        raw_anomaly = self.safe_get(flagged_row, "anomaly_score", 1.0)
        anomaly_score = self.normalize_anomaly(raw_anomaly)
        amount = self.safe_get(flagged_row, "amount", 0)
        hour = self.safe_get(flagged_row, "hour", None)

        # ----------------------------
        # 🔥 IMPROVED pattern score
        # weighted severity + confidence
        # ----------------------------
        valid_patterns = [
            p for p in patterns
            if p not in ["UNCLASSIFIED", "ISOLATED_LOW_RISK"]
        ]

        pattern_score = 0.0
        for p in valid_patterns:
            pattern_score += severity_map.get(p, 0.3)

        pattern_score = min(pattern_score, 1.0)

        # ----------------------------
        #velocity
        # ----------------------------
        velocity_score = min(features["txn_velocity"] / 10.0, 1.0)

        # ----------------------------
        #cross border
        # ----------------------------
        cross_border_score = self.safe_get(flagged_row, "is_cross_border", 0)

        # ----------------------------
        #structuring
        # ----------------------------
        structuring_score = 1.0 if "SMURFING" in patterns else 0.0

        # ----------------------------
        #high amount detection
        # ----------------------------
        p99_amount = self.global_stats.get("p99_amount", 1e6)
        high_amount_flag = 1.0 if amount > p99_amount else 0.0

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
            0.30 * anomaly_score +        # 🔥 stronger anomaly
            0.25 * pattern_score +
            0.15 * velocity_score +
            0.10 * cross_border_score +
            0.10 * structuring_score +
            0.10 * high_amount_flag +     # 🔥 NEW
            0.10 * time_risk              # 🔥 NEW
        )

        risk_score = round(min(max(risk_score, 0), 1), 4)

        # 🔥 FIXED routing (IMPORTANT)
        if risk_score < 0.3:
            tier = "LOW"
            routing = "EXIT"

        elif risk_score < 0.6:
            tier = "MEDIUM"
            routing = "INVESTIGATE"   # FIXED

        else:
            tier = "HIGH"
            routing = "INVESTIGATE"

        return {
            "account_id": account_id,
            "risk_score": risk_score,
            "risk_tier": tier,
            "score_components": {
                "anomaly_score": round(anomaly_score, 4),
                "pattern_score": round(pattern_score, 4),
                "velocity_score": round(velocity_score, 4),
                "cross_border_score": round(cross_border_score, 4),
                "structuring_score": round(structuring_score, 4)
            },
            "routing_decision": routing
        }

    def normalize_anomaly(self, x):
        try:
            x = float(x)
        except:
            return 0.0
        p95 = self.global_stats.get("anomaly_p95", 1.0)
        if p95 == 0:
            return 0.0
        return min(x / p95, 1.0)
    
    def safe_get(self, row, key, default):
        try:
            return row[key]
        except:
            try:
                return getattr(row, key)
            except:
                return default