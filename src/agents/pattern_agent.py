# File: src/agents/pattern_agent.py

class PatternAgent:
    """
    Phase 2.3 Pattern Detection Agent

    Input:
        feature_result from FeatureAgent.extract_features()

    Output:
        {
          "account_id": str,
          "detected_patterns": [...],
          "pattern_confidence": {...},
          "is_isolated": bool
        }
    """

    def __init__(self, global_stats: dict):
        self.global_stats = global_stats

        # 🔥 severity map (IMPORTANT FIX)
        self.pattern_severity = {
            "SCATTERING": 0.6,
            "FUNNELING": 0.7,
            "CIRCULAR_FLOW": 0.9,
            "LAYERING": 0.8,
            "HIGH_VELOCITY": 0.7,
            "SMURFING": 0.95,
            "HIGH_RISK_COMBO": 1.0,
            "ISOLATED_LOW_RISK": 0.2,
            "UNCLASSIFIED": 0.1,
            "LARGE_VALUE": 0.85
        }

    def detect_patterns(self, feature_result: dict):

        account_id = feature_result["account_id"]
        node_count = feature_result["subgraph_node_count"]
        f = feature_result["features"]

        patterns = []
        confidence = {}
        severity_scores = {}

        sender_mean = self.global_stats["txn_per_sender_mean"]
        sender_std = self.global_stats["txn_per_sender_std"]

        receiver_mean = self.global_stats["txn_per_receiver_mean"]
        receiver_std = self.global_stats["txn_per_receiver_std"]

        amount_mean = self.global_stats["amount_mean"]
        amount_std_global = self.global_stats["amount_std"]

        velocity_mean = self.global_stats.get("velocity_mean", sender_mean)
        velocity_std = self.global_stats.get("velocity_std", sender_std)

        in_degree = f["in_degree"]
        out_degree = f["out_degree"]
        has_cycle = f["has_cycle"]
        avg_amount = f["avg_amount"]
        amount_std = f["amount_std"]
        num_intermediaries = f["num_intermediaries"]
        velocity = f["txn_velocity"]

        # -------------------------
        # SCATTERING
        # -------------------------
        if out_degree > 5 and f["in_out_ratio"] < 0.33:
            patterns.append("SCATTERING")
            confidence["SCATTERING"] = 0.8

        # -------------------------
        # FUNNELING
        # -------------------------
        if in_degree > 5 and f["in_out_ratio"] > 3.0:
            patterns.append("FUNNELING")
            confidence["FUNNELING"] = 0.8

        # -------------------------
        # CIRCULAR FLOW
        # -------------------------
        if has_cycle:
            patterns.append("CIRCULAR_FLOW")
            confidence["CIRCULAR_FLOW"] = 0.9

        # -------------------------
        # LAYERING
        # -------------------------
        if num_intermediaries > 3 and f["max_path_length"] > 3:
            patterns.append("LAYERING")
            confidence["LAYERING"] = 0.7

        # -------------------------
        # HIGH VELOCITY
        # -------------------------
        if velocity > 10:
            patterns.append("HIGH_VELOCITY")
            confidence["HIGH_VELOCITY"] = 0.7

        # -------------------------
        # SMURFING
        # -------------------------
        if (amount_std < 0.3 * self.global_stats["amount_std"] and avg_amount < 5000 and out_degree > 5 ):
            patterns.append("SMURFING")
            confidence["SMURFING"] = 0.9

        # -------------------------
        # LARGE VALUE TRANSFERS (🔥 NEW)
        # -------------------------
        p95_amount = self.global_stats.get("p95_amount", 100000)

        if avg_amount > p95_amount:
            patterns.append("LARGE_VALUE")
            confidence["LARGE_VALUE"] = 0.85

        # -------------------------
        # 7. COMBO
        # -------------------------
        if "SCATTERING" in patterns and "CIRCULAR_FLOW" in patterns:
            patterns.append("HIGH_RISK_COMBO")
            confidence["HIGH_RISK_COMBO"] = 0.95

        # -------------------------
        # 8. ISOLATED
        # -------------------------
        is_isolated = False
        if node_count < 3 and in_degree <= 1 and out_degree <= 1:
            patterns.append("ISOLATED_LOW_RISK")
            confidence["ISOLATED_LOW_RISK"] = 0.95
            is_isolated = True

        if len(patterns) == 0:
            patterns.append("UNCLASSIFIED")
            confidence["UNCLASSIFIED"] = 0.3

        # 🔥 attach severity
        for p in patterns:
            severity_scores[p] = self.pattern_severity.get(p, 0.3)

        return {
            "account_id": account_id,
            "detected_patterns": patterns,
            "pattern_confidence": confidence,
            "pattern_severity": severity_scores,
            "is_isolated": is_isolated
        }