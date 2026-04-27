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
            "LARGE_VALUE": 0.85,
            "DRAINING": 0.85,
            "RAPID_FLOW": 0.6,
            "HUB_ACTIVITY": 0.65
        }

    def detect_patterns(self, feature_result: dict):
        account_id = feature_result["account_id"]
        node_count = feature_result["subgraph_node_count"]

        f = feature_result["features"]

        patterns = []
        confidence = {}

        in_degree = f["in_degree"]
        out_degree = f["out_degree"]
        ratio = f["in_out_ratio"]
        has_cycle = f["has_cycle"]
        avg_amount = f["avg_amount"]
        amount_std = f["amount_std"]
        num_intermediaries = f["num_intermediaries"]
        max_path = f["max_path_length"]

        # -------------------------------------------------
        # 1. FUNNELING
        # Many senders -> one receiver
        # -------------------------------------------------
        if in_degree > 5 and ratio > 3.0:
            patterns.append("FUNNELING")

            score1 = min(in_degree / 15.0, 1.0)
            score2 = min(ratio / 6.0, 1.0)

            confidence["FUNNELING"] = round((score1 + score2) / 2, 3)

        # -------------------------
        # SCATTERING
        # -------------------------
        if out_degree >= 3 and f["in_out_ratio"] < 0.5:
            patterns.append("SCATTERING")
            confidence["SCATTERING"] = 0.8

        # -------------------------
        # FUNNELING
        # -------------------------
        if in_degree >= 3 and f["in_out_ratio"] > 2.0:
            patterns.append("FUNNELING")
            confidence["FUNNELING"] = 0.8

        # -------------------------------------------------
        # 3. CIRCULAR FLOW
        # -------------------------------------------------
        if has_cycle:
            patterns.append("CIRCULAR_FLOW")
            confidence["CIRCULAR_FLOW"] = 0.90

        # -------------------------
        # LAYERING
        # -------------------------
        if num_intermediaries > 3 and f["max_path_length"] > 3:
            patterns.append("LAYERING")
            confidence["LAYERING"] = 0.7

        # -------------------------
        # HIGH VELOCITY
        # -------------------------
        if velocity > velocity_mean + 1.5 * velocity_std:
            patterns.append("HIGH_VELOCITY")
            confidence["HIGH_VELOCITY"] = 0.7

        # -------------------------
        # SMURFING
        # -------------------------
        p50_amount = self.global_stats.get("p50_amount", amount_mean)
        if ( amount_std < 0.5 * amount_std_global and avg_amount < p50_amount and out_degree >= 3):
            patterns.append("SMURFING")

            score1 = 1 - min(avg_amount / 10000.0, 1.0)
            score2 = 1 - min(amount_std / 1500.0, 1.0)

            confidence["SMURFING"] = round((score1 + score2) / 2, 3)

        # -------------------------
        # LARGE VALUE TRANSFERS
        # -------------------------
        p95_amount = self.global_stats.get("p95_amount", 100000)

        if avg_amount > p95_amount:
            patterns.append("LARGE_VALUE")
            confidence["LARGE_VALUE"] = 0.85

        # -------------------------
        # DRAINING (🔥 NEW)
        # -------------------------
        if f["net_flow"] < -0.8 * max(f["total_sent"], 1):
            patterns.append("DRAINING")
            confidence["DRAINING"] = 0.8

        # -------------------------
        # RAPID FLOW (🔥 NEW)
        # -------------------------
        if in_degree >= 1 and out_degree >= 1 and velocity > velocity_mean:
            patterns.append("RAPID_FLOW")
            confidence["RAPID_FLOW"] = 0.7

        # -------------------------
        # HUB ACTIVITY (🔥 NEW)
        # -------------------------
        if (in_degree + out_degree) >= 6:
            patterns.append("HUB_ACTIVITY")
            confidence["HUB_ACTIVITY"] = 0.7

        # -------------------------
        # 7. COMBO
        # -------------------------
        if "SCATTERING" in patterns and "CIRCULAR_FLOW" in patterns:
            patterns.append("HIGH_RISK_COMBO")
            confidence["HIGH_RISK_COMBO"] = 0.95

        # -------------------------------------------------
        # 6. ISOLATED / LOW RISK
        # -------------------------------------------------
        is_isolated = False

        if (
            node_count < 3 and
            in_degree <= 1 and
            out_degree <= 1
        ):
            patterns.append("ISOLATED_LOW_RISK")
            confidence["ISOLATED_LOW_RISK"] = 0.95
            is_isolated = True

        # -------------------------------------------------
        # If nothing matched
        # -------------------------------------------------
        if len(patterns) == 0:
            patterns.append("UNCLASSIFIED")
            confidence["UNCLASSIFIED"] = 0.30

        # -------------------------------------------------
        # Final Output
        # -------------------------------------------------
        return {
            "account_id": account_id,
            "detected_patterns": patterns,
            "pattern_confidence": confidence,
            "is_isolated": is_isolated
        }