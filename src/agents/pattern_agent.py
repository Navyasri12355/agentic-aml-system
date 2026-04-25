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

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Main Detection
    # ---------------------------------------------------------
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

        # -------------------------------------------------
        # 2. SCATTERING
        # One sender -> many receivers
        # -------------------------------------------------
        if out_degree > 5 and ratio < 0.33:
            patterns.append("SCATTERING")

            score1 = min(out_degree / 15.0, 1.0)
            score2 = min((0.33 - ratio) / 0.33, 1.0)

            confidence["SCATTERING"] = round((score1 + score2) / 2, 3)

        # -------------------------------------------------
        # 3. CIRCULAR FLOW
        # -------------------------------------------------
        if has_cycle:
            patterns.append("CIRCULAR_FLOW")
            confidence["CIRCULAR_FLOW"] = 0.90

        # -------------------------------------------------
        # 4. SMURFING / STRUCTURING
        # low std + avg just below threshold
        # -------------------------------------------------
        if avg_amount < 10000 and amount_std < 1500 and node_count > 5:
            patterns.append("SMURFING")

            score1 = 1 - min(avg_amount / 10000.0, 1.0)
            score2 = 1 - min(amount_std / 1500.0, 1.0)

            confidence["SMURFING"] = round((score1 + score2) / 2, 3)

        # -------------------------------------------------
        # 5. LAYERING
        # many intermediaries + long paths
        # -------------------------------------------------
        if num_intermediaries > 3 and max_path > 2:
            patterns.append("LAYERING")

            score1 = min(num_intermediaries / 10.0, 1.0)
            score2 = min(max_path / 5.0, 1.0)

            confidence["LAYERING"] = round((score1 + score2) / 2, 3)

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