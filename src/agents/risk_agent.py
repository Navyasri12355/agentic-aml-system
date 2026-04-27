# File: src/agents/risk_agent.py - TWO-STAGE CLASSIFIER (MINIMAL)

import math
from datetime import datetime

class RiskAgent:
    """
    Phase 2.4 Risk Scoring Agent - TWO-STAGE CLASSIFIER
    """

    def __init__(self, global_stats: dict):
        self.global_stats = global_stats

    def compute_risk(self, flagged_row, feature_result, pattern_result, graph_result=None, transaction_id=None):
        
        account_id = feature_result["account_id"]
        features = feature_result["features"]
        patterns = pattern_result["detected_patterns"]
        severity_map = pattern_result.get("pattern_severity", {})

        if transaction_id is None:
            transaction_id = self.safe_get(flagged_row, "transaction_id", f"UNKNOWN_{account_id}")
        
        raw_anomaly = self.safe_get(flagged_row, "anomaly_score", 1.0)
        anomaly_score = self.normalize_anomaly(raw_anomaly)
        amount = self.safe_get(flagged_row, "amount", 0)
        hour = self.safe_get(flagged_row, "hour_of_day", None)
        flag_reason = self.safe_get(flagged_row, "flag_reason", "")

        # Extract features
        v = features.get("txn_velocity", 0)
        v_p95 = self.global_stats.get("velocity_p95", 10)
        velocity_score = min(1.0, v / (v_p95 + 1e-6))
        cross_border_score = self.safe_get(flagged_row, "is_cross_border", 0)
        
        p99_amount = self.global_stats.get("p99_amount", 1e6)
        amount_score = min(1.0, math.log1p(amount) / math.log1p(p99_amount + 1e-6))
        
        if hour is not None and hour in [0, 1, 2, 3, 4]:
            time_risk = 1.0
        else:
            time_risk = 0.0

        # Pattern score
        valid_patterns = [p for p in patterns if p not in ["UNCLASSIFIED", "ISOLATED_LOW_RISK"]]
        if valid_patterns:
            pattern_score = max(severity_map.get(p, 0.3) for p in valid_patterns)
        else:
            pattern_score = 0.1

        # STAGE 1: Determine flag type
        is_ml_detection = "Random Forest" in flag_reason or "anomaly" in flag_reason.lower()
        is_high_amount = "High amount" in flag_reason
        is_unusual_hour = "Unusual transaction hour" in flag_reason

        # STAGE 2: Different scoring logic per flag type
        if is_ml_detection:
            risk_score = (
                0.40 * anomaly_score +
                0.35 * pattern_score +
                0.10 * velocity_score +
                0.05 * cross_border_score +
                0.05 * amount_score +
                0.05 * time_risk
            )
            if "CIRCULAR_FLOW" in patterns:
                risk_score += 0.08
            if anomaly_score > 0.6:
                risk_score += 0.05
            high_threshold = 0.48
            med_threshold = 0.35
            
        elif is_high_amount:
            if anomaly_score < 0.15:
                risk_score = 0.0
            else:
                risk_score = (
                    0.50 * anomaly_score +
                    0.20 * pattern_score +
                    0.10 * velocity_score +
                    0.05 * cross_border_score +
                    0.10 * amount_score +
                    0.05 * time_risk
                )
            high_threshold = 0.65
            med_threshold = 0.45
            
        elif is_unusual_hour:
            if anomaly_score < 0.20 or len(valid_patterns) < 2:
                risk_score = 0.0
            else:
                risk_score = (
                    0.45 * anomaly_score +
                    0.40 * pattern_score +
                    0.05 * velocity_score +
                    0.05 * cross_border_score +
                    0.05 * amount_score
                )
            high_threshold = 0.70
            med_threshold = 0.55
            
        else:
            risk_score = (
                0.30 * anomaly_score +
                0.40 * pattern_score +
                0.10 * velocity_score +
                0.05 * cross_border_score +
                0.05 * amount_score +
                0.05 * time_risk
            )
            high_threshold = 0.55
            med_threshold = 0.35

        risk_score = min(1.0, max(0.0, risk_score))

        # Apply thresholds
        if risk_score < med_threshold:
            tier = "LOW"
            routing = "EXIT"
        elif risk_score < high_threshold:
            tier = "MEDIUM"
            routing = "EXIT"
        else:
            tier = "HIGH"
            routing = "INVESTIGATE"

        # MINIMAL RESULT - only essential fields
        result = {
            "transaction_id": transaction_id,  # REQUIRED for Phase 3 merge
            "account_id": account_id,
            "risk_score": round(risk_score, 3),
            "risk_tier": tier,
            "routing_decision": routing,
            "detected_patterns": patterns,
            "score_components": {
                "anomaly_score": round(anomaly_score, 4),
                "pattern_score": round(pattern_score, 4)
            }
        }
        
        # ONLY add graph_data if provided (for Phase 5 frontend)
        if graph_result is not None:
            G = graph_result.get("graph")
            if G is not None:
                result["graph_data"] = {
                    "nodes": [
                        {"id": str(node), "label": str(node)[:8]} 
                        for node in list(G.nodes())[:50]
                    ],
                    "edges": [
                        {"source": str(u), "target": str(v), "amount": data.get("amount", 0)}
                        for u, v, data in list(G.edges(data=True))[:100]
                    ]
                }
        
        return result

    def normalize_anomaly(self, x):
        try:
            x = float(x)
        except:
            return 0.0
        
        p1 = self.global_stats.get("anomaly_p1", 0.0)
        p95 = self.global_stats.get("anomaly_p95", 1.0)
        
        if p95 - p1 == 0:
            return 0.0
            
        x = max(min(x, p95), p1)
        norm = ((x - p1) / (p95 - p1)) ** 0.7
        return round(norm, 4)
    
    def safe_get(self, row, key, default):
        try:
            return row[key]
        except (TypeError, KeyError):
            try:
                return getattr(row, key)
            except AttributeError:
                return default