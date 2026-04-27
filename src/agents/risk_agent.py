# File: src/agents/risk_agent.py - TWO-STAGE CLASSIFIER

import math

class RiskAgent:
    """
    Phase 2.4 Risk Scoring Agent - TWO-STAGE CLASSIFIER
    Different logic for different flag reasons
    """

    def __init__(self, global_stats: dict):
        self.global_stats = global_stats

    def compute_risk(self, flagged_row, feature_result, pattern_result):

        account_id = feature_result["account_id"]
        features = feature_result["features"]
        patterns = pattern_result["detected_patterns"]
        severity_map = pattern_result.get("pattern_severity", {})

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

        # ----------------------------
        # STAGE 1: Determine flag type
        # ----------------------------
        is_ml_detection = "Random Forest" in flag_reason or "anomaly" in flag_reason.lower()
        is_high_amount = "High amount" in flag_reason
        is_unusual_hour = "Unusual transaction hour" in flag_reason

        # ----------------------------
        # STAGE 2: Different scoring logic per flag type
        # ----------------------------
        
        if is_ml_detection:
            # ML detections are more reliable - use aggressive scoring
            risk_score = (
                0.40 * anomaly_score +      # Anomaly most important
                0.35 * pattern_score +
                0.10 * velocity_score +
                0.05 * cross_border_score +
                0.05 * amount_score +
                0.05 * time_risk
            )
            
            # Bonuses for ML detections
            if "CIRCULAR_FLOW" in patterns:
                risk_score += 0.08
            if anomaly_score > 0.6:
                risk_score += 0.05
                
            # Lower thresholds for ML detections
            high_threshold = 0.48
            med_threshold = 0.35
            
        elif is_high_amount:
            # High amount alerts are often false positives - be conservative
            # Anomaly score must be significant
            if anomaly_score < 0.15:
                risk_score = 0.0  # Immediate EXIT
            else:
                risk_score = (
                    0.50 * anomaly_score +      # Anomaly dominates
                    0.20 * pattern_score +
                    0.10 * velocity_score +
                    0.05 * cross_border_score +
                    0.10 * amount_score +
                    0.05 * time_risk
                )
            
            # High thresholds for high amount alerts
            high_threshold = 0.65
            med_threshold = 0.45
            
        elif is_unusual_hour:
            # Unusual hour alerts are mostly false positives - very conservative
            # Must have strong patterns AND decent anomaly
            if anomaly_score < 0.20 or len(valid_patterns) < 2:
                risk_score = 0.0  # Immediate EXIT
            else:
                risk_score = (
                    0.45 * anomaly_score +
                    0.40 * pattern_score +
                    0.05 * velocity_score +
                    0.05 * cross_border_score +
                    0.05 * amount_score
                )
            
            # Very high thresholds
            high_threshold = 0.70
            med_threshold = 0.55
            
        else:
            # Default fallback
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

        return {
            "account_id": account_id,
            "risk_score": round(risk_score, 3),
            "risk_tier": tier,
            "score_components": {
                "anomaly_score": round(anomaly_score, 4),
                "pattern_score": round(pattern_score, 4),
                "flag_type": "ML" if is_ml_detection else ("HIGH_AMOUNT" if is_high_amount else "UNUSUAL_HOUR"),
                "routing_decision": routing
            },
            "routing_decision": routing
        }

    def normalize_anomaly(self, x):
        """Original working normalization"""
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
        except:
            try:
                return getattr(row, key)
            except:
                return default