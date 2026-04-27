# File: src/agents/risk_agent.py - TWO-STAGE CLASSIFIER WITH ENHANCED OUTPUT

import math
import pandas as pd
from datetime import datetime

class RiskAgent:
    """
    Phase 2.4 Risk Scoring Agent - TWO-STAGE CLASSIFIER
    Different logic for different flag reasons
    """

    def __init__(self, global_stats: dict):
        self.global_stats = global_stats

    def compute_risk(self, flagged_row, feature_result, pattern_result, graph_result=None, transaction_id=None):
        """
        Compute risk score and routing decision
        
        Parameters:
        - flagged_row: Single transaction row from flagged dataset
        - feature_result: Output from FeatureAgent
        - pattern_result: Output from PatternAgent  
        - graph_result: Output from GraphAgent (optional, for frontend visualization)
        - transaction_id: Transaction ID (optional, will extract from flagged_row if not provided)
        """
        
        account_id = feature_result["account_id"]
        features = feature_result["features"]
        patterns = pattern_result["detected_patterns"]
        severity_map = pattern_result.get("pattern_severity", {})

        # Extract transaction_id if not provided
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
            flag_type = "ML"
            
        elif is_high_amount:
            # High amount alerts are often false positives - be conservative
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
            flag_type = "HIGH_AMOUNT"
            
        elif is_unusual_hour:
            # Unusual hour alerts are mostly false positives
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
            flag_type = "UNUSUAL_HOUR"
            
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
            flag_type = "DEFAULT"

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

        # Create base result
        result = {
            "transaction_id": transaction_id,
            "account_id": account_id,
            "risk_score": round(risk_score, 3),
            "risk_tier": tier,
            "score_components": {
                "anomaly_score": round(anomaly_score, 4),
                "pattern_score": round(pattern_score, 4),
                "velocity_score": round(velocity_score, 4),
                "cross_border_score": round(cross_border_score, 4),
                "amount_score": round(amount_score, 4),
                "time_risk": round(time_risk, 4),
                "flag_type": flag_type
            },
            "routing_decision": routing,
            "pattern_confidence": pattern_result.get("pattern_confidence", {}),
            "detected_patterns": patterns
        }
        
        # Add enhanced fields for Phase 3-5 (if graph_result is provided)
        if graph_result is not None:
            # Get the actual graph object
            G = graph_result.get("graph")
            
            # Prepare graph data for frontend
            graph_data = {"nodes": [], "edges": []}
            if G is not None:
                # Limit nodes to 50 for performance
                nodes = list(G.nodes())[:50]
                graph_data["nodes"] = [
                    {"id": str(node), "label": str(node)[:8], "degree": G.degree(node)} 
                    for node in nodes
                ]
                
                # Limit edges to 100 for performance
                edges = list(G.edges(data=True))[:100]
                graph_data["edges"] = [
                    {
                        "source": str(u), 
                        "target": str(v), 
                        "amount": data.get("amount", 0),
                        "timestamp": str(data.get("timestamp", ""))
                    }
                    for u, v, data in edges
                ]
            
            # Add Phase 3-5 fields
            result.update({
                "investigation_id": f"INV_{transaction_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "investigation_timestamp": datetime.now().isoformat(),
                "pipeline_version": "2.0",
                "sar_ready": tier in ["HIGH", "MEDIUM"],
                "sar_narrative": None,
                "sar_generated_at": None,
                "graph_data": graph_data,
                "routing_reason": self.generate_routing_reason(routing, risk_score, patterns),
                "processing_time_ms": 0,
                "agent_timeline": {
                    "graph_agent": 0,
                    "feature_agent": 0,
                    "pattern_agent": 0,
                    "risk_agent": 0
                }
            })
        
        return result

    def generate_routing_reason(self, routing_decision, risk_score, patterns):
        """Generate human-readable reason for routing decision"""
        if routing_decision == "INVESTIGATE":
            pattern_list = patterns[:3] if patterns else ["no specific patterns"]
            return f"High risk score ({risk_score:.2f}) with patterns: {', '.join(pattern_list)}"
        else:
            return f"Risk score ({risk_score:.2f}) below threshold, no actionable patterns"

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
        """Safely get value from row (dict or object)"""
        try:
            # Try dictionary access
            return row[key]
        except (TypeError, KeyError):
            try:
                # Try attribute access
                return getattr(row, key)
            except AttributeError:
                return default