from src.agents.risk_agent import compute_risk_score

def test_low_risk():
    features = {"account_id":"ACC_001","txn_velocity":0.5,"cross_border_ratio":0.0,"avg_amount":100.0,"amount_std":10.0,"subgraph_node_count":2,"in_degree":1,"out_degree":1,"in_out_ratio":1.0,"betweenness":0.0,"has_cycle":False,"num_intermediaries":0,"max_path_length":1,"net_flow":0.0,"burst_score":1.0,"total_sent":0.0,"total_received":0.0,"hop_count":1,"subgraph_edge_count":1}
    pattern_result = {"detected_patterns":[],"pattern_confidence":{},"is_isolated":True}
    result = compute_risk_score(features, pattern_result, anomaly_score=0.3)
    assert result["risk_tier"] == "LOW"

def test_high_risk():
    features = {"account_id":"ACC_002","txn_velocity":18.0,"cross_border_ratio":0.9,"avg_amount":9500.0,"amount_std":100.0,"subgraph_node_count":20,"in_degree":15,"out_degree":2,"in_out_ratio":7.5,"betweenness":0.8,"has_cycle":True,"num_intermediaries":5,"max_path_length":4,"net_flow":500000.0,"burst_score":8.0,"total_sent":100000.0,"total_received":600000.0,"hop_count":4,"subgraph_edge_count":40}
    pattern_result = {"detected_patterns":["FUNNELING","CIRCULAR","SMURFING"],"pattern_confidence":{"FUNNELING":0.9,"CIRCULAR":1.0,"SMURFING":0.8},"is_isolated":False}
    result = compute_risk_score(features, pattern_result, anomaly_score=-0.4)
    assert result["risk_tier"] == "HIGH"