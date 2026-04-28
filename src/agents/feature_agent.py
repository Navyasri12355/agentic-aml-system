# File: src/agents/feature_agent.py

import networkx as nx
import numpy as np
import pandas as pd


class FeatureAgent:
    """
    Phase 2.2 Feature Extraction Agent

    Input:
        graph_result from GraphAgent.build_subgraph()

    Output:
        Dictionary of node-level, temporal, and structural features
        for the flagged account.
    """

    def __init__(self, global_stats):
        self.global_stats = global_stats

    # ---------------------------------------------------------
    # Main extractor
    # ---------------------------------------------------------
    def extract_features(self, graph_result: dict):
        G = graph_result["graph"]
        account_id = graph_result["account_id"]
        hop_count = graph_result.get("hop_radius_used", 1)

        # If suspicious node absent
        if account_id not in G.nodes():
            return self.empty_result(account_id, G, hop_count)

        # ----------------------------
        # NODE FEATURES
        # ----------------------------
        in_degree = G.in_degree(account_id)
        out_degree = G.out_degree(account_id)

        in_out_ratio = in_degree / (out_degree + 1)

        # Betweenness (costly but useful)
        betweenness_scores = nx.betweenness_centrality(G)
        betweenness = betweenness_scores.get(account_id, 0.0)

        total_received = 0.0
        total_sent = 0.0

        # Incoming money
        for u, v, data in G.in_edges(account_id, data=True):
            total_received += data.get("amount", 0)

        # Outgoing money
        for u, v, data in G.out_edges(account_id, data=True):
            total_sent += data.get("amount", 0)

        net_flow = total_received - total_sent

        # ----------------------------
        # TEMPORAL FEATURES
        # ----------------------------
        timestamps = []
        amounts = []

        for _, _, data in G.edges(data=True):
            timestamps.append(pd.to_datetime(data["timestamp"]))
            amounts.append(data.get("amount", 0))

        timestamps = sorted(timestamps)
        amounts = np.array(amounts) if amounts else np.array([])

        # ----------------------------
        # TEMPORAL FEATURES (FIXED)
        # ----------------------------

        account_edges = list(G.out_edges(account_id, data=True)) + list(G.in_edges(account_id, data=True))

        timestamps = [pd.to_datetime(d["timestamp"]) for _, _, d in account_edges]
        amounts = [d.get("amount", 0) for _, _, d in account_edges]

        timestamps = sorted(timestamps)
        amounts = np.array(amounts) if amounts else np.array([])

        # ----------------------------
        # velocity (account-level, NOT subgraph-level)
        # ----------------------------
        if len(timestamps) > 1:
            days = max((max(timestamps) - min(timestamps)).days, 1)
            txn_velocity = len(timestamps) / days
        else:
            txn_velocity = float(len(timestamps))

        # CAP velocity (IMPORTANT for stability)
        txn_velocity = min(txn_velocity, 50)

        # ----------------------------
        # burst score (stable version)
        # ----------------------------
        burst_score = self.compute_burst_score(timestamps)

        # ----------------------------
        # amount stats
        # ----------------------------
        avg_amount = float(np.mean(amounts)) if len(amounts) > 0 else 0.0
        amount_std = float(np.std(amounts)) if len(amounts) > 0 else 0.0

        # ----------------------------
        # STRUCTURAL FEATURES
        # ----------------------------

        # cycle detection
        try:
            cycle = nx.find_cycle(G, orientation="original")
            has_cycle = True if cycle else False
        except:
            has_cycle = False

        # longest shortest path from flagged node
        try:
            lengths = nx.single_source_shortest_path_length(G, account_id)
            max_path_length = max(lengths.values()) if lengths else 0
        except:
            max_path_length = 0

        # intermediary nodes
        num_intermediaries = sum(
            1 for n in G.nodes()
            if G.in_degree(n) > 0 and G.out_degree(n) > 0 and n != account_id
        )

        # ----------------------------
        # FINAL OUTPUT
        # ----------------------------
        return {
            "account_id": account_id,
            "subgraph_node_count": G.number_of_nodes(),
            "subgraph_edge_count": G.number_of_edges(),
            "features": {
                "in_degree": int(in_degree),
                "out_degree": int(out_degree),
                "in_out_ratio": float(round(in_out_ratio, 4)),
                "betweenness": float(round(betweenness, 6)),
                "total_received": float(round(total_received, 2)),
                "total_sent": float(round(total_sent, 2)),
                "net_flow": float(round(net_flow, 2)),
                "txn_velocity": float(round(txn_velocity, 4)),
                "burst_score": float(round(burst_score, 4)),
                "avg_amount": float(round(avg_amount, 2)),
                "amount_std": float(round(amount_std, 2)),
                "has_cycle": bool(has_cycle),
                "max_path_length": int(max_path_length),
                "num_intermediaries": int(num_intermediaries),
                "hop_count": int(hop_count)
            }
        }

    # ---------------------------------------------------------
    # Burst score = max txns in one day / avg txns per day
    # ---------------------------------------------------------
    def compute_burst_score(self, timestamps):
        """
        Stable burst detection:
        avoids inflation on small samples
        """

        if len(timestamps) <= 1:
            return 0.0

        day_counts = {}

        for ts in timestamps:
            d = ts.date()
            day_counts[d] = day_counts.get(d, 0) + 1

        values = list(day_counts.values())

        max_day = max(values)
        avg_day = sum(values) / len(values)

        if avg_day == 0:
            return 0.0

        burst = max_day / avg_day

        # normalize burst to 0–1 range (IMPORTANT)
        return min(burst / 5.0, 1.0)
    # ---------------------------------------------------------
    # Empty safe output
    # ---------------------------------------------------------
    def empty_result(self, account_id, G, hop_count):
        return {
            "account_id": account_id,
            "subgraph_node_count": G.number_of_nodes(),
            "subgraph_edge_count": G.number_of_edges(),
            "features": {
                "in_degree": 0,
                "out_degree": 0,
                "in_out_ratio": 0.0,
                "betweenness": 0.0,
                "total_received": 0.0,
                "total_sent": 0.0,
                "net_flow": 0.0,
                "txn_velocity": 0.0,
                "burst_score": 0.0,
                "avg_amount": 0.0,
                "amount_std": 0.0,
                "has_cycle": False,
                "max_path_length": 0,
                "num_intermediaries": 0,
                "hop_count": hop_count
            }
        }