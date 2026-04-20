"""
feature_agent.py
----------------
Phase 2: Extract topological and temporal features from a transaction subgraph.

Input:
    G          : nx.DiGraph (from graph_agent.py)
    account_id : The account under investigation
    df         : Cleaned transaction DataFrame (for temporal features)

Output:
    dict of features — see extract_features() return docstring for full schema
"""

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_features(
    G: nx.DiGraph,
    account_id: str,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Extract all features for a flagged account from its subgraph.

    Args:
        G:          Transaction subgraph centred on account_id.
        account_id: The account being investigated.
        df:         Full cleaned DataFrame for temporal feature computation.
                    If None, temporal features will be set to -1 (unknown).

    Returns:
        {
          "account_id"         : str,
          "subgraph_node_count": int,
          "subgraph_edge_count": int,

          # Degree features
          "in_degree"          : int,
          "out_degree"         : int,
          "in_out_ratio"       : float,

          # Centrality
          "betweenness"        : float,

          # Financial flow
          "total_received"     : float,
          "total_sent"         : float,
          "net_flow"           : float,

          # Temporal (requires df)
          "txn_velocity"       : float,  # txns/day in time window
          "burst_score"        : float,  # max 24h txns / avg daily txns
          "avg_amount"         : float,
          "amount_std"         : float,

          # Structural
          "has_cycle"          : bool,
          "max_path_length"    : int,
          "num_intermediaries" : int,
          "hop_count"          : int,

          # Cross-border ratio
          "cross_border_ratio" : float,
        }
    """
    if account_id not in G.nodes:
        logger.warning(
            f"Account {account_id} not found in subgraph. "
            "Returning zero-value features."
        )
        return _zero_features(account_id)

    features: dict = {
        "account_id": account_id,
        "subgraph_node_count": G.number_of_nodes(),
        "subgraph_edge_count": G.number_of_edges(),
    }

    features.update(_degree_features(G, account_id))
    features.update(_centrality_features(G, account_id))
    features.update(_flow_features(G, account_id))
    features.update(_structural_features(G, account_id))

    if df is not None:
        features.update(_temporal_features(df, account_id))
        features.update(_cross_border_features(df, account_id))
    else:
        features.update({
            "txn_velocity": -1.0,
            "burst_score": -1.0,
            "avg_amount": -1.0,
            "amount_std": -1.0,
            "cross_border_ratio": -1.0,
        })

    return features


# ---------------------------------------------------------------------------
# Feature sub-extractors
# ---------------------------------------------------------------------------

def _degree_features(G: nx.DiGraph, account_id: str) -> dict:
    in_deg = G.in_degree(account_id)
    out_deg = G.out_degree(account_id)
    ratio = in_deg / (out_deg + 1e-6)
    return {
        "in_degree": int(in_deg),
        "out_degree": int(out_deg),
        "in_out_ratio": round(float(ratio), 4),
    }


def _centrality_features(G: nx.DiGraph, account_id: str) -> dict:
    try:
        bc = nx.betweenness_centrality(G, normalized=True)
        betweenness = bc.get(account_id, 0.0)
    except Exception:
        betweenness = 0.0
    return {"betweenness": round(float(betweenness), 6)}


def _flow_features(G: nx.DiGraph, account_id: str) -> dict:
    node_data = G.nodes[account_id]
    return {
        "total_received": round(node_data.get("total_received", 0.0), 2),
        "total_sent": round(node_data.get("total_sent", 0.0), 2),
        "net_flow": round(node_data.get("net_flow", 0.0), 2),
    }


def _structural_features(G: nx.DiGraph, account_id: str) -> dict:
    # Cycle detection
    try:
        cycles = list(nx.simple_cycles(G))
        has_cycle = any(account_id in c for c in cycles)
    except Exception:
        has_cycle = False

    # Longest shortest path from account_id
    try:
        lengths = nx.single_source_shortest_path_length(G, account_id)
        max_path = max(lengths.values()) if lengths else 0
    except Exception:
        max_path = 0

    # Intermediary accounts: in_degree > 0 AND out_degree > 0
    intermediaries = [
        n for n in G.nodes
        if G.in_degree(n) > 0 and G.out_degree(n) > 0 and n != account_id
    ]

    return {
        "has_cycle": bool(has_cycle),
        "max_path_length": int(max_path),
        "num_intermediaries": int(len(intermediaries)),
        "hop_count": int(max_path),  # aliased — same concept for now
    }


def _temporal_features(df: pd.DataFrame, account_id: str) -> dict:
    account_txns = df[
        (df["sender_id"] == account_id) | (df["receiver_id"] == account_id)
    ].copy()

    if account_txns.empty:
        return {
            "txn_velocity": 0.0,
            "burst_score": 0.0,
            "avg_amount": 0.0,
            "amount_std": 0.0,
        }

    # Velocity: transactions per day
    time_range_days = (
        account_txns["timestamp"].max() - account_txns["timestamp"].min()
    ).days or 1
    velocity = len(account_txns) / time_range_days

    # Burst score: max daily transactions / average daily transactions
    account_txns["date"] = account_txns["timestamp"].dt.date
    daily_counts = account_txns.groupby("date").size()
    burst = float(daily_counts.max()) / float(daily_counts.mean() + 1e-6)

    return {
        "txn_velocity": round(float(velocity), 4),
        "burst_score": round(float(burst), 4),
        "avg_amount": round(float(account_txns["amount"].mean()), 2),
        "amount_std": round(float(account_txns["amount"].std()), 2),
    }


def _cross_border_features(df: pd.DataFrame, account_id: str) -> dict:
    account_txns = df[
        (df["sender_id"] == account_id) | (df["receiver_id"] == account_id)
    ]
    if account_txns.empty:
        return {"cross_border_ratio": 0.0}
    ratio = account_txns["is_cross_border"].mean()
    return {"cross_border_ratio": round(float(ratio), 4)}


def _zero_features(account_id: str) -> dict:
    return {
        "account_id": account_id,
        "subgraph_node_count": 0,
        "subgraph_edge_count": 0,
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
        "hop_count": 0,
        "cross_border_ratio": 0.0,
    }