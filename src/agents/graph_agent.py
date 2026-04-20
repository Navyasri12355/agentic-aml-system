"""
graph_agent.py
--------------
Phase 2: Build a directed transaction graph from flagged transactions
and expand context to multi-hop neighbouring accounts.

Input:
    flagged_df  : DataFrame of flagged transactions (from detection_agent.py)
    all_df      : Full cleaned DataFrame (for context expansion)
    account_id  : The account under investigation
    hop_radius  : How many hops to expand (default: 2)
    time_window_days : How many days back to include (default: 30)

Output:
    networkx.DiGraph with:
        Nodes : account IDs
            attrs: total_sent, total_received, degree
        Edges : transactions
            attrs: amount, timestamp, transaction_type, transaction_id
"""

import logging
from datetime import timedelta
from typing import Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_transaction_graph(
    flagged_df: pd.DataFrame,
    all_df: pd.DataFrame,
    account_id: str,
    hop_radius: int = 2,
    time_window_days: int = 30,
) -> nx.DiGraph:
    """
    Build a directed subgraph centred on account_id.

    Args:
        flagged_df:        Flagged transactions (to identify the anchor event).
        all_df:            Full cleaned transaction DataFrame.
        account_id:        Account under investigation.
        hop_radius:        Number of hops to expand from account_id.
        time_window_days:  Temporal window (days before the last flagged txn).

    Returns:
        nx.DiGraph of the transaction subgraph.
    """
    # Determine time window from latest flagged transaction for this account
    account_flags = flagged_df[
        (flagged_df["sender_id"] == account_id)
        | (flagged_df["receiver_id"] == account_id)
    ]

    if account_flags.empty:
        logger.warning(
            f"Account {account_id} not found in flagged transactions. "
            "Building graph from all_df directly."
        )
        reference_time = all_df["timestamp"].max()
    else:
        reference_time = account_flags["timestamp"].max()

    window_start = reference_time - timedelta(days=time_window_days)

    # Filter all transactions to the time window
    windowed_df = all_df[all_df["timestamp"] >= window_start].copy()

    logger.info(
        f"Building graph for account {account_id} | "
        f"window: {window_start.date()} → {reference_time.date()} | "
        f"hops: {hop_radius}"
    )

    # BFS-style expansion
    included_accounts = {account_id}
    frontier = {account_id}

    for hop in range(hop_radius):
        new_frontier = set()
        for acct in frontier:
            connected = _get_connected_accounts(acct, windowed_df)
            new_accounts = connected - included_accounts
            new_frontier.update(new_accounts)
            included_accounts.update(new_accounts)
        frontier = new_frontier
        if not frontier:
            logger.info(f"Expansion stopped at hop {hop + 1} — no new accounts.")
            break

    # Build the subgraph from all edges between included accounts
    subgraph_edges = windowed_df[
        windowed_df["sender_id"].isin(included_accounts)
        & windowed_df["receiver_id"].isin(included_accounts)
    ]

    G = _build_digraph(subgraph_edges)
    _annotate_nodes(G, windowed_df, included_accounts)

    logger.info(
        f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G


def _get_connected_accounts(account_id: str, df: pd.DataFrame) -> set:
    """Return all accounts directly connected to account_id (in or out)."""
    sent_to = set(df[df["sender_id"] == account_id]["receiver_id"].unique())
    received_from = set(df[df["receiver_id"] == account_id]["sender_id"].unique())
    return sent_to | received_from


def _build_digraph(edges_df: pd.DataFrame) -> nx.DiGraph:
    """Construct a DiGraph from an edge DataFrame."""
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            amount=row["amount"],
            timestamp=str(row["timestamp"]),
            transaction_type=row["transaction_type"],
            transaction_id=row["transaction_id"],
        )
    return G


def _annotate_nodes(
    G: nx.DiGraph, df: pd.DataFrame, account_ids: set
) -> None:
    """Add aggregate financial stats as node attributes."""
    for acct in account_ids:
        total_sent = df[df["sender_id"] == acct]["amount"].sum()
        total_received = df[df["receiver_id"] == acct]["amount"].sum()
        G.nodes[acct]["total_sent"] = float(total_sent)
        G.nodes[acct]["total_received"] = float(total_received)
        G.nodes[acct]["net_flow"] = float(total_received - total_sent)


def graph_to_dict(G: nx.DiGraph) -> dict:
    """
    Serialize a NetworkX DiGraph to a JSON-compatible dict
    for storage in AMLAgentState and API responses.

    Returns:
        {
          "nodes": [{"id": str, "total_sent": float, ...}, ...],
          "edges": [{"source": str, "target": str, "amount": float, ...}, ...]
        }
    """
    nodes = [
        {"id": node, **attrs}
        for node, attrs in G.nodes(data=True)
    ]
    edges = [
        {"source": u, "target": v, **attrs}
        for u, v, attrs in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}