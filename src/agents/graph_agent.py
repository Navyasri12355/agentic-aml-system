# File: src/agents/graph_agent.py

import pandas as pd
import networkx as nx
from datetime import timedelta


class GraphAgent:
    """
    Phase 2.1 Graph Construction (Corrected / Optimized)

    Features:
    - Internal-edge filtering
    - hop_radius=1 safe default
    - hub control (limit neighbors per node)
    - faster lookup using indexes
    - time window filtering
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()

        # Ensure datetime
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Sort newest first (helps hub control use recent rows)
        self.df = self.df.sort_values("timestamp", ascending=False)

        # Build lookup maps using indices (much faster than creating DataFrame for each group)
        self.sender_groups = self.df.groupby("sender_id").groups
        self.receiver_groups = self.df.groupby("receiver_id").groups

    # ---------------------------------------------------------
    # Get linked rows for one account (recent + capped)
    # ---------------------------------------------------------
    def get_connected_rows(
        self,
        account_id,
        cutoff_date,
        max_neighbors=150
    ):
        sender_idx = self.sender_groups.get(account_id, [])
        receiver_idx = self.receiver_groups.get(account_id, [])

        if len(sender_idx) == 0 and len(receiver_idx) == 0:
            return pd.DataFrame()

        # Combine indices and slice the dataframe once
        combined_idx = list(sender_idx) + list(receiver_idx)
        rows = self.df.loc[combined_idx]

        # Time filter
        rows = rows[rows["timestamp"] >= cutoff_date]

        if rows.empty:
            return rows

        # Hub control: keep only most recent rows
        rows = rows.head(max_neighbors)

        return rows

    # ---------------------------------------------------------
    # BFS hop expansion
    # ---------------------------------------------------------
    def expand_accounts(
        self,
        seed_account,
        cutoff_date,
        hop_radius=1,
        max_neighbors=150
    ):
        visited = set()
        frontier = {seed_account}
        discovered = {seed_account}

        for _ in range(hop_radius):
            next_frontier = set()

            for account in frontier:
                if account in visited:
                    continue

                rows = self.get_connected_rows(
                    account,
                    cutoff_date,
                    max_neighbors=max_neighbors
                )

                for row in rows.itertuples(index=False):
                    sender = row.sender_id
                    receiver = row.receiver_id

                    if sender not in discovered:
                        discovered.add(sender)
                        next_frontier.add(sender)

                    if receiver not in discovered:
                        discovered.add(receiver)
                        next_frontier.add(receiver)

                visited.add(account)

            frontier = next_frontier

            if not frontier:
                break

        return discovered

    # ---------------------------------------------------------
    # Build subgraph
    # ---------------------------------------------------------
    def build_subgraph(
        self,
        account_id,
        flag_date,
        hop_radius=1,           # SAFE MODE default
        time_window_days=30,
        max_neighbors=150
    ):
        flag_date = pd.to_datetime(flag_date)
        cutoff_date = flag_date - timedelta(days=time_window_days)

        # Step 1: discover relevant accounts
        accounts = self.expand_accounts(
            seed_account=account_id,
            cutoff_date=cutoff_date,
            hop_radius=hop_radius,
            max_neighbors=max_neighbors
        )

        # Step 2: INTERNAL EDGE FILTERING
        # Keep only rows where BOTH sender and receiver are in accounts
        sub_df = self.df[
            (self.df["timestamp"] >= cutoff_date) &
            (self.df["sender_id"].isin(accounts)) &
            (self.df["receiver_id"].isin(accounts))
        ]

        # Step 3: Build graph
        G = nx.DiGraph()

        for row in sub_df.itertuples(index=False):
            G.add_edge(
                row.sender_id,
                row.receiver_id,
                amount=row.amount,
                timestamp=row.timestamp,
                txn_type=getattr(row, "payment_type", "UNKNOWN"),
                transaction_id=getattr(row, "transaction_id", ""),
                is_cross_border=getattr(row, "is_cross_border", 0)
            )

        # Step 4: isolated check
        isolated = True
        if account_id in G.nodes:
            if G.degree(account_id) >= 2:
                isolated = False

        return {
            "account_id": account_id,
            "graph": G,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "is_isolated": isolated,
            "hop_radius_used": hop_radius,
            "accounts_discovered": len(accounts)
        }


# ---------------------------------------------------------
# Helper Loader
# ---------------------------------------------------------
def load_graph_agent(csv_path):
    df = pd.read_csv(csv_path)
    return GraphAgent(df)

def build_transaction_graph(flagged_df: pd.DataFrame, all_df: pd.DataFrame, account_id: str, hop_radius: int = 2, time_window_days: int = 30) -> nx.DiGraph:
    agent = GraphAgent(all_df)
    
    # Extract the flag_date from flagged_df for this account
    account_flags = flagged_df[(flagged_df['sender_id'] == account_id) | (flagged_df['receiver_id'] == account_id)]
    if not account_flags.empty:
        flag_date = account_flags['timestamp'].max()
    else:
        # Default to the most recent transaction in all_df for this account
        account_txns = all_df[(all_df['sender_id'] == account_id) | (all_df['receiver_id'] == account_id)]
        if not account_txns.empty:
            flag_date = pd.to_datetime(account_txns['timestamp'].max())
        else:
            flag_date = pd.Timestamp.now()
            
    result = agent.build_subgraph(
        account_id=account_id,
        flag_date=flag_date,
        hop_radius=hop_radius,
        time_window_days=time_window_days
    )
    return result["graph"]

def graph_to_dict(G: nx.DiGraph) -> dict:
    return nx.node_link_data(G)