# File: src/agents/graph_agent.py

import pandas as pd
import networkx as nx
from datetime import timedelta


class GraphAgent:
    """
    Builds investigation subgraphs around flagged accounts.

    Nodes  = accounts
    Edges  = transactions (sender -> receiver)
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()

        # Ensure timestamp column is datetime
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # Build indexes for fast lookup
        self.sender_map = {
            acc: grp for acc, grp in self.df.groupby("sender_id")
        }

        self.receiver_map = {
            acc: grp for acc, grp in self.df.groupby("receiver_id")
        }

    # ---------------------------------------------------------
    # Get all transactions linked to one account
    # ---------------------------------------------------------
    def get_connected_rows(self, account_id, cutoff_date):
        sent = self.sender_map.get(account_id, pd.DataFrame())
        received = self.receiver_map.get(account_id, pd.DataFrame())

        combined = pd.concat([sent, received], ignore_index=True)

        if combined.empty:
            return combined

        combined = combined[
            combined["timestamp"] >= cutoff_date
        ]

        return combined

    # ---------------------------------------------------------
    # Expand accounts by hop radius
    # ---------------------------------------------------------
    def expand_accounts(self, seed_account, cutoff_date, hop_radius=2):
        visited = set()
        frontier = {seed_account}
        discovered = {seed_account}

        for _ in range(hop_radius):
            next_frontier = set()

            for account in frontier:
                if account in visited:
                    continue

                rows = self.get_connected_rows(account, cutoff_date)

                for _, row in rows.iterrows():
                    sender = row["sender_id"]
                    receiver = row["receiver_id"]

                    if sender not in discovered:
                        next_frontier.add(sender)
                        discovered.add(sender)

                    if receiver not in discovered:
                        next_frontier.add(receiver)
                        discovered.add(receiver)

                visited.add(account)

            frontier = next_frontier

            if not frontier:
                break

        return discovered

    # ---------------------------------------------------------
    # Build graph for one flagged account
    # ---------------------------------------------------------
    def build_subgraph(
        self,
        account_id,
        flag_date,
        hop_radius=2,
        time_window_days=30
    ):
        cutoff_date = pd.to_datetime(flag_date) - timedelta(days=time_window_days)

        accounts = self.expand_accounts(
            seed_account=account_id,
            cutoff_date=cutoff_date,
            hop_radius=hop_radius
        )

        sub_df = self.df[
            (
                self.df["sender_id"].isin(accounts)
                | self.df["receiver_id"].isin(accounts)
            )
            & (self.df["timestamp"] >= cutoff_date)
        ]

        G = nx.DiGraph()

        for _, row in sub_df.iterrows():
            sender = row["sender_id"]
            receiver = row["receiver_id"]

            G.add_edge(
                sender,
                receiver,
                amount=row["amount"],
                timestamp=row["timestamp"],
                txn_type=row.get("payment_type", "UNKNOWN"),
                transaction_id=row.get("transaction_id", ""),
                is_cross_border=row.get("is_cross_border", 0)
            )

        isolated = False
        if G.degree(account_id) < 2:
            isolated = True

        return {
            "account_id": account_id,
            "graph": G,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "is_isolated": isolated
        }


# ---------------------------------------------------------
# Utility loader
# ---------------------------------------------------------
def load_graph_agent(clean_csv_path):
    df = pd.read_csv(clean_csv_path)
    return GraphAgent(df)