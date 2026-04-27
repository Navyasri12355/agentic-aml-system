import pandas as pd

def build_global_stats(clean_df):
    stats = {}

    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    # -------------------------
    # Velocity (FIXED: derived properly)
    # -------------------------
    tx_per_day = len(clean_df) / ((clean_df["timestamp"].max() - clean_df["timestamp"].min()).days + 1)

    stats["velocity_mean"] = tx_per_day
    stats["velocity_std"] = tx_per_day * 0.5
    stats["velocity_p95"] = tx_per_day * 2
    stats["velocity_p99"] = tx_per_day * 3

    # -------------------------
    # Anomaly
    # -------------------------
    stats["anomaly_p1"] = clean_df["anomaly_score"].quantile(0.01)
    stats["anomaly_p95"] = clean_df["anomaly_score"].quantile(0.95)
    stats["anomaly_p99"] = clean_df["anomaly_score"].quantile(0.99)

    # -------------------------
    # Amount
    # -------------------------
    stats["amount_mean"] = clean_df["amount"].mean()
    stats["amount_std"] = clean_df["amount"].std()
    stats["amount_median"] = clean_df["amount"].median()

    # -------------------------
    # Amount percentiles (🔥 ADD THIS)
    # -------------------------
    stats["p95_amount"] = clean_df["amount"].quantile(0.95)
    stats["p99_amount"] = clean_df["amount"].quantile(0.99)

    # -------------------------
    # Network activity
    # -------------------------
    sender_counts = clean_df.groupby("sender_id").size()
    receiver_counts = clean_df.groupby("receiver_id").size()

    stats["txn_per_sender_mean"] = sender_counts.mean()
    stats["txn_per_sender_std"] = sender_counts.std()

    stats["txn_per_receiver_mean"] = receiver_counts.mean()
    stats["txn_per_receiver_std"] = receiver_counts.std()

    return stats