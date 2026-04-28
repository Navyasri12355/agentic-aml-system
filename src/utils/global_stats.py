import pandas as pd

def build_global_stats(clean_df):
    stats = {}

    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    # -------------------------
    # Velocity per account
    # -------------------------
    sender_counts = clean_df.groupby("sender_id").size()
    receiver_counts = clean_df.groupby("receiver_id").size()

    account_activity = sender_counts.add(receiver_counts, fill_value=0)

    stats["velocity_mean"] = account_activity.mean()
    stats["velocity_std"] = account_activity.std()
    stats["velocity_p95"] = account_activity.quantile(0.95)
    stats["velocity_p99"] = account_activity.quantile(0.99)

    # -------------------------
    # Anomaly
    # -------------------------
    stats["anomaly_p1"] = clean_df["anomaly_score"].quantile(0.01)
    stats["anomaly_p95"] = clean_df["anomaly_score"].quantile(0.95)
    stats["anomaly_p99"] = clean_df["anomaly_score"].quantile(0.99)
    stats["anomaly_p10"] = clean_df["anomaly_score"].quantile(0.10)
    stats["anomaly_p50"] = clean_df["anomaly_score"].quantile(0.50)
    stats["anomaly_p90"] = clean_df["anomaly_score"].quantile(0.90)

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