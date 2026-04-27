# File: src/pipeline/run_phase2.py

import os
import json
import pandas as pd
from tqdm import tqdm
from time import time

from src.agents.graph_agent import GraphAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.risk_agent import RiskAgent
from src.utils import global_stats
from src.utils.global_stats import build_global_stats

def ensure_folder(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def choose_account(flagged_row):
    """
    Decide which account to investigate.
    Default = sender_id.
    Fallback = receiver_id.
    """
    sender = getattr(flagged_row, "sender_id", None)
    receiver = getattr(flagged_row, "receiver_id", None)

    if pd.notna(sender):
        return str(sender)

    if pd.notna(receiver):
        return str(receiver)

    return None


def safe_timestamp(row):
    """
    Get timestamp from row safely.
    """
    ts = getattr(row, "timestamp", None)
    if ts is not None:
        return ts

    dt = getattr(row, "date", None)
    return dt

def get_transaction_id(row):
    """
    Safely extract transaction ID from row.
    """
    for col in ["transaction_id"]:
        val = getattr(row, col, None)
        if val is not None and pd.notna(val):
            return str(val)
    return None

def process_one_case(
    row,
    graph_agent,
    feature_agent,
    pattern_agent,
    risk_agent,
    hop_radius=2,
    time_window_days=30,
    max_neighbors=50
):
    """
    Runs complete Phase 2 for one flagged transaction.
    """

    account_id = choose_account(row)
    flag_date = safe_timestamp(row)
    transaction_id = get_transaction_id(row)

    if account_id is None or flag_date is None:
        return None

    # -----------------------------
    # 2.1 Graph Construction
    # -----------------------------
    graph_result = graph_agent.build_subgraph(
        account_id=account_id,
        flag_date=flag_date,
        hop_radius=hop_radius,
        time_window_days=time_window_days,
        max_neighbors=max_neighbors
    )
    # logging
    # 🔥 DEBUG PRINT (only for first few)
    if hasattr(process_one_case, "counter"):
        process_one_case.counter += 1
    else:
        process_one_case.counter = 1

    # -----------------------------
    # 2.2 Feature Extraction
    # -----------------------------
    feature_result = feature_agent.extract_features(graph_result)

    # -----------------------------
    # 2.3 Pattern Detection
    # -----------------------------
    pattern_result = pattern_agent.detect_patterns(feature_result)

    # -----------------------------
    # 2.4 Risk Scoring
    # -----------------------------
    risk_result = risk_agent.compute_risk(
        flagged_row=row,
        feature_result=feature_result,
        pattern_result=pattern_result
    )
    if process_one_case.counter <= 10:
        print(f"[DEBUG] Account: {account_id} | Nodes: {graph_result['node_count']} | Edges: {graph_result['edge_count']}")
        print(f"  Patterns: {pattern_result['detected_patterns']}")
        print(f"  Velocity: {feature_result['features']['txn_velocity']}")
        print(f"  Risk: {risk_result['risk_score']}\n")

    # -----------------------------
    # Merge outputs
    # -----------------------------
    final_result = {
        "transaction_id": transaction_id,
        "account_id": account_id,
        "flag_date": str(flag_date),

        "graph_summary": {
            "node_count": graph_result.get("node_count", 0),
            "edge_count": graph_result.get("edge_count", 0),
            "is_isolated": graph_result.get("is_isolated", 0)
        },

        "features": feature_result["features"],

        "patterns": pattern_result["detected_patterns"],
        "pattern_confidence": pattern_result["pattern_confidence"],

        "risk_score": risk_result["risk_score"],
        "risk_tier": risk_result["risk_tier"],
        "score_components": risk_result["score_components"],
        "routing_decision": risk_result["routing_decision"]
    }

    return final_result


def main():
    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    clean_path = "data/processed/phase1_full_results.csv"
    flagged_path = "data/processed/flagged_hybrid_final.csv"
    output_path = "data/processed/risk_scored_accounts.json"

    # -----------------------------------------------------
    # Load Data
    # -----------------------------------------------------
    print("Loading datasets...")

    clean_df = pd.read_csv(clean_path)
    start_idx = 24300
    end_idx = 24320
    flagged_df = pd.read_csv(flagged_path).iloc[start_idx:end_idx]

    print("Clean transactions:", len(clean_df))
    print("Flagged transactions:", len(flagged_df))

    #building global stats for thresholds
    global_stats = build_global_stats(clean_df)

    # Ensure timestamp parsing
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

    if "timestamp" in flagged_df.columns:
        flagged_df["timestamp"] = pd.to_datetime(flagged_df["timestamp"])

    if "hour_of_day" in flagged_df.columns:
        flagged_df["hour"] = flagged_df["hour_of_day"]  # Create alias for consistency

    # -----------------------------------------------------
    # Initialize Agents
    # -----------------------------------------------------
    print("Initializing agents...") 
    graph_agent = GraphAgent(clean_df) 
    feature_agent = FeatureAgent(global_stats) 
    pattern_agent = PatternAgent(global_stats) 
    risk_agent = RiskAgent(global_stats)

    # -----------------------------------------------------
    # Process All Flagged Cases
    # -----------------------------------------------------
    results = {}

    print("Running Phase 2 pipeline...")

    for row in tqdm(flagged_df.itertuples(index=False),total=len(flagged_df), desc=f"Processing {start_idx}:{end_idx}", unit="tx"):
        try:
            account_id = choose_account(row)
            flag_date = safe_timestamp(row)

            result = process_one_case(
                row=row,
                graph_agent=graph_agent,
                feature_agent=feature_agent,
                pattern_agent=pattern_agent,
                risk_agent=risk_agent,
                hop_radius=2,
                time_window_days=30,
                max_neighbors=50
            )

            if result is not None:
                if account_id not in results:
                    results[account_id] = []

                results[account_id].append(result)

        except Exception as e:
            print(f"Error processing account: {e}")
            continue    
    # -----------------------------------------------------
    # Save Output
    # -----------------------------------------------------
    ensure_folder(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        flat_results = [item for sublist in results.values() for item in sublist]
        json.dump(flat_results, f, indent=2)

    print("\nPhase 2 complete.")
    print("Cases processed:", len(results))
    print("Saved to:", output_path)

    # -----------------------------------------------------
    # Quick Summary
    # -----------------------------------------------------
    if len(results) > 0:
        risk_df = pd.DataFrame(flat_results)

        print("\nRisk Tier Counts:")
        print(risk_df["risk_tier"].value_counts())

    # -----------------------------------------------------
    # MINI EVALUATION
    # -----------------------------------------------------
    print(f"\n{start_idx}:{end_idx} {len(flagged_df)} total flagged transactions")
    eval_df = flagged_df.copy()
    possible_labels = ["is_laundering", "Is Laundering", "label", "target"]
    label_col = next((col for col in possible_labels if col in eval_df.columns), None)
    if label_col is not None:
        # flatten results
        risk_df = pd.DataFrame(flat_results)
        # merge predictions back
        merged = eval_df.merge(
            risk_df[["transaction_id", "routing_decision"]],
            on="transaction_id",
            how="left"
        )
        actual_positive = int(merged[label_col].sum())
        investigate_df = merged[merged["routing_decision"] == "INVESTIGATE"]
        investigate_count = len(investigate_df)
        true_positive = int(investigate_df[label_col].sum())
        false_positive = investigate_count - true_positive
        missed_cases = actual_positive - true_positive
        print(f"\n--- EVALUATION ON {len(eval_df)} TRANSACTIONS ---")
        print(f"Actual laundering cases      : {actual_positive}")
        print(f"Sent for investigation       : {investigate_count}")
        print(f"True laundering investigated : {true_positive}")
        print(f"False positives investigated : {false_positive}")
        print(f"Missed laundering cases      : {missed_cases}")
    else:
        print("\nNo laundering label column found. Evaluation skipped.")
if __name__ == "__main__":
    main()