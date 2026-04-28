# File: src/pipeline/run_phase2.py - OPTIMIZED VERSION

import os
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from src.agents.graph_agent import GraphAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.risk_agent import RiskAgent
from src.utils.global_stats import build_global_stats


def ensure_folder(path):
    """Create directory if it doesn't exist"""
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def choose_account(flagged_row):
    """Decide which account to investigate. Default = sender_id. Fallback = receiver_id."""
    sender = getattr(flagged_row, "sender_id", None)
    receiver = getattr(flagged_row, "receiver_id", None)

    if pd.notna(sender):
        return str(sender)
    if pd.notna(receiver):
        return str(receiver)
    return None


def safe_timestamp(row):
    """Get timestamp from row safely."""
    ts = getattr(row, "timestamp", None)
    if ts is not None:
        return ts
    return getattr(row, "date", None)


def get_transaction_id(row):
    """Safely extract transaction ID from row."""
    val = getattr(row, "transaction_id", None)
    return str(val) if val is not None and pd.notna(val) else None


def process_one_case(row, graph_agent, feature_agent, pattern_agent, risk_agent,
                     hop_radius=2, time_window_days=30, max_neighbors=50):
    """Runs complete Phase 2 for one flagged transaction."""
    
    account_id = choose_account(row)
    flag_date = safe_timestamp(row)
    transaction_id = get_transaction_id(row)

    if account_id is None or flag_date is None:
        return None

    # 2.1 Graph Construction
    graph_result = graph_agent.build_subgraph(
        account_id=account_id,
        flag_date=flag_date,
        hop_radius=hop_radius,
        time_window_days=time_window_days,
        max_neighbors=max_neighbors
    )

    # 2.2 Feature Extraction
    feature_result = feature_agent.extract_features(graph_result)

    # 2.3 Pattern Detection
    pattern_result = pattern_agent.detect_patterns(feature_result)

    # 2.4 Risk Scoring
    risk_result = risk_agent.compute_risk(
        flagged_row=row,
        feature_result=feature_result,
        pattern_result=pattern_result,
        graph_result=graph_result,
        transaction_id=transaction_id
    )
    
    return risk_result


def main():
    # Paths
    clean_path = "data/processed/phase1_full_results.csv"
    flagged_path = "data/processed/flagged_hybrid_final.csv"
    output_path = "data/processed/risk_scored_accounts.json"

    # Configurable slice (modify these as needed)
    start_idx = 575
    end_idx = 595
    
    # Load Data
    print("Loading datasets...")
    clean_df = pd.read_csv(clean_path)
    flagged_df = pd.read_csv(flagged_path).iloc[start_idx:end_idx]

    print(f"Clean transactions: {len(clean_df):,}")
    print(f"Flagged transactions: {len(flagged_df)}")

    # Build global stats
    global_stats = build_global_stats(clean_df)

    # Ensure timestamp parsing
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])
    flagged_df["timestamp"] = pd.to_datetime(flagged_df["timestamp"])
    
    # Create hour alias if needed
    if "hour_of_day" in flagged_df.columns:
        flagged_df["hour"] = flagged_df["hour_of_day"]

    # Initialize Agents
    print("Initializing agents...")
    graph_agent = GraphAgent(clean_df)
    feature_agent = FeatureAgent(global_stats)
    pattern_agent = PatternAgent(global_stats)
    risk_agent = RiskAgent(global_stats)

    # Process All Flagged Cases - SINGLE PROGRESS BAR
    print("Running Phase 2 pipeline...")
    results = {}
    flat_results = []
    
    # Convert to list for consistent iteration with single tqdm
    rows = list(flagged_df.itertuples(index=False))
    
    with tqdm(total=len(rows), desc=f"Processing transactions", unit="tx", 
              bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}') as pbar:
        
        for row in rows:
            try:
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
                    account_id = result["account_id"]
                    if account_id not in results:
                        results[account_id] = []
                    results[account_id].append(result)
                    flat_results.append(result)

            except Exception as e:
                print(f"\nError processing: {e}")
            finally:
                pbar.update(1)
                # Update progress bar description with current risk stats occasionally
                if pbar.n % 10 == 0 and flat_results:
                    high_count = sum(1 for r in flat_results if r.get("risk_tier") == "HIGH")
                    pbar.set_postfix({"HIGH": high_count})
    
    # Save Output
    ensure_folder(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(flat_results, f, indent=2)

    print(f"\n✅ Phase 2 complete.")
    print(f"   Cases processed: {len(flat_results)}")
    print(f"   Unique accounts: {len(results)}")
    print(f"   Saved to: {output_path}")

    # Quick Summary
    if flat_results:
        risk_df = pd.DataFrame(flat_results)
        print("\n📊 Risk Tier Distribution:")
        tier_counts = risk_df["risk_tier"].value_counts()
        for tier, count in tier_counts.items():
            print(f"   {tier}: {count} ({count/len(flat_results)*100:.1f}%)")

    # Evaluation
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY (Rows {start_idx}:{end_idx})")
    print(f"{'='*60}")
    
    eval_df = flagged_df.copy()
    possible_labels = ["is_laundering", "Is Laundering", "label", "target"]
    label_col = next((col for col in possible_labels if col in eval_df.columns), None)
    
    if label_col is not None and flat_results:
        risk_df = pd.DataFrame(flat_results)
        
        # Ensure transaction_id exists for merge
        if "transaction_id" not in risk_df.columns:
            print("⚠️ Warning: transaction_id missing from results")
            return
            
        merged = eval_df.merge(
            risk_df[["transaction_id", "routing_decision"]],
            on="transaction_id",
            how="left"
        )
        
        actual_positive = int(merged[label_col].sum())
        investigate_df = merged[merged["routing_decision"] == "INVESTIGATE"]
        investigate_count = len(investigate_df)
        true_positive = int(investigate_df[label_col].sum()) if len(investigate_df) > 0 else 0
        false_positive = investigate_count - true_positive
        missed_cases = actual_positive - true_positive
        
        print(f"\n--- EVALUATION ON {len(eval_df)} TRANSACTIONS ---")
        print(f"   Total transactions        : {len(eval_df)}")
        print(f"   Actual laundering cases   : {actual_positive}")
        print(f"   Sent for investigation    : {investigate_count}")
        print(f"   True laundering caught    : {true_positive}")
        print(f"   False positives           : {false_positive}")
        print(f"   Missed laundering cases   : {missed_cases}")
        
        if actual_positive > 0:
            print(f"\n   Recall  (TP/Actual)      : {true_positive/actual_positive*100:.1f}%")
        if investigate_count > 0:
            print(f"   Precision (TP/Investigate): {true_positive/investigate_count*100:.1f}%")
            
    else:
        print("\n⚠️ No laundering label column found. Evaluation skipped.")


if __name__ == "__main__":
    main()