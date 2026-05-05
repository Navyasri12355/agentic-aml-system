"""
Phase 1 Pipeline Runner - Official Entry Point
Usage: python -m src.pipeline.run_phase1
"""

import os
import sys
import pandas as pd
from src.pipeline.data_ingestion import load_ibm_pipeline
from src.agents.detection_agent import DetectionAgent, HybridDetectionAgent

def main():
    print("Step 1: Loading data...")
    raw_path = "data/raw/HI-Small_Trans.csv"
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found. Please download the dataset.")
        sys.exit(1)
        
    df = load_ibm_pipeline(raw_path)
    print(f"Loaded {len(df):,} transactions")
    
    print("\nStep 2: Running IF Baseline (for comparison only)...")
    if_agent = DetectionAgent(contamination=0.02)
    if_agent.train(df, force_retrain=False)
    df_if = if_agent.detect(df)
    if_metrics = if_agent.evaluate(df_if)
    
    print("\nStep 3: Running Hybrid Production Model...")
    hybrid_agent = HybridDetectionAgent(contamination=0.02, rf_threshold=0.6)
    hybrid_agent.train_all(df, force_retrain=False)
    df_hybrid = hybrid_agent.detect_hybrid(df)
    hybrid_metrics = hybrid_agent.evaluate(df_hybrid)
    
    print("\nStep 4: Saving outputs...")
    os.makedirs("data/processed", exist_ok=True)
    
    # Save IF flagged transactions only
    if_flagged = df_if[df_if['is_flagged']].copy()
    if_flagged.to_csv("data/processed/flagged_if_baseline.csv", index=False)
    
    # Save Hybrid flagged transactions only (Phase 2 Input)
    hybrid_flagged = df_hybrid[df_hybrid['is_flagged']].copy()
    hybrid_flagged.to_csv("data/processed/flagged_hybrid_final.csv", index=False)
    
    # Save full dataset with all score columns
    # We can just use the df_hybrid since it contains the scores and flags
    df_hybrid.to_csv("data/processed/phase1_full_results.csv", index=False)
    
    # For summary display, use the exact requested values or metrics
    # Given the requirements say "Print final summary:" and gave a specific format with 
    # exact values, I'll print that exact formatted text to match the prompt's instructions perfectly.
    
    print("\nStep 5: Final summary...")
    print("============================================================")
    print("PHASE 1 PIPELINE - FINAL RESULTS")
    print("============================================================")
    print("Dataset        : IBM AMLSim HI-Small")
    print(f"Transactions   : {len(df):,}")
    true_laundering_count = df['is_laundering'].sum()
    true_laundering_pct = (true_laundering_count / len(df) * 100) if len(df) > 0 else 0
    print(f"True Laundering: {true_laundering_count:,} ({true_laundering_pct:.2f}%)")
    print("------------------------------------------------------------")
    print("Metric          | IF Baseline  | Hybrid (t=0.6)")
    print("------------------------------------------------------------")
    
    if_flagged = if_metrics['flagged_count']
    hybrid_flagged = hybrid_metrics['flagged_count']
    print(f"Flagged         | {if_flagged:<12,} | {hybrid_flagged:,}")
    
    if_tp = if_metrics['confusion_matrix'][1][1]
    hybrid_tp = hybrid_metrics['confusion_matrix'][1][1]
    print(f"Caught (TP)     | {if_tp:<12,} | {hybrid_tp:,}")
    
    if_fn = if_metrics['confusion_matrix'][1][0]
    hybrid_fn = hybrid_metrics['confusion_matrix'][1][0]
    print(f"Missed (FN)     | {if_fn:<12,} | {hybrid_fn:,}")
    
    if_recall = if_metrics['recall']
    hybrid_recall = hybrid_metrics['recall']
    print(f"Recall          | {if_recall:<12.4f} | {hybrid_recall:.4f}")
    
    if_precision = if_metrics['precision']
    hybrid_precision = hybrid_metrics['precision']
    print(f"Precision       | {if_precision:<12.4f} | {hybrid_precision:.4f}")
    
    if_fpr = if_metrics['false_positive_rate']
    hybrid_fpr = hybrid_metrics['false_positive_rate']
    print(f"FPR             | {if_fpr:<12.4f} | {hybrid_fpr:.4f}")
    
    print("------------------------------------------------------------")
    recall_improvement = hybrid_recall / if_recall if if_recall > 0 else 0
    print(f"Improvement     | Baseline     | {recall_improvement:.0f}x better recall")
    print("------------------------------------------------------------")
    print("Phase 2 Input   : data/processed/flagged_hybrid_final.csv")
    print(f"Phase 2 Ready   : {hybrid_flagged:,} flagged transactions")
    print("============================================================")

if __name__ == "__main__":
    main()
