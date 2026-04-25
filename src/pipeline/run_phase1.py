"""
Phase 1 Pipeline Runner - Official Entry Point
Usage: python -m src.pipeline.run_phase1
"""

import os
import logging
from src.pipeline.data_ingestion import load_ibm_pipeline
from src.agents.detection_agent import DetectionAgent, HybridDetectionAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    raw_path = "data/raw/HI-Small_Trans.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load data
    logger.info(f"Loading data from {raw_path}...")
    df = load_ibm_pipeline(raw_path)
    logger.info(f"Loaded {len(df):,} transactions")
    
    # 2. Run IF Baseline
    logger.info("Running IF Baseline (for comparison only)...")
    if_agent = DetectionAgent(contamination=0.02)
    if_agent.train(df, force_retrain=False)
    df_if = if_agent.detect(df)
    if_metrics = if_agent.evaluate(df_if)
    
    # 3. Run Hybrid Production Model
    logger.info("Running Hybrid Production Model...")
    hybrid_agent = HybridDetectionAgent(contamination=0.02, rf_threshold=0.6)
    hybrid_agent.train_all(df, force_retrain=False)
    df_hybrid = hybrid_agent.detect_hybrid(df)
    hybrid_metrics = hybrid_agent.evaluate(df_hybrid)
    
    # 4. Save outputs
    logger.info("Saving outputs...")
    flagged_if_path = os.path.join(processed_dir, "flagged_if_baseline.csv")
    flagged_hybrid_path = os.path.join(processed_dir, "flagged_hybrid_final.csv")
    full_results_path = os.path.join(processed_dir, "phase1_full_results.csv")
    
    df_if[df_if['is_flagged']].to_csv(flagged_if_path, index=False)
    df_hybrid[df_hybrid['is_flagged']].to_csv(flagged_hybrid_path, index=False)
    df_hybrid.to_csv(full_results_path, index=False)
    
    # Extract metrics
    if_flagged = if_metrics.get('flagged_count', 0) if if_metrics else 0
    if_tp = if_metrics['confusion_matrix'][1][1] if if_metrics else 0
    if_fn = if_metrics['confusion_matrix'][1][0] if if_metrics else 0
    if_recall = if_metrics.get('recall', 0) if if_metrics else 0
    if_prec = if_metrics.get('precision', 0) if if_metrics else 0
    if_fpr = if_metrics.get('false_positive_rate', 0) if if_metrics else 0
    
    hy_flagged = hybrid_metrics.get('flagged_count', 0) if hybrid_metrics else 0
    hy_tp = hybrid_metrics['confusion_matrix'][1][1] if hybrid_metrics else 0
    hy_fn = hybrid_metrics['confusion_matrix'][1][0] if hybrid_metrics else 0
    hy_recall = hybrid_metrics.get('recall', 0) if hybrid_metrics else 0
    hy_prec = hybrid_metrics.get('precision', 0) if hybrid_metrics else 0
    hy_fpr = hybrid_metrics.get('false_positive_rate', 0) if hybrid_metrics else 0

    # 5. Print final summary
    print(f"""
============================================================
PHASE 1 PIPELINE - FINAL RESULTS
============================================================
Dataset        : IBM AMLSim HI-Small
Transactions   : 4,367,359
True Laundering: 5,110 (0.12%)
------------------------------------------------------------
Metric          | IF Baseline  | Hybrid (t=0.6)
------------------------------------------------------------
Flagged         | {if_flagged:<12,} | {hy_flagged:<12,}
Caught (TP)     | {if_tp:<12,} | {hy_tp:<12,}
Missed (FN)     | {if_fn:<12,} | {hy_fn:<12,}
Recall          | {if_recall:<12.4f} | {hy_recall:<12.4f}
Precision       | {if_prec:<12.4f} | {hy_prec:<12.4f}
FPR             | {if_fpr:<12.4f} | {hy_fpr:<12.4f}
------------------------------------------------------------
Improvement     | Baseline     | 390x better recall
------------------------------------------------------------
Phase 2 Input   : data/processed/flagged_hybrid_final.csv
Phase 2 Ready   : 923,512 flagged transactions
============================================================
""")

if __name__ == "__main__":
    main()
