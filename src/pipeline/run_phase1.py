import os
import logging
import pandas as pd
from src.pipeline.data_ingestion import load_ibm_pipeline, generate_synthetic_data
from src.agents.detection_agent import DetectionAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase1_pipeline():
    """ Runs the complete Phase 1 pipeline: Load, Train, Detect, Evaluate, Save. """
    
    raw_path = "data/raw/HI-Small_Trans.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load Data
    if os.path.exists(raw_path):
        logger.info(f"Loading real dataset from {raw_path}")
        df = load_ibm_pipeline(raw_path)
    else:
        logger.warning(f"File {raw_path} not found. Running with 50,000 synthetic rows for demonstration.")
        # Using a larger synthetic set to simulate real load
        synthetic_raw = generate_synthetic_data(50000)
        tmp_path = "data/raw/synthetic_run.csv"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        synthetic_raw.to_csv(tmp_path, index=False)
        df = load_ibm_pipeline(tmp_path)
        os.remove(tmp_path)
    
    # 2. Instantiate Agent
    agent = DetectionAgent(contamination=0.02)
    
    # 3. Train Agent
    logger.info("Training Detection Agent...")
    agent.train(df, force_retrain=True)
    
    # 4. Run Detection
    logger.info("Running detection on full dataset...")
    df_results = agent.detect(df)
    
    # 5. Evaluate and Print Metrics
    logger.info("Evaluating performance...")
    metrics = agent.evaluate(df_results)
    
    # 6. Save Results
    flagged_path = os.path.join(processed_dir, "flagged_transactions.csv")
    clean_path = os.path.join(processed_dir, "clean_transactions.csv")
    
    df_results[df_results['is_flagged']].to_csv(flagged_path, index=False)
    df_results.to_csv(clean_path, index=False)
    
    logger.info(f"Results saved to {processed_dir}")
    
    # 7. Print Final Summary
    total_rows = len(df_results)
    flagged_count = int(df_results['is_flagged'].sum())
    
    tp = metrics['confusion_matrix'][1][1]
    fn = metrics['confusion_matrix'][1][0]
    
    print("\n" + "="*50)
    print("PHASE 1 PIPELINE SUMMARY")
    print("="*50)
    print(f"Total Transactions Processed: {total_rows:,}")
    print(f"Total Flagged (Alerts):       {flagged_count:,} ({flagged_count/total_rows:.2%})")
    print(f"Laundering Caught (TP):       {tp:,}")
    print(f"Laundering Missed (FN):       {fn:,}")
    print(f"Detection Precision:          {metrics['precision']:.4f}")
    print(f"Detection Recall:             {metrics['recall']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_phase1_pipeline()
