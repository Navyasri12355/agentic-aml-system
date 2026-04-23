import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.agents.detection_agent import DetectionAgent
from src.pipeline.data_ingestion import generate_synthetic_data, load_and_clean, normalize_ibm_amlsim

@pytest.fixture
def processed_data():
    """Returns a processed DataFrame for testing."""
    df_raw = generate_synthetic_data(200)
    with open("temp_test_agent.csv", "w") as f:
        df_raw.to_csv(f, index=False)
    df_norm = normalize_ibm_amlsim("temp_test_agent.csv")
    df_clean = load_and_clean(df_norm)
    os.remove("temp_test_agent.csv")
    return df_clean

def test_agent_trains_without_error(processed_data):
    """train() on 200 synthetic rows completes."""
    model_path = "models/test_model.joblib"
    agent = DetectionAgent(model_path=model_path)
    agent.train(processed_data, force_retrain=True)
    assert os.path.exists(model_path)
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

def test_detect_adds_columns(processed_data):
    """detect() output has anomaly_score, is_flagged, flag_reason."""
    agent = DetectionAgent()
    # Train on small data to enable detect
    agent.train(processed_data, force_retrain=True)
    results = agent.detect(processed_data)
    
    assert 'anomaly_score' in results.columns
    assert 'is_flagged' in results.columns
    assert 'flag_reason' in results.columns

def test_flagged_is_bool(processed_data):
    """is_flagged column dtype is bool."""
    agent = DetectionAgent()
    agent.train(processed_data, force_retrain=True)
    results = agent.detect(processed_data)
    assert results['is_flagged'].dtype == bool

def test_flag_reason_empty_for_clean(processed_data):
    """is_flagged==False rows have flag_reason == ""."""
    agent = DetectionAgent()
    agent.train(processed_data, force_retrain=True)
    results = agent.detect(processed_data)
    clean_rows = results[~results['is_flagged']]
    assert (clean_rows['flag_reason'] == "").all()

def test_evaluate_returns_all_keys(processed_data):
    """evaluate() dict has all required keys."""
    agent = DetectionAgent()
    agent.train(processed_data, force_retrain=True)
    results = agent.detect(processed_data)
    metrics = agent.evaluate(results)
    
    required_keys = {
        'precision', 'recall', 'f1', 'false_positive_rate', 
        'confusion_matrix', 'flagged_count', 'total_count', 'flag_rate'
    }
    assert required_keys.issubset(metrics.keys())

def test_model_save_and_load(processed_data):
    """save then load model, detect still works."""
    model_path = "models/save_load_test.joblib"
    agent = DetectionAgent(model_path=model_path)
    agent.train(processed_data, force_retrain=True)
    
    # New agent instance
    agent2 = DetectionAgent(model_path=model_path)
    agent2.load_model()
    results = agent2.detect(processed_data)
    assert 'anomaly_score' in results.columns
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

def test_contamination_affects_flag_rate(processed_data):
    """higher contamination -> higher flag rate."""
    agent_low = DetectionAgent(contamination=0.01)
    agent_high = DetectionAgent(contamination=0.1)
    
    agent_low.train(processed_data, force_retrain=True)
    results_low = agent_low.detect(processed_data)
    
    agent_high.train(processed_data, force_retrain=True)
    results_high = agent_high.detect(processed_data)
    
    assert results_high['is_flagged'].sum() > results_low['is_flagged'].sum()
