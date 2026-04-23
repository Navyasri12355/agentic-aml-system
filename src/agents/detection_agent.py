import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionAgent:
    """
    AML Detection Agent using Isolation Forest for anomaly detection.
    """
    def __init__(self, contamination: float = 0.02, 
                 model_path: str = "models/isolation_forest.joblib",
                 flag_threshold: float = 0.0):
        """
        Initialize the Detection Agent.
        
        Args:
            contamination: Expected proportion of outliers in the data.
            model_path: Path to save/load the model.
            flag_threshold: Anomaly score threshold for flagging (default 0.0).
        """
        self.contamination = contamination
        self.model_path = model_path
        self.flag_threshold = flag_threshold
        self.pipeline: Optional[Pipeline] = None
        
    def _get_preprocessor(self) -> ColumnTransformer:
        """
        Defines the preprocessing steps for the numerical and categorical features.
        """
        numerical_features = ['amount_log', 'hour_of_day', 'day_of_week', 'is_cross_border']
        categorical_features = ['transaction_type']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ]
        )
        return preprocessor

    def train(self, df: pd.DataFrame, force_retrain: bool = False) -> None:
        """
        Train the Isolation Forest model on the provided DataFrame.
        
        Args:
            df: Training data.
            force_retrain: If True, retrain even if a model exists.
        """
        if os.path.exists(self.model_path) and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            self.load_model()
            return
            
        logger.info(f"Training Isolation Forest model. Shape: {df.shape}, Contamination: {self.contamination}")
        
        # Ensure is_cross_border is int for standard scaler
        df_train = df.copy()
        df_train['is_cross_border'] = df_train['is_cross_border'].astype(int)
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self._get_preprocessor()),
            ('model', IsolationForest(
                contamination=self.contamination, 
                random_state=42, 
                n_estimators=100
            ))
        ])
        
        self.pipeline.fit(df_train)
        self.save_model()
        logger.info(f"Model trained and saved to {self.model_path}")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run detection on the provided DataFrame.
        
        Args:
            df: Data to analyze.
            
        Returns:
            pd.DataFrame: Original DF with anomaly columns appended.
        """
        if self.pipeline is None:
            self.load_model()
            
        df_detect = df.copy()
        df_detect['is_cross_border'] = df_detect['is_cross_border'].astype(int)
        
        # Get anomaly scores (decision_function output)
        df_detect['anomaly_score'] = self.pipeline.decision_function(df_detect)
        df_detect['is_flagged'] = df_detect['anomaly_score'] < self.flag_threshold
        
        # Define flag reason logic
        amount_95th = df_detect['amount_log'].quantile(0.95)
        amount_median = df_detect['amount'].median()
        
        def get_reason(row):
            if not row['is_flagged']:
                return ""
            
            if row['amount_log'] > amount_95th:
                return "High amount outlier"
            if row['hour_of_day'] in {0, 1, 2, 3, 4, 23}:
                return "Unusual transaction hour"
            if row['is_cross_border'] == 1 and row['amount'] > amount_median * 3:
                return "Cross-border high value"
            
            return "Isolation Forest anomaly"
            
        df_detect['flag_reason'] = df_detect.apply(get_reason, axis=1)
        
        # Cast is_flagged back to bool if needed, but decision_function result is already float
        df_detect['is_flagged'] = df_detect['is_flagged'].astype(bool)
        
        return df_detect

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model performance against ground truth labels.
        """
        if 'is_laundering' not in df.columns:
            logger.error("Column 'is_laundering' missing for evaluation")
            return {}
            
        y_true = df['is_laundering']
        y_pred = df['is_flagged'].astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'false_positive_rate': float(fpr),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'flagged_count': int(y_pred.sum()),
            'total_count': int(len(df)),
            'flag_rate': float(y_pred.mean())
        }
        
        print("\n--- Detection Evaluation ---")
        print(f"{'Metric':<20} | {'Value':<10} | {'Target':<10}")
        print("-" * 45)
        print(f"{'Precision':<20} | {precision:<10.4f} | >0.60")
        print(f"{'Recall':<20} | {recall:<10.4f} | >0.70")
        print(f"{'F1 Score':<20} | {f1:<10.4f} | >0.65")
        print(f"{'FPR':<20} | {fpr:<10.4f} | <0.30")
        
        # Suggestions
        if precision < 0.60:
            print("Suggestion: Precision below target. Try raising flag_threshold or increasing contamination.")
        if recall < 0.70:
            print(f"Suggestion: Recall below target: lower contamination from {self.contamination} to a higher value (e.g. 0.05).")
        if fpr > 0.30:
            print("Suggestion: FPR above target. Adjust flag_threshold to be more stringent.")
            
        return metrics

    def save_model(self) -> None:
        """Save the pipeline to model_path."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self) -> None:
        """Load the pipeline from model_path."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

if __name__ == "__main__":
    from src.pipeline.data_ingestion import generate_synthetic_data, load_and_clean, normalize_ibm_amlsim
    
    # Generate synthetic data for demo
    logger.info("Starting Detection Agent Demo...")
    synthetic_raw = generate_synthetic_data(1000)
    # We need to simulate the pipeline
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        synthetic_raw.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
        
    df_norm = normalize_ibm_amlsim(tmp_path)
    df_processed = load_and_clean(df_norm)
    os.remove(tmp_path)
    
    # Instantiate Agent
    agent = DetectionAgent(contamination=0.05) # Using 0.05 because synthetic labels are ~5%
    
    # Train
    agent.train(df_processed, force_retrain=True)
    
    # Detect
    df_results = agent.detect(df_processed)
    
    # Evaluate
    agent.evaluate(df_results)
    
    # Save flagged
    os.makedirs("data/processed", exist_ok=True)
    flagged_path = "data/processed/flagged_transactions_demo.csv"
    df_results[df_results.is_flagged].to_csv(flagged_path, index=False)
    logger.info(f"Flagged rows saved to {flagged_path}")
