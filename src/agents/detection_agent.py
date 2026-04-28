import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectionAgent:
    def __init__(self, contamination: float = 0.02, 
                 model_path: str = "models/isolation_forest.joblib",
                 flag_threshold: float = 0.0):
        self.contamination = contamination
        self.model_path = model_path
        self.flag_threshold = flag_threshold
        self.pipeline: Optional[Pipeline] = None
        
    def _get_preprocessor(self) -> ColumnTransformer:
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
        if os.path.exists(self.model_path) and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            self.load_model()
            return
            
        logger.info(f"Training Isolation Forest model. Shape: {df.shape}")
        
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

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            self.load_model()
            
        df_detect = df.copy()
        df_detect['is_cross_border'] = df_detect['is_cross_border'].astype(int)
        
        df_detect['anomaly_score'] = self.pipeline.decision_function(df_detect)
        df_detect['is_flagged'] = df_detect['anomaly_score'] < self.flag_threshold
        
        amount_95th = df_detect['amount_log'].quantile(0.95) if len(df_detect) > 0 else 0
        amount_median = df_detect['amount'].median() if len(df_detect) > 0 else 0
        
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
        df_detect['is_flagged'] = df_detect['is_flagged'].astype(bool)
        
        return df_detect

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if 'is_laundering' not in df.columns:
            return {}
            
        y_true = df['is_laundering']
        y_pred = df['is_flagged'].astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(y_true.unique()) > 1 else (0,0,0,0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'false_positive_rate': float(fpr),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'flagged_count': int(y_pred.sum()),
            'total_count': int(len(df)),
            'flag_rate': float(y_pred.mean()) if len(y_pred) > 0 else 0.0
        }
        return metrics

    def save_model(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)

    def load_model(self) -> None:
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No model found at {self.model_path}")

# ================================================
# HybridDetectionAgent - PRODUCTION MODEL
# IF + SMOTE Random Forest
# rf_threshold=0.6 selected via threshold sweep:
# 
# Threshold | Caught | Recall | FPR
# 0.5       | 4,019  | 0.7865 | 0.3449
# 0.6       | 3,194  | 0.6250 | 0.2110  ← SELECTED
# 0.7       | 2,023  | 0.3959 | 0.0968
# 0.8       | 1,091  | 0.2135 | 0.0473
# 0.9       |   172  | 0.0337 | 0.0217
#
# Result: 3,194/5,110 laundering cases caught (62.5%)
# 390x recall improvement over IF alone
# ================================================
class HybridDetectionAgent(DetectionAgent):
    def __init__(self, contamination: float = 0.02,
                 model_path: str = "models/isolation_forest.joblib", 
                 rf_model_path: str = "models/random_forest.joblib",
                 rf_threshold: float = 0.6,
                 flag_threshold: float = 0.0):
        super().__init__(contamination=contamination, model_path=model_path, flag_threshold=flag_threshold)
        self.rf_model_path = rf_model_path
        self.rf_model = None
        self.rf_threshold = rf_threshold

    def train_supervised(self, df: pd.DataFrame, force_retrain: bool = False) -> None:
        if os.path.exists(self.rf_model_path) and not force_retrain:
            logger.info(f"Loading existing RF model from {self.rf_model_path}")
            self.rf_model = joblib.load(self.rf_model_path)
            return

        if self.pipeline is None:
            self.train(df, force_retrain=force_retrain)
            
        preprocessor = self.pipeline.named_steps['preprocessor']
        X = preprocessor.transform(df)
        y = df['is_laundering']
        
        smote = SMOTE(sampling_strategy=0.1, random_state=42)
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
        except ValueError as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            X_resampled, y_resampled = X, y

        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            class_weight='balanced',
            random_state=42, 
            n_jobs=-1
        )
        self.rf_model.fit(X_resampled, y_resampled)
        
        os.makedirs(os.path.dirname(self.rf_model_path), exist_ok=True)
        joblib.dump(self.rf_model, self.rf_model_path)

    def train_all(self, df: pd.DataFrame, force_retrain: bool = False) -> None:
        self.train(df, force_retrain=force_retrain)
        self.train_supervised(df, force_retrain=force_retrain)

    def detect_hybrid(self, df: pd.DataFrame) -> pd.DataFrame:
        df_res = self.detect(df)
        
        if self.rf_model is None:
            self.rf_model = joblib.load(self.rf_model_path)
            
        preprocessor = self.pipeline.named_steps['preprocessor']
        X = preprocessor.transform(df)
        
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        
        for i in range(len(df_res)):
            rf_prob = rf_probs[i]
            rf_flag = rf_prob > self.rf_threshold
            if_flag = df_res.at[df_res.index[i], 'is_flagged']
            
            if rf_flag and not if_flag:
                df_res.at[df_res.index[i], 'flag_reason'] = "Random Forest detection"
            elif not rf_flag and not if_flag:
                df_res.at[df_res.index[i], 'flag_reason'] = ""
                
            df_res.at[df_res.index[i], 'is_flagged'] = if_flag or rf_flag
            
        return df_res

if __name__ == "__main__":
    from src.pipeline.data_ingestion import generate_synthetic_data, load_and_clean, normalize_ibm_amlsim
    import tempfile

    print("Running Detection Agent Demo (synthetic data)...")
    synthetic_raw = generate_synthetic_data(500)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        synthetic_raw.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
        
    df_norm = normalize_ibm_amlsim(tmp_path)
    df_processed = load_and_clean(df_norm)
    os.remove(tmp_path)
    
    # Run DetectionAgent
    print("\n--- Isolation Forest Baseline ---")
    if_agent = DetectionAgent(contamination=0.02)
    if_agent.train(df_processed, force_retrain=True)
    df_if = if_agent.detect(df_processed)
    if_metrics = if_agent.evaluate(df_if)
    print(if_metrics)
    
    # Run HybridDetectionAgent
    print("\n--- Hybrid Detection Agent ---")
    hybrid_agent = HybridDetectionAgent(contamination=0.02, rf_threshold=0.6)
    hybrid_agent.train_all(df_processed, force_retrain=True)
    df_hybrid = hybrid_agent.detect_hybrid(df_processed)
    hybrid_metrics = hybrid_agent.evaluate(df_hybrid)
    print(hybrid_metrics)
    
    print("\n--- Side by Side Comparison ---")
    print(f"IF Recall: {if_metrics.get('recall', 0):.4f} | IF FPR: {if_metrics.get('false_positive_rate', 0):.4f}")
    print(f"Hybrid Recall: {hybrid_metrics.get('recall', 0):.4f} | Hybrid FPR: {hybrid_metrics.get('false_positive_rate', 0):.4f}")
