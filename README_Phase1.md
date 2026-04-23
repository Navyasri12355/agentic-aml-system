# Agentic AML Investigation System - Phase 1

This repository contains the foundation for an **Agentic AI-Based Anti-Money Laundering (AML) Investigation System**. Phase 1 implements a high-performance data ingestion pipeline and a baseline anomaly detection agent designed to process millions of financial transactions.

## 📋 Phase 1 Detailed Report

### 1. Project Overview
Phase 1 established a robust, modular pipeline for data ingestion and a baseline anomaly detection agent. The architecture is designed for scale and is ready for integration into more complex agentic workflows (e.g., LangGraph) in Phase 2.

### 2. Dataset: IBM AMLSim HI-Small
The system processes the **IBM HI-Small** dataset (~4.3 Million transactions), which simulates realistic financial patterns with ground-truth labels for laundering activities.

### 3. Architecture & Implementation
*   **Data Ingestion Pipeline**: Handles normalization, cleaning, and feature engineering (temporal features, log-amounts, cross-border checks). Supports processing in chunks for memory efficiency.
*   **Detection Agent**: Uses an **Isolation Forest** model to identify statistical outliers, combined with heuristic "Flag Reasons" for interpretability.
*   **Testing & Reliability**: Verified with a suite of 15 automated unit tests ensuring data integrity and model stability.
*   **Analysis Suite**: Includes Jupyter notebooks for Exploratory Data Analysis (EDA) and Model Evaluation.

### 4. Metrics & Results (Real Dataset)
- **Total Transactions**: 4,367,359
- **Flag Rate**: 2.00% (87,340 alerts)
- **Results**: The system successfully processes the full dataset, generating detailed anomaly reports in `data/processed/`.

---

## 🚀 Getting Started

Follow these steps to set up and run the system on your local machine.

### 1. Prerequisites
- Python 3.8 or higher
- Git

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/aml-investigation-system.git
cd aml-investigation-system

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset Setup
1. Download the **IBM AMLSim HI-Small** dataset from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
2. Create the raw data directory: `mkdir -p data/raw`
3. Place the `HI-Small_Trans.csv` file inside `data/raw/`.

### 4. Running the Pipeline
You can run the full end-to-end pipeline (Load -> Train -> Detect -> Evaluate -> Save) with a single command:
```bash
python -m src.pipeline.run_phase1
```
*Note: If the raw dataset is missing, the script will automatically fallback to synthetic data for demonstration.*

### 5. Running Tests
Ensure everything is working correctly by running the test suite:
```bash
python -m pytest tests/ -v
```

### 6. Exploratory Analysis
To view the EDA and Evaluation notebooks:
```bash
jupyter notebook notebooks/
```

---

## 📂 Project Structure
- `data/`: Raw, processed, and report data files.
- `models/`: Saved model pipelines (`.joblib`).
- `notebooks/`: Jupyter notebooks for analysis.
- `src/`: Source code for the pipeline and detection agents.
- `tests/`: Unit tests for all modules.

## 🛠️ Tools Used
- **Core**: Python, Pandas, Numpy
- **ML**: Scikit-Learn (Isolation Forest)
- **Testing**: Pytest, Joblib
- **Visualization**: Matplotlib, Seaborn, Jupyter

## ✅ Phase 2 Roadmap
- [ ] Integration with LangGraph for agentic reasoning.
- [ ] Knowledge Graph construction for transaction networks.
- [ ] LLM-assisted investigation reports.
