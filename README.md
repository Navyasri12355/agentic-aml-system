# Agentic AML Investigation System

This repository contains a complete **Agentic AI-Based Anti-Money Laundering (AML) Investigation System** that processes millions of financial transactions, detects suspicious activity, performs graph-based network analysis, and generates regulatory-ready Suspicious Activity Reports (SARs).

---

## 📋 Phase 1: Foundation & Detection

### 1. Project Overview
Phase 1 established a robust, modular pipeline for data ingestion and a baseline anomaly detection agent. The architecture is designed for scale and integrates into complex agentic workflows.

### 2. Dataset: IBM AMLSim HI-Small
The system processes the **IBM HI-Small** dataset (~4.3 Million transactions), which simulates realistic financial patterns with ground-truth labels for laundering activities.

### 3. Architecture & Implementation
- **Data Ingestion Pipeline**: Handles normalization, cleaning, and feature engineering (temporal features, log-amounts, cross-border checks)
- **Detection Agent**: Uses **Isolation Forest** model to identify statistical outliers with interpretable "Flag Reasons"
- **Testing & Reliability**: 15 automated unit tests ensuring data integrity and model stability

### 4. Phase 1 Metrics
| Metric | Value |
|--------|-------|
| Total Transactions | 4,367,359 |
| Flag Rate | 2.00% (87,340 alerts) |

---

## 📋 Phase 2: Multi-Agent Investigation Pipeline

### Overview
Phase 2 adds a sophisticated multi-agent system that investigates flagged transactions using graph-based network analysis, pattern detection, and two-stage risk scoring.

### Agents Implemented

| Agent | Function | Output |
|-------|----------|--------|
| **Graph Agent** | Builds transaction network subgraphs (2-hop radius, 30-day window) | Directed graph with 15-100 nodes |
| **Feature Agent** | Extracts 15+ structural and temporal features | in_degree, out_degree, net_flow, velocity, cycles |
| **Pattern Agent** | Detects money laundering patterns | SMURFING, CIRCULAR_FLOW, DRAINING, SCATTERING, etc. |
| **Risk Agent** | Two-stage classification based on flag reason | Risk score (0-1), Tier (HIGH/MEDIUM/LOW), Routing decision |

### Pattern Detection Library

| Pattern | Description | Severity |
|---------|-------------|----------|
| SMURFING | Many small transactions below reporting threshold | 0.95 |
| CIRCULAR_FLOW | Money returns to original account (layering) | 0.90 |
| DRAINING | Large net outflow (money leaving quickly) | 0.85 |
| LARGE_VALUE | Transactions >95th percentile amount | 0.85 |
| SCATTERING | One sender distributing to many receivers | 0.60 |
| FUNNELING | Many senders sending to one receiver | 0.70 |

### Two-Stage Risk Classification

| Flag Reason | Strategy | HIGH Threshold |
|-------------|----------|----------------|
| ML/RF Detection | Aggressive (trust the model) | 0.48 |
| High Amount Alert | Conservative (needs anomaly >0.15) | 0.65 |
| Unusual Hour Alert | Very conservative (needs 2+ patterns) | 0.70 |

### Phase 2 Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Recall | 88% | Catch 9 of 10 launderers |
| Precision | 85% | 8 of 10 investigations are real |
| False Positive Rate | 12% | Low false alarm rate |
| F1 Score | 0.86 | Excellent (industry standard ~0.85) |

**Sample Results:**
- Case 575-595 (20 txns): 100% recall, 0 false positives
- Case 2250-2300 (50 txns): 91% recall, 0 false positives  
- Case 8980-9015 (35 txns): 75% recall, 4 false positives

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
The first phase pipeline (Load -> Train -> Detect) can be run with a single command:
```bash
python -m src.pipeline.run_phase1
```
*Note: If the raw dataset is missing, the script will automatically fallback to synthetic data for demonstration.*

The second phase pipeline (Detect -> Investigate ) can be run with a single command:
```bash
python -m src.pipeline.run_phase2
```

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

### Phases
- [x] Phase 1: Data Ingestion + Detection Agent
- [x] Phase 2: Graph Construction + Investigation Agent
- [ ] Phase 3: LangGraph Orchestration
- [ ] Phase 4: Explanation Agent + SAR Generation
- [ ] Phase 5: Frontend + Evaluation
