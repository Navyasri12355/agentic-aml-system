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

## 📋 Phase 3: LangGraph Orchestration & Risk Investigation

### Overview
Phase 3 is now the production orchestration layer for the AML system. It wraps the investigation flow in a **LangGraph-based state machine** and exposes it through both a CLI wrapper and a FastAPI backend. Each flagged transaction is processed through the canonical orchestration runner so the API, CLI, and tests all exercise the same code path.

### Architecture: LangGraph State Machine

The Phase 3 pipeline uses **LangGraph** to manage state and routing across seven nodes:

```
START
    ↓
[detection_node] → Phase 1 cleanup + flagged transaction detection
    ↓
[graph_construction_node] → Build transaction subgraph
    ↓
[feature_extraction_node] → Extract structural features
    ↓
[pattern_detection_node] → Detect AML patterns
    ↓
[risk_scoring_node] → Compute risk score & route
    ↓
[low_risk_exit_node] / [explanation_node]
    ↓
END
```

### Runner-Based Execution

The canonical entry point is `src.orchestration.run.OrchestrationRunner`:

```python
runner = create_runner(enable_debug_logging=False, enable_recovery=True)
result = runner.investigate(
    raw_transaction_path="data/processed/phase1_full_results.csv",
    account_id="ACC_123",
    hop_radius=2,
    time_window_days=30,
    max_neighbors=50,
    contamination=0.02,
)
```

### Node Functions

| Node | Input | Function | Output |
|------|-------|----------|--------|
| **detection_node** | raw transaction path | Loads, cleans, and flags suspicious transactions | clean_df, flagged_df |
| **graph_construction_node** | flagged_row, account_id | Builds the transaction subgraph | subgraph, graph metadata |
| **feature_extraction_node** | subgraph | Extracts graph and temporal features | features, feature statistics |
| **pattern_detection_node** | features | Detects AML patterns | detected_patterns, confidence, severity |
| **risk_scoring_node** | flagged_row, features, patterns | Computes risk score and routing decision | risk_result, routing_decision |
| **low_risk_exit_node** | low-risk result | Generates a minimal exit report | final_report |
| **explanation_node** | risk result | Phase 4 stub for SAR generation | final_report with placeholder narrative |

### Routing Logic

```
IF routing_decision == "EXIT":
    → low_risk_exit_node
ELSE:
    → explanation_node
```

The low-risk path exits early with a compact report. Higher-risk cases continue to the explanation stub, which currently marks the case for Phase 4 SAR generation.

### Input & Output

**Input:**
- `data/processed/phase1_full_results.csv` – Full transaction dataset with anomaly scores
- `data/processed/flagged_hybrid_final.csv` – Flagged transactions from Phase 1 hybrid detection

**Output:**
- `data/processed/phase3_risk_results.json` – Array of final reports from the phase-wise runner

### API Endpoints

The FastAPI service in `src/api/main.py` now exposes the Phase 3 runner directly:

- `POST /investigate/v3` – Single-account investigation using an uploaded CSV
- `POST /investigate/batch` – Batch investigation for a comma-separated account list
- `GET /health` – Health check

**Example Output Entry:**
```json
{
  "account_id": "ACC_789",
  "report_id": "rpt_abc123",
  "risk_score": 0.732,
  "risk_tier": "HIGH",
  "detected_patterns": ["CIRCULAR_FLOW", "SMURFING"],
  "sar_narrative": "[Phase 4 - SAR generation pending]",
  "graph_summary": {
    "node_count": 18,
    "edge_count": 31
  }
}
```

### Error Handling

- **Detection failure** → falls back to an empty flagged set and records the error
- **Graph construction failure** → falls back to a minimal graph
- **Feature/pattern failure** → falls back to empty/default features or `UNCLASSIFIED`
- **Risk scoring failure** → falls back to a neutral MEDIUM risk result
- **Pipeline crash** → generates a fallback final report

### Phase 3 Performance (Sample: rows 575–595)

| Metric | Value |
|--------|-------|
| Transactions Processed | 20 |
| Succeeded | 20 |
| Errors | 0 |
| Routed to explanation stub | 9 |
| Risk Tier Distribution | LOW: 11, HIGH: 9 |

### Running Phase 3

```bash
python -m src.pipeline.run_phase3 --clean-path data/processed/phase1_full_results.csv --flagged-path data/processed/flagged_hybrid_final.csv --start-idx 575 --end-idx 595 --output-path data/processed/phase3_risk_results.json
```

**Configuration options:**
```python
--start-idx / --end-idx     # Slice the flagged dataset
--hop-radius                # Graph expansion depth
--time-window-days          # Historical lookback window
--max-neighbors             # Cap neighbors per node
--contamination             # Detection contamination ratio
```

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
The first phase pipeline (Load → Train → Detect) can be run with a single command:
```bash
python -m src.pipeline.run_phase1
```

The second phase pipeline (Detect → Investigate with multi-agent network analysis) can be run with:
```bash
python -m src.pipeline.run_phase2
```

The third phase pipeline (LangGraph orchestration for risk investigation) can be run with:
```bash
python -m src.pipeline.run_phase3
```

*Note: If the raw dataset is missing, Phase 1 will automatically fallback to synthetic data for demonstration.*

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
- `data/`: AML dataset files and generated outputs.
    - `raw/`: Original IBM HI-Small input files.
    - `processed/`: Phase outputs (`flagged_*`, `phase*_results`, `risk_scored_accounts.json`).
- `docs/`: Prompt engineering and project notes.
- `dummy_codes/`: Experimental/prototype agent implementations.
- `frontend/`: React + Vite UI for investigation workflows.
    - `src/api/client.js`: Frontend API client.
    - `src/App.jsx`: Main UI.
- `models/`: Persisted ML artifacts (`isolation_forest.joblib`, `random_forest.joblib`).
- `notebooks/`: EDA, evaluation notebooks, and notebook model artifacts.
- `src/`: Production backend code.
    - `src/agents/`: Detection, graph, feature, pattern, risk, and explanation agents.
    - `src/orchestration/`: LangGraph state machine, state definitions, and orchestrator runner.
    - `src/pipeline/`: Phase entry points (`run_phase1.py`, `run_phase2.py`, `run_phase3.py`) and ingestion.
    - `src/api/main.py`: FastAPI service exposing investigation endpoints.
- `tests/`: Unit tests for ingestion, graph, detection, and risk logic.
- `requirements.txt`: Python dependency specification.
- `check.py`: Utility script at repository root.

## 🛠️ Tools Used
- **Backend/Core**: Python, Pandas, NumPy, tqdm, python-dotenv
- **Machine Learning**: scikit-learn (Isolation Forest, Random Forest), imbalanced-learn (SMOTE), joblib
- **Graph & Orchestration**: NetworkX, LangGraph
- **API Layer**: FastAPI, Pydantic, CORS middleware
- **Frontend**: React, Vite, Axios, Tailwind CSS, React Flow, Recharts, Zustand
- **Testing & Analysis**: pytest, Jupyter

### Phases
- [x] Phase 1: Data Ingestion + Detection Agent
- [x] Phase 2: Graph Construction + Investigation Agent
- [x] Phase 3: LangGraph Orchestration & Risk Investigation
- [ ] Phase 4: Explanation Agent + SAR Generation
- [ ] Phase 5: Frontend + Evaluation
