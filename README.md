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
Phase 3 integrates the Phase 2 agents into a **LangGraph-based orchestration pipeline** that processes high-risk flagged transactions through a complete investigation workflow. Each flagged transaction flows through a deterministic state machine that chains together graph construction, feature extraction, pattern detection, and risk scoring.

### Architecture: LangGraph State Machine

The Phase 3 pipeline uses **LangGraph** to manage state and routing across five sequential nodes:

```
Entry Point
    ↓
[graph_node] → Build transaction subgraph
    ↓
[feature_node] → Extract structural features
    ↓
[pattern_node] → Detect AML patterns
    ↓
[risk_node] → Compute risk score & route
    ↓
[explanation_node] → Mark for Phase 4 (conditional)
    ↓
END
```

### State Definition (`InvestigationState`)

Each transaction carries state through the pipeline:

```python
{
    "transaction_id": str,
    "account_id": str,
    "flagged_row": dict,
    "graph_result": dict,
    "feature_result": dict,
    "pattern_result": dict,
    "risk_result": dict,
    "routing_decision": str,  # EXIT or INVESTIGATE
    "error": str or None
}
```

### Node Functions

| Node | Input | Function | Output |
|------|-------|----------|--------|
| **graph_node** | `flagged_row`, account_id, timestamp | Builds 2-hop transaction network subgraph | graph, node_count, edge_count, accounts_discovered |
| **feature_node** | graph_result | Extracts 13+ graph/temporal/structural features | in_degree, out_degree, net_flow, velocity, burst_score, cycles |
| **pattern_node** | feature_result | Rule-based pattern matching against 13 AML patterns | detected_patterns[], confidence{}, severity{} |
| **risk_node** | feature_result, pattern_result, flagged_row | Two-stage risk classification | risk_score, risk_tier (HIGH/MEDIUM/LOW), routing_decision |
| **explanation_node** | risk_result | Placeholder for Phase 4 SAR generation | Sets explanation_status: "pending_phase4" |

### Routing Logic

```
IF risk_score >= HIGH_threshold:
    routing_decision = "INVESTIGATE"
    → explanation_node
ELSE:
    routing_decision = "EXIT"
    → END
```

The HIGH threshold varies by flag reason:
- **ML Detection**: 0.48 (aggressive)
- **High Amount**: 0.65 (conservative)
- **Unusual Hour**: 0.70 (very conservative)

### Input & Output

**Input:**
- `data/processed/phase1_full_results.csv` – Full transaction dataset with anomaly scores
- `data/processed/flagged_hybrid_final.csv` – Flagged transactions from Phase 1 hybrid detection

**Output:**
- `data/processed/phase3_risk_results.json` – Array of risk results with transaction_id, account_id, risk_score, risk_tier, routing_decision, detected_patterns

**Example Output Entry:**
```json
{
  "transaction_id": "TXN_12345",
  "account_id": "ACC_789",
  "risk_score": 0.732,
  "risk_tier": "HIGH",
  "routing_decision": "INVESTIGATE",
  "detected_patterns": ["CIRCULAR_FLOW", "SMURFING"],
  "score_components": {
    "anomaly_score": 0.65,
    "pattern_score": 0.90
  }
}
```

### Error Handling

- **Graph construction fails** → graph_node returns error, downstream nodes skip, case logged
- **Missing account** → feature_node returns empty result (all zeros), pattern detection continues
- **Exception during execution** → Caught at node level, error stored in state, transaction logged

### Phase 3 Performance (Sample: rows 575–595)

| Metric | Value |
|--------|-------|
| Transactions Processed | 20 |
| Succeeded | 20 (100%) |
| Errors | 0 |
| Sent to Explanation (HIGH) | 9 |
| Risk Tier Distribution | LOW: 11 (55%), HIGH: 9 (45%) |
| **Recall** | **100.0%** |
| **Precision** | **100.0%** |

### Running Phase 3

```bash
python -m src.pipeline.run_phase3
```

**Configuration (in `run_phase3.py`):**
```python
START_IDX = 575      # Start row in flagged dataset
END_IDX = 595        # End row in flagged dataset
hop_radius = 2       # Network hops (1-2 recommended)
time_window_days = 30  # Historical window
max_neighbors = 50   # Hub control (cap neighbors per node)
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
# On Linux/macOS/WSL:
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
