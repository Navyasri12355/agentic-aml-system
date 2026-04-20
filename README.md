# Agentic AI-Based Anti Money Laundering Investigation System

**Team 38 — RV College of Engineering, VI Semester Experiential Learning, EVEN 2025–26**

Mentor: Dr. Narasimha Swamy S, Department of AIML

---

## Overview

A multi-agent pipeline that goes beyond transaction detection to automate the full AML investigation workflow — graph construction, pattern recognition, risk scoring, and explainable SAR report generation — orchestrated via LangGraph.

```
CSV Transactions
      │
      ▼
[Detection Agent]        ← Isolation Forest anomaly detection
      │
      ▼
[Graph Agent]            ← NetworkX directed transaction graph
      │
      ▼
[Feature Agent]          ← Topological + temporal features
      │
      ▼
[Pattern Agent]          ← Funneling, Scattering, Circular, Layering
      │
      ▼
[Risk Scoring Agent]     ← Weighted 0–1 risk score + tier
      │
   ┌──┴──┐
  LOW   MED/HIGH
   │      │
   ▼      ▼
 Exit  [Explanation Agent]  ← Groq LLM → SAR narrative
          │
          ▼
    FastAPI Backend
          │
          ▼
    React + Vite UI
```

---

## Project Structure

```
aml-investigation-system/
├── data/
│   ├── raw/                    # place downloaded datasets here (gitignored)
│   ├── processed/              # pipeline outputs (gitignored)
│   └── reports/                # generated SAR JSON reports
├── models/                     # saved model artifacts (gitignored)
├── notebooks/                  # EDA and evaluation notebooks
├── src/
│   ├── pipeline/
│   │   └── data_ingestion.py   # data loading, cleaning, feature engineering
│   ├── agents/
│   │   ├── detection_agent.py  # Isolation Forest anomaly detection
│   │   ├── graph_agent.py      # NetworkX graph construction + context expansion
│   │   ├── feature_agent.py    # topological and temporal feature extraction
│   │   ├── pattern_agent.py    # laundering pattern classification
│   │   ├── risk_agent.py       # weighted risk scoring and tier assignment
│   │   └── explanation_agent.py# Groq LLM SAR report generation
│   ├── orchestration/
│   │   ├── state.py            # AMLAgentState TypedDict
│   │   ├── graph.py            # LangGraph node and edge definitions
│   │   └── run.py              # pipeline entry point
│   └── api/
│       └── main.py             # FastAPI application
├── frontend/                   # React + Vite investigator UI
├── docs/                       # architecture diagrams and notes
├── tests/                      # unit tests per module
├── .env.example                # environment variable template
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- A [Groq API key](https://console.groq.com)

### 1. Clone and set up Python environment

```bash
git clone <repo-url>
cd aml-investigation-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY
```

### 3. Add your dataset

Download the IBM AMLSim dataset (HI-Small recommended) and place the CSV in:

```
data/raw/transactions.csv
```

Expected columns: `transaction_id, timestamp, sender_id, receiver_id, amount, transaction_type, sender_country, receiver_country, is_laundering`

### 4. Run the backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

UI available at: `http://localhost:5173`

---

## Running the Pipeline Directly

```bash
python src/orchestration/run.py \
  --file data/raw/transactions.csv \
  --account ACC_000123 \
  --hops 2 \
  --window 30
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Phase Progress

| Phase | Description | Status |
|---|---|---|
| 1 | Data Foundation & Detection Agent | 🔲 Not started |
| 2 | Graph Construction & Investigation Agent | 🔲 Not started |
| 3 | LangGraph Orchestration | 🔲 Not started |
| 4 | Explanation Agent & SAR Generation | 🔲 Not started |
| 5 | Frontend + Evaluation | 🔲 Not started |

