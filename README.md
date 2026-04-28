# Agentic AI-Based AML Investigation System
## RV College of Engineering - Experiential Learning 2025-26

### Project Overview
Multi-agent pipeline for automated Anti-Money Laundering investigation.
Goes beyond detection to automate graph construction, pattern recognition,
risk scoring, and SAR report generation.

### Phase 1 Status: COMPLETE ✅

#### What Phase 1 Does
1. Ingests IBM AMLSim HI-Small dataset (4.3M transactions)
2. Normalizes and cleans data
3. Runs Hybrid IF+SMOTE RF detection
4. Outputs flagged transactions for Phase 2

#### Phase 1 Results
| Metric | Value |
|--------|-------|
| Transactions processed | 4,367,359 |
| Flagged for Phase 2 | 923,512 |
| Laundering caught | 3,194 / 5,110 (62.5%) |
| Recall | 0.6250 |
| FPR | 0.2110 |

### Setup

1. Clone repo:
```
git clone https://github.com/Navyasri12355/agentic-aml-system.git
cd agentic-aml-system
```

3. Create virtual environment:
```
python -m venv venv
venv\\Scripts\\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

5. Install dependencies:
```
pip install -r requirements.txt
```

7. Download dataset:
Download HI-Small_Trans.csv from:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml [https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml]
Place at: data/raw/HI-Small_Trans.csv

### Running Phase 1

Run full pipeline:
```
python -m src.pipeline.run_phase1
```

Run tests:
```
python -m pytest tests/ -v
```

Run EDA notebook:
jupyter notebook:
notebooks/phase1_eda.ipynb [notebooks/phase1_eda.ipynb]

Run detection eval notebook:
jupyter notebook:
notebooks/phase1_detection_eval.ipynb [notebooks/phase1_detection_eval.ipynb]

### Output Files
| File | Description |
|------|-------------|
| data/processed/flagged_hybrid_final.csv [data/processed/flagged_hybrid_final.csv] | Phase 2 input |
| data/processed/flagged_if_baseline.csv [data/processed/flagged_if_baseline.csv] | IF baseline comparison |
| data/processed/phase1_full_results.csv [data/processed/phase1_full_results.csv] | Full dataset with scores |
| models/isolation_forest.joblib [models/isolation_forest.joblib] | Trained IF model |
| models/random_forest.joblib [models/random_forest.joblib] | Trained RF model |

### Tech Stack
- Python 3.x
- pandas, numpy, scikit-learn
- imbalanced-learn (SMOTE)
- joblib, pytest, jupyter, matplotlib, seaborn

### Team
| Name | USN | Branch |
|------|-----|--------|
| Angela Varghese | 1RV23IS014 | ISE |
| Navyasri Pulipati | 1RV23AI065 | AIML |
| Sridula O S | 1RV23CS303 | CSE |
| Suraj Sreedhara | 1RV23CS257 | CSE |
| Thushitha R | 1RV23CY057 | CY |

Mentor: Dr. Narasimha Swamy S

### Phases
- [x] Phase 1: Data Ingestion + Detection Agent
- [ ] Phase 2: Graph Construction + Investigation Agent
- [ ] Phase 3: LangGraph Orchestration
- [ ] Phase 4: Explanation Agent + SAR Generation
- [ ] Phase 5: Frontend + Evaluation
