# File: src/pipeline/state_phase4.py

from typing import TypedDict, Optional

class InvestigationState(TypedDict):
    # Input
    transaction_id: str
    account_id: str
    flagged_row: dict

    # Agent outputs
    graph_result: Optional[dict]
    feature_result: Optional[dict]
    pattern_result: Optional[dict]
    risk_result: Optional[dict]
    explanation_result: Optional[dict]

    # Control flow
    routing_decision: str  # "INVESTIGATE" | "EXIT"
    error: Optional[str]
