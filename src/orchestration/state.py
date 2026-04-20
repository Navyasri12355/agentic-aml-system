"""
state.py
--------
Phase 3: Shared agent state definition for the LangGraph pipeline.

AMLAgentState is a TypedDict that flows through every node in the graph.
Each node reads from and writes to this shared state.
No node modifies keys it does not own.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class AMLAgentState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Keys are grouped by the phase/node that writes them.
    All keys are Optional so the state can be initialised sparsely.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    raw_transaction_path: str          # absolute or relative path to input CSV
    account_id: str                    # account under investigation
    hop_radius: int                    # graph expansion depth (default: 2)
    time_window_days: int              # days of history to include (default: 30)

    # ── Phase 1 — Detection Agent ──────────────────────────────────────────
    clean_df: Optional[Any]            # pd.DataFrame — cleaned transactions
    flagged_df: Optional[Any]          # pd.DataFrame — suspicious transactions
                                       #   columns: transaction_id, sender_id,
                                       #            receiver_id, amount, timestamp,
                                       #            anomaly_score, is_flagged,
                                       #            flag_reason

    # ── Phase 2 — Graph + Feature + Pattern + Risk Agents ─────────────────
    subgraph: Optional[dict]           # serialised graph: {nodes: [...], edges: [...]}
    features: Optional[dict]          # feature dict from feature_agent
    pattern_result: Optional[dict]    # {detected_patterns, pattern_confidence, is_isolated}
    risk_result: Optional[dict]       # {risk_score, risk_tier, routing_decision,
                                       #  score_components}

    # ── Phase 4 — Explanation Agent ────────────────────────────────────────
    final_report: Optional[dict]       # complete investigation output including
                                       # sar_narrative or exit_summary

    # ── Metadata ───────────────────────────────────────────────────────────
    errors: list[str]                  # non-fatal error accumulator


def initial_state(
    raw_transaction_path: str,
    account_id: str,
    hop_radius: int = 2,
    time_window_days: int = 30,
) -> AMLAgentState:
    """
    Return a correctly initialised AMLAgentState for a new investigation.

    Args:
        raw_transaction_path: Path to input CSV.
        account_id:           Account to investigate.
        hop_radius:           Graph expansion depth.
        time_window_days:     Temporal context window in days.

    Returns:
        AMLAgentState with input fields set and all others None/empty.
    """
    return AMLAgentState(
        raw_transaction_path=raw_transaction_path,
        account_id=account_id,
        hop_radius=hop_radius,
        time_window_days=time_window_days,
        clean_df=None,
        flagged_df=None,
        subgraph=None,
        features=None,
        pattern_result=None,
        risk_result=None,
        final_report=None,
        errors=[],
    )