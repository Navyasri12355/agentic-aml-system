"""
graph.py
--------
Phase 3: LangGraph pipeline definition.

Nodes wrap Phase 1 and 2 agent functions. Conditional routing
is applied after risk_scoring_node based on risk tier.

Graph topology:
    detection
        └── graph_construction
                └── feature_extraction
                        └── pattern_detection
                                └── risk_scoring
                                        ├── (LOW)      ──► low_risk_exit ──► END
                                        └── (MED/HIGH) ──► explanation   ──► END

NOTE: This module is the Phase 3 entry point.
      Phases 1 and 2 modules must be fully working before running this graph.
"""

import logging

from langgraph.graph import StateGraph, END

from src.orchestration.state import AMLAgentState
from src.pipeline.data_ingestion import load_and_clean
from src.agents.detection_agent import run_detection
from src.agents.graph_agent import build_transaction_graph, graph_to_dict
from src.agents.feature_agent import extract_features
from src.agents.pattern_agent import detect_patterns
from src.agents.risk_agent import compute_risk_score
from src.agents.explanation_agent import generate_sar_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def detection_node(state: AMLAgentState) -> AMLAgentState:
    """
    Load raw CSV, clean it, and run Isolation Forest detection.

    Reads : raw_transaction_path
    Writes: clean_df, flagged_df
    """
    logger.info("[Node] detection_node — starting")
    try:
        clean_df = load_and_clean(state["raw_transaction_path"])
        flagged_df = run_detection(clean_df)
        return {**state, "clean_df": clean_df, "flagged_df": flagged_df}
    except Exception as e:
        logger.error(f"detection_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def graph_construction_node(state: AMLAgentState) -> AMLAgentState:
    """
    Build a directed transaction subgraph for the account under investigation.

    Reads : flagged_df, clean_df, account_id, hop_radius, time_window_days
    Writes: subgraph
    """
    logger.info("[Node] graph_construction_node — starting")
    try:
        G = build_transaction_graph(
            flagged_df=state["flagged_df"],
            all_df=state["clean_df"],
            account_id=state["account_id"],
            hop_radius=state.get("hop_radius", 2),
            time_window_days=state.get("time_window_days", 30),
        )
        return {**state, "subgraph": graph_to_dict(G), "_graph_obj": G}
    except Exception as e:
        logger.error(f"graph_construction_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def feature_extraction_node(state: AMLAgentState) -> AMLAgentState:
    """
    Extract topological and temporal features from the subgraph.

    Reads : _graph_obj (internal), account_id, clean_df
    Writes: features
    """
    logger.info("[Node] feature_extraction_node — starting")
    try:
        G = state.get("_graph_obj")
        if G is None:
            raise ValueError("Graph object not found in state — did graph_construction_node run?")

        features = extract_features(
            G=G,
            account_id=state["account_id"],
            df=state.get("clean_df"),
        )
        return {**state, "features": features}
    except Exception as e:
        logger.error(f"feature_extraction_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def pattern_detection_node(state: AMLAgentState) -> AMLAgentState:
    """
    Classify laundering patterns from extracted features.

    Reads : features
    Writes: pattern_result
    """
    logger.info("[Node] pattern_detection_node — starting")
    try:
        pattern_result = detect_patterns(state["features"])
        return {**state, "pattern_result": pattern_result}
    except Exception as e:
        logger.error(f"pattern_detection_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def risk_scoring_node(state: AMLAgentState) -> AMLAgentState:
    """
    Compute weighted risk score and assign risk tier.

    Reads : features, pattern_result, flagged_df (for anomaly score)
    Writes: risk_result
    """
    logger.info("[Node] risk_scoring_node — starting")
    try:
        flagged_df = state.get("flagged_df")
        account_id = state["account_id"]

        # Get the most anomalous score for this account from the flagged df
        if flagged_df is not None and not flagged_df.empty:
            account_rows = flagged_df[
                (flagged_df["sender_id"] == account_id)
                | (flagged_df["receiver_id"] == account_id)
            ]
            anomaly_score = (
                float(account_rows["anomaly_score"].min())
                if not account_rows.empty
                else -0.1
            )
        else:
            anomaly_score = -0.1  # default: mildly anomalous

        risk_result = compute_risk_score(
            features=state["features"],
            pattern_result=state["pattern_result"],
            anomaly_score=anomaly_score,
        )
        return {**state, "risk_result": risk_result}
    except Exception as e:
        logger.error(f"risk_scoring_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def explanation_node(state: AMLAgentState) -> AMLAgentState:
    """
    Generate a SAR narrative via Groq LLM (MEDIUM and HIGH risk cases).

    Reads : account_id, risk_result, features, pattern_result
    Writes: final_report
    """
    logger.info("[Node] explanation_node — starting")
    try:
        risk_result = state["risk_result"]
        final_report = generate_sar_report(
            account_id=state["account_id"],
            risk_score=risk_result["risk_score"],
            risk_tier=risk_result["risk_tier"],
            features=state["features"],
            pattern_result=state["pattern_result"],
            risk_result=risk_result,
        )
        return {**state, "final_report": final_report}
    except Exception as e:
        logger.error(f"explanation_node failed: {e}")
        return {**state, "errors": state.get("errors", []) + [str(e)]}


def low_risk_exit_node(state: AMLAgentState) -> AMLAgentState:
    """
    Generate a minimal exit summary for LOW risk accounts. No LLM call.

    Reads : account_id, risk_result
    Writes: final_report
    """
    logger.info("[Node] low_risk_exit_node — account cleared as low risk")
    from datetime import datetime, timezone

    risk_result = state.get("risk_result", {})
    final_report = {
        "account_id": state["account_id"],
        "risk_score": risk_result.get("risk_score", 0.0),
        "risk_tier": "LOW",
        "detected_patterns": [],
        "sar_narrative": None,
        "exit_summary": (
            "Transaction analysis complete. Risk score below threshold. "
            "No suspicious laundering patterns detected. "
            "No further investigation required at this time."
        ),
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": None,
    }
    return {**state, "final_report": final_report}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_risk_scoring(state: AMLAgentState) -> str:
    """
    Conditional edge: route based on risk tier.

    Returns:
        "low_risk_exit" if tier is LOW
        "explanation"   if tier is MEDIUM or HIGH
    """
    risk_result = state.get("risk_result", {})
    routing = risk_result.get("routing_decision", "INVESTIGATE")
    return "low_risk_exit" if routing == "EXIT" else "explanation"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_aml_graph() -> StateGraph:
    """
    Assemble and return the compiled LangGraph AML investigation pipeline.

    Returns:
        Compiled LangGraph application ready for .invoke() or .stream()
    """
    workflow = StateGraph(AMLAgentState)

    # Register nodes
    workflow.add_node("detection",          detection_node)
    workflow.add_node("graph_construction", graph_construction_node)
    workflow.add_node("feature_extraction", feature_extraction_node)
    workflow.add_node("pattern_detection",  pattern_detection_node)
    workflow.add_node("risk_scoring",       risk_scoring_node)
    workflow.add_node("explanation",        explanation_node)
    workflow.add_node("low_risk_exit",      low_risk_exit_node)

    # Entry point
    workflow.set_entry_point("detection")

    # Linear edges
    workflow.add_edge("detection",          "graph_construction")
    workflow.add_edge("graph_construction", "feature_extraction")
    workflow.add_edge("feature_extraction", "pattern_detection")
    workflow.add_edge("pattern_detection",  "risk_scoring")

    # Conditional split after risk scoring
    workflow.add_conditional_edges(
        "risk_scoring",
        route_after_risk_scoring,
        {
            "low_risk_exit": "low_risk_exit",
            "explanation":   "explanation",
        },
    )

    # Terminal edges
    workflow.add_edge("low_risk_exit", END)
    workflow.add_edge("explanation",   END)

    return workflow.compile()


# Singleton compiled graph — import this in run.py and main.py
aml_pipeline = build_aml_graph()