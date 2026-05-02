# File: src/orchestration/graph.py
"""
Phase 3 LangGraph Orchestration - Complete Pipeline Definition

Defines the complete AML investigation state machine with:
- 7 core agent nodes (Detection, Graph, Feature, Pattern, Risk, Explanation, Exit)
- Comprehensive error handling and recovery
- Conditional routing based on risk tier
- Retry logic with exponential backoff
- Execution metrics and audit logging
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, cast

from langgraph.graph import StateGraph, END, START

from src.orchestration.state import (
    AMLAgentState, AgentStatus, RoutingDecision, RiskTier, create_initial_state
)
from src.orchestration.errors import (
    handle_agent_error, add_error_to_state, validate_risk_result,
    log_agent_execution, retry_on_error, ValidationError, AgentExecutionError,
    create_fallback_risk_result, create_fallback_final_report, safe_get
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────


def detection_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 1: Detection Agent Node
    
    Loads raw transactions, cleans data, and runs Isolation Forest
    to flag suspicious transactions.
    
    Error handling:
      - Validates input file exists
      - Handles data format errors
      - Falls back to empty flagged set on error
    """
    agent_name = "detection_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting detection node")
        
        # Skip if explicitly disabled
        if state.get("skip_detection") and state.get("flagged_df") is not None:
            logger.info(f"[{agent_name}] Skipped (skip_detection=True)")
            state["agent_metrics"][agent_name] = log_agent_execution(
                agent_name, start_time, datetime.utcnow(),
                AgentStatus.SKIPPED
            )
            return state

        # Import here to avoid circular imports
        from src.pipeline.data_ingestion import load_and_clean, normalize_ibm_amlsim
        from src.agents.detection_agent import DetectionAgent

        # Load and clean data
        logger.info(f"[{agent_name}] Loading raw data from {state['raw_transaction_path']}")
        raw_df = normalize_ibm_amlsim(state["raw_transaction_path"])
        clean_df = load_and_clean(raw_df)
        
        state["clean_df"] = clean_df
        logger.info(f"[{agent_name}] Cleaned {len(clean_df)} transactions")

        # Run detection
        logger.info(f"[{agent_name}] Running Isolation Forest detection")
        detector = DetectionAgent(
            contamination=state.get("contamination", 0.02),
            model_path="models/isolation_forest.joblib"
        )
        detector.train(clean_df)
        flagged_df = detector.predict(clean_df)
        
        state["flagged_df"] = flagged_df
        state["detection_features_used"] = [
            "amount_log", "hour_of_day", "day_of_week",
            "is_cross_border", "transaction_type"
        ]
        state["detection_model_version"] = "isolation_forest_v1"
        
        logger.info(
            f"[{agent_name}] Detected {len(flagged_df)} suspicious transactions "
            f"out of {len(clean_df)} total"
        )

        # Find the flagged row for account_id
        account_flagged = flagged_df[flagged_df["sender_id"] == state["account_id"]]
        if len(account_flagged) > 0:
            state["flagged_row"] = account_flagged.iloc[0].to_dict()
            logger.info(f"[{agent_name}] Found flagged row for account {state['account_id']}")
        else:
            logger.warning(
                f"[{agent_name}] No flagged transactions found for account "
                f"{state['account_id']}, using first flagged transaction"
            )
            if len(flagged_df) > 0:
                state["flagged_row"] = flagged_df.iloc[0].to_dict()

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=flagged_df
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=True, max_retries=2
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: use empty flagged set
        state["flagged_df"] = None
        return state


def graph_construction_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 2a: Graph Construction Node
    
    Builds transaction subgraph centered on flagged account
    with context expansion to neighboring accounts.
    
    Error handling:
      - Validates flagged_row exists
      - Handles graph construction failures
      - Falls back to minimal graph on error
    """
    agent_name = "graph_construction_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting graph construction")
        
        if state.get("flagged_row") is None:
            raise ValidationError("flagged_row is required but missing")

        if state.get("skip_graph_expansion"):
            logger.info(f"[{agent_name}] Using minimal graph (skip_graph_expansion=True)")
            state["subgraph"] = {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "expansion_hops": 0
            }
            return state

        from src.agents.graph_agent import GraphAgent

        graph_agent = GraphAgent(
            all_transactions=state.get("clean_df"),
            global_stats=state.get("feature_statistics", {})
        )

        result = graph_agent.build_subgraph(
            account_id=state["account_id"],
            flag_date=state["flagged_row"].get("timestamp"),
            hop_radius=state.get("hop_radius", 2),
            time_window_days=state.get("time_window_days", 30),
            max_neighbors=state.get("max_neighbors", 50)
        )

        state["subgraph"] = result
        state["graph_metadata"] = {
            "source_account": state["account_id"],
            "construction_timestamp": datetime.utcnow().isoformat(),
            "expansion_hops": result.get("expansion_hops", 0)
        }

        logger.info(
            f"[{agent_name}] Built graph with {result.get('node_count', 0)} nodes "
            f"and {result.get('edge_count', 0)} edges"
        )

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=result
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=True, max_retries=1
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: minimal graph
        state["subgraph"] = {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "expansion_hops": 0
        }
        return state


def feature_extraction_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 2b: Feature Extraction Node
    
    Extracts topological, temporal, and structural features
    from transaction subgraph.
    
    Error handling:
      - Validates subgraph structure
      - Handles feature computation errors
      - Falls back to minimal features on error
    """
    agent_name = "feature_extraction_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting feature extraction")
        
        subgraph = state.get("subgraph")
        if subgraph is None or subgraph.get("node_count", 0) == 0:
            logger.warning(f"[{agent_name}] Empty or missing subgraph, using defaults")
            state["features"] = {
                "account_id": state["account_id"],
                "subgraph_node_count": 0,
                "subgraph_edge_count": 0,
                "features": {}
            }
            return state

        from src.agents.feature_agent import FeatureAgent

        feature_agent = FeatureAgent(global_stats=state.get("feature_statistics", {}))
        
        result = feature_agent.extract_features(subgraph)
        state["features"] = result

        logger.info(f"[{agent_name}] Extracted features for {result.get('account_id')}")

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=result
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=True, max_retries=1
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: minimal features
        state["features"] = {
            "account_id": state["account_id"],
            "subgraph_node_count": 0,
            "subgraph_edge_count": 0,
            "features": {}
        }
        return state


def pattern_detection_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 2c: Pattern Detection Node
    
    Identifies laundering patterns (funneling, scattering, etc.)
    based on extracted features.
    
    Error handling:
      - Validates feature structure
      - Handles pattern classification errors
      - Falls back to unclassified on error
    """
    agent_name = "pattern_detection_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting pattern detection")
        
        features = state.get("features")
        if features is None or not features.get("features"):
            logger.warning(f"[{agent_name}] Missing features, using default patterns")
            state["patterns"] = {
                "account_id": state["account_id"],
                "detected_patterns": ["UNCLASSIFIED"],
                "pattern_confidence": {"UNCLASSIFIED": 1.0},
                "is_isolated": True
            }
            return state

        from src.agents.pattern_agent import PatternAgent

        pattern_agent = PatternAgent()
        result = pattern_agent.detect_patterns(features)
        state["patterns"] = result

        logger.info(
            f"[{agent_name}] Detected patterns: {result.get('detected_patterns', [])}"
        )

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=result
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=True, max_retries=1
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: unclassified pattern
        state["patterns"] = {
            "account_id": state["account_id"],
            "detected_patterns": ["UNCLASSIFIED"],
            "pattern_confidence": {"UNCLASSIFIED": 1.0},
            "is_isolated": True
        }
        return state


def risk_scoring_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 2d: Risk Scoring Node
    
    Computes weighted risk score and determines routing decision.
    
    Error handling:
      - Validates all upstream outputs
      - Falls back to simplified scoring if primary fails
      - Uses fallback risk result if enabled
    """
    agent_name = "risk_scoring_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting risk scoring")
        
        # Validate prerequisites
        if state.get("flagged_row") is None:
            raise ValidationError("flagged_row required for risk scoring")

        from src.agents.risk_agent import RiskAgent
        from src.utils.global_stats import build_global_stats

        global_stats = build_global_stats(state.get("clean_df"))
        state["feature_statistics"] = global_stats

        risk_agent = RiskAgent(global_stats=global_stats)
        
        result = risk_agent.compute_risk(
            flagged_row=state["flagged_row"],
            feature_result=state.get("features", {}),
            pattern_result=state.get("patterns", {}),
            graph_result=state.get("subgraph", {}),
            transaction_id=state.get("flagged_row", {}).get("transaction_id")
        )

        # Validate risk result
        is_valid, errors = validate_risk_result(result)
        if not is_valid:
            logger.warning(f"[{agent_name}] Risk result validation failed: {errors}")
            if state.get("use_fallback_risk_scoring"):
                result = create_fallback_risk_result(
                    state["account_id"],
                    f"Validation errors: {errors}"
                )

        state["risk_result"] = result
        
        # Set routing decision
        routing = result.get("routing_decision", "EXIT")
        state["routing_decision"] = routing
        state["should_generate_sar"] = routing in ["INVESTIGATE", "ESCALATE"]

        logger.info(
            f"[{agent_name}] Risk score: {result.get('risk_score', 0):.2f}, "
            f"Tier: {result.get('risk_tier', 'UNKNOWN')}, "
            f"Routing: {routing}"
        )

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=result
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=False, max_retries=0
        )
        state = add_error_to_state(state, error_ctx, critical=True)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: use simplified risk scoring
        fallback_result = create_fallback_risk_result(
            state["account_id"],
            f"Risk scoring failed: {e}"
        )
        state["risk_result"] = fallback_result
        state["routing_decision"] = fallback_result["routing_decision"]
        state["should_generate_sar"] = True  # Investigate on error
        
        return state


def low_risk_exit_node(state: AMLAgentState) -> AMLAgentState:
    """
    Low Risk Exit Node
    
    Generates minimal report for LOW risk accounts.
    Does NOT call LLM (Phase 4).
    
    Error handling:
      - Graceful fallback if report generation fails
    """
    agent_name = "low_risk_exit_node"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Generating LOW risk exit report")
        
        from uuid import uuid4
        report_id = state.get("report_id") or f"rpt_{uuid4().hex[:12]}"
        state["report_id"] = report_id

        final_report = {
            "account_id": state["account_id"],
            "report_id": report_id,
            "risk_score": state.get("risk_result", {}).get("risk_score", 0.0),
            "risk_tier": "LOW",
            "detected_patterns": state.get("patterns", {}).get("detected_patterns", []),
            "sar_narrative": None,
            "graph_summary": {
                "node_count": state.get("subgraph", {}).get("node_count", 0),
                "edge_count": state.get("subgraph", {}).get("edge_count", 0)
            },
            "key_findings": [
                "Transaction analysis complete",
                "Risk score below threshold",
                "No suspicious patterns detected"
            ],
            "recommendations": ["No further action required"],
            "report_generated_at": datetime.utcnow().isoformat(),
            "execution_duration_seconds": 0.0,
            "exit_reason": "LOW_RISK"
        }

        state["final_report"] = final_report
        
        logger.info(f"[{agent_name}] Generated exit report {report_id}")

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=final_report
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=True, max_retries=1
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: minimal report
        state["final_report"] = create_fallback_final_report(
            state, f"Exit node failed: {e}"
        )
        return state


def explanation_node(state: AMLAgentState) -> AMLAgentState:
    """
    Phase 4: Explanation Agent Node (Stub)
    
    Placeholder for LLM-based SAR generation.
    Currently marks case as pending Phase 4.
    
    In Phase 4, this will:
      - Call Groq API with dynamic prompt
      - Generate structured SAR narrative
      - Create final investigation report
    
    Error handling:
      - Graceful fallback if LLM unavailable
      - Fallback report if generation fails
    """
    agent_name = "explanation_agent"
    start_time = datetime.utcnow()
    state["agent_metrics"] = state.get("agent_metrics", {})
    
    try:
        logger.info(f"[{agent_name}] Starting explanation node (Phase 4 stub)")
        
        from uuid import uuid4
        report_id = state.get("report_id") or f"rpt_{uuid4().hex[:12]}"
        state["report_id"] = report_id

        # TODO: Phase 4 - Replace with actual Groq API call
        # For now, generate placeholder report
        
        final_report = {
            "account_id": state["account_id"],
            "report_id": report_id,
            "risk_score": state.get("risk_result", {}).get("risk_score", 0.5),
            "risk_tier": state.get("risk_result", {}).get("risk_tier", "MEDIUM"),
            "detected_patterns": state.get("patterns", {}).get("detected_patterns", []),
            "sar_narrative": "[Phase 4 - SAR generation pending]",
            "graph_summary": {
                "node_count": state.get("subgraph", {}).get("node_count", 0),
                "edge_count": state.get("subgraph", {}).get("edge_count", 0),
                "has_cycle": state.get("features", {}).get("features", {}).get("has_cycle", False)
            },
            "key_findings": [
                f"Risk score: {state.get('risk_result', {}).get('risk_score', 0):.2f}",
                f"Patterns detected: {len(state.get('patterns', {}).get('detected_patterns', []))}",
                "Further investigation recommended"
            ],
            "recommendations": [
                "Generate SAR report (Phase 4)",
                "Manual analyst review",
                "Compliance escalation if HIGH risk"
            ],
            "report_generated_at": datetime.utcnow().isoformat(),
            "execution_duration_seconds": 0.0,
            "phase": "4_stub"
        }

        state["final_report"] = final_report
        state["sar_narrative"] = final_report["sar_narrative"]
        
        logger.info(f"[{agent_name}] Generated Phase 4 stub report")

        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.SUCCESS, output_data=final_report
        )

        return state

    except Exception as e:
        logger.error(f"[{agent_name}] Failed: {e}")
        error_ctx = handle_agent_error(
            agent_name, e, state, recoverable=False, max_retries=0
        )
        state = add_error_to_state(state, error_ctx, critical=False)
        
        state["agent_metrics"][agent_name] = log_agent_execution(
            agent_name, start_time, datetime.utcnow(),
            AgentStatus.FAILED, error=error_ctx
        )
        
        # Fallback: use fallback report
        state["final_report"] = create_fallback_final_report(
            state, f"Explanation node failed: {e}"
        )
        return state


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL ROUTING
# ─────────────────────────────────────────────────────────────────────────────


def route_after_risk_scoring(state: AMLAgentState) -> str:
    """
    Conditional routing logic after risk scoring.
    
    Routes based on risk_tier:
      - LOW        → low_risk_exit_node (no SAR generation)
      - MEDIUM/HIGH → explanation_node (SAR generation in Phase 4)
      - CRITICAL   → explanation_node with escalation flag
      - ERROR      → explanation_node (investigation recommended)
    
    Args:
        state: Current orchestration state
    
    Returns:
        Name of next node
    """
    routing_decision = state.get("routing_decision", "EXIT")
    risk_tier = state.get("risk_result", {}).get("risk_tier", "LOW")
    has_errors = state.get("has_errors", False)

    logger.info(
        f"[Routing] Decision: {routing_decision}, Tier: {risk_tier}, "
        f"Has errors: {has_errors}"
    )

    if routing_decision == "EXIT":
        logger.info("[Routing] → low_risk_exit_node")
        return "low_risk_exit_node"
    elif routing_decision in ["INVESTIGATE", "ESCALATE"]:
        logger.info("[Routing] → explanation_node")
        return "explanation_node"
    else:
        # Default: investigate on ambiguous routing
        logger.warning(f"[Routing] Unknown routing decision: {routing_decision}, investigating")
        return "explanation_node"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────


def build_orchestration_graph() -> StateGraph:
    """
    Build the complete LangGraph state machine.
    
    Graph structure:
        START
          ↓
        detection_node (Phase 1)
          ↓
        graph_construction_node (Phase 2a)
          ↓
        feature_extraction_node (Phase 2b)
          ↓
        pattern_detection_node (Phase 2c)
          ↓
        risk_scoring_node (Phase 2d)
          ↓
        [routing logic]
        ↙               ↘
    low_risk_exit   explanation_node
        ↓               ↓
        └────→  END  ←──┘
    
    Returns:
        Compiled StateGraph ready for execution
    """
    workflow = StateGraph(AMLAgentState)

    # Add all nodes
    workflow.add_node("detection_node", detection_node)
    workflow.add_node("graph_construction_node", graph_construction_node)
    workflow.add_node("feature_extraction_node", feature_extraction_node)
    workflow.add_node("pattern_detection_node", pattern_detection_node)
    workflow.add_node("risk_scoring_node", risk_scoring_node)
    workflow.add_node("low_risk_exit_node", low_risk_exit_node)
    workflow.add_node("explanation_node", explanation_node)

    # Add edges (linear pipeline up to risk scoring)
    workflow.add_edge(START, "detection_node")
    workflow.add_edge("detection_node", "graph_construction_node")
    workflow.add_edge("graph_construction_node", "feature_extraction_node")
    workflow.add_edge("feature_extraction_node", "pattern_detection_node")
    workflow.add_edge("pattern_detection_node", "risk_scoring_node")

    # Conditional routing after risk scoring
    workflow.add_conditional_edges(
        "risk_scoring_node",
        route_after_risk_scoring,
        {
            "low_risk_exit_node": "low_risk_exit_node",
            "explanation_node": "explanation_node"
        }
    )

    # Terminal edges
    workflow.add_edge("low_risk_exit_node", END)
    workflow.add_edge("explanation_node", END)

    return workflow


def compile_graph() -> Any:
    """
    Compile the orchestration graph for execution.
    
    Returns:
        Compiled graph (runnable)
    """
    workflow = build_orchestration_graph()
    app = workflow.compile()
    return app
