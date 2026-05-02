# File: src/orchestration/state.py
"""
Comprehensive Phase 3 LangGraph State Management

This module defines the AMLAgentState TypedDict that represents the complete
execution state of the AML investigation pipeline. It tracks:
  - Input data and parameters
  - Intermediate outputs from each phase
  - Error states and recovery metadata
  - Final investigation results
  - Execution metrics and audit trail
"""

from typing import TypedDict, Optional, Any, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from enum import Enum


class RiskTier(str, Enum):
    """Risk classification tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RoutingDecision(str, Enum):
    """Routing decisions after risk scoring."""
    EXIT = "EXIT"
    INVESTIGATE = "INVESTIGATE"
    ESCALATE = "ESCALATE"
    QUARANTINE = "QUARANTINE"


class AgentStatus(str, Enum):
    """Status of individual agents during execution."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRY = "RETRY"


@dataclass
class ErrorContext:
    """Detailed error information for debugging and recovery."""
    agent_name: str
    error_type: str
    error_message: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    traceback: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_message: Optional[str] = None

    def should_retry(self) -> bool:
        """Check if error is retryable and retry count not exceeded."""
        retryable_errors = [
            "ConnectionError",
            "TimeoutError",
            "IOError",
            "ValueError",
            "KeyError"
        ]
        return (self.error_type in retryable_errors and 
                self.retry_count < self.max_retries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "traceback": self.traceback,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_message": self.recovery_message
        }


@dataclass
class AgentExecutionMetrics:
    """Metrics for each agent execution."""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: AgentStatus = AgentStatus.NOT_STARTED
    duration_seconds: float = 0.0
    memory_used_mb: float = 0.0
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    rows_processed: int = 0
    rows_skipped: int = 0
    error: Optional[ErrorContext] = None

    def compute_duration(self) -> float:
        """Compute and cache duration."""
        if self.end_time:
            delta = self.end_time - self.start_time
            self.duration_seconds = delta.total_seconds()
        return self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        self.compute_duration()
        return {
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "memory_used_mb": self.memory_used_mb,
            "input_size_mb": self.input_size_mb,
            "output_size_mb": self.output_size_mb,
            "rows_processed": self.rows_processed,
            "rows_skipped": self.rows_skipped,
            "error": self.error.to_dict() if self.error else None
        }


class AMLAgentState(TypedDict, total=False):
    """
    Comprehensive Phase 3 Orchestration State.
    
    This TypedDict maintains all data flowing through the LangGraph pipeline:
    - Input parameters
    - Intermediate results from each phase
    - Error states and recovery information
    - Final investigation outputs
    - Execution metrics and audit trail
    """

    # ─────── INPUT SECTION ────────────────────────────────────────────────
    # Entry parameters provided by user/API

    raw_transaction_path: str
    """Path to raw transaction CSV file"""

    account_id: str
    """Account ID under investigation"""

    hop_radius: int
    """Graph expansion radius (default: 2 hops)"""

    time_window_days: int
    """Lookback window in days (default: 30 days)"""

    max_neighbors: int
    """Maximum number of neighbors to include in graph (default: 50)"""

    contamination: float
    """Expected contamination rate for Isolation Forest (default: 0.02)"""

    # ─────── PHASE 1: DETECTION AGENT ────────────────────────────────────

    clean_df: Optional[pd.DataFrame]
    """Cleaned transactions from Phase 1"""

    flagged_df: Optional[pd.DataFrame]
    """Flagged suspicious transactions with anomaly scores"""

    flagged_row: Optional[Dict[str, Any]]
    """Single flagged transaction row under investigation"""

    detection_features_used: Optional[List[str]]
    """Feature columns used in detection model"""

    detection_model_version: Optional[str]
    """Version of detection model used"""

    # ─────── PHASE 2a: GRAPH CONSTRUCTION ────────────────────────────────

    subgraph: Optional[Dict[str, Any]]
    """
    Constructed transaction subgraph:
    {
        "nodes": list of account nodes with metadata,
        "edges": list of transaction edges with amounts/dates,
        "node_count": number of accounts in subgraph,
        "edge_count": number of transactions in subgraph,
        "expansion_hops": actual number of hops used
    }
    """

    graph_metadata: Optional[Dict[str, Any]]
    """
    Metadata about graph construction:
    {
        "source_account": account_id,
        "construction_timestamp": ISO datetime,
        "expansion_time_window": actual time range used,
        "neighbors_found": count of neighbors at each hop
    }
    """

    # ─────── PHASE 2b: FEATURE EXTRACTION ────────────────────────────────

    features: Optional[Dict[str, Any]]
    """
    Extracted features per flagged account:
    {
        "account_id": str,
        "subgraph_node_count": int,
        "subgraph_edge_count": int,
        "features": {
            "in_degree": int,
            "out_degree": int,
            "in_out_ratio": float,
            "betweenness": float,
            "total_received": float,
            "total_sent": float,
            "net_flow": float,
            "txn_velocity": float,
            "burst_score": float,
            "avg_amount": float,
            "amount_std": float,
            "has_cycle": bool,
            "max_path_length": int,
            "num_intermediaries": int,
            "hop_count": int
        }
    }
    """

    feature_statistics: Optional[Dict[str, Any]]
    """
    Global statistics for feature normalization:
    {
        "velocity_p95": float,
        "velocity_p99": float,
        "p99_amount": float,
        "avg_degree": float,
        "max_degree": int
    }
    """

    # ─────── PHASE 2c: PATTERN DETECTION ─────────────────────────────────

    patterns: Optional[Dict[str, Any]]
    """
    Detected laundering patterns:
    {
        "account_id": str,
        "detected_patterns": list of pattern names,
        "pattern_confidence": {pattern_name: confidence_score},
        "is_isolated": bool,
        "pattern_descriptions": {pattern_name: explanation}
    }
    """

    # ─────── PHASE 2d: RISK SCORING ──────────────────────────────────────

    risk_result: Optional[Dict[str, Any]]
    """
    Final risk scoring output:
    {
        "account_id": str,
        "risk_score": float [0.0-1.0],
        "risk_tier": RiskTier enum,
        "score_components": {
            "anomaly_score": float,
            "pattern_score": float,
            "velocity_score": float,
            "cross_border_score": float,
            "structuring_score": float,
            "temporal_score": float
        },
        "routing_decision": RoutingDecision enum,
        "risk_justification": str,
        "decision_timestamp": ISO datetime
    }
    """

    # ─────── ROUTING & DECISION ──────────────────────────────────────────

    routing_decision: str
    """
    Routing decision after risk scoring:
    "EXIT" (LOW risk), "INVESTIGATE" (MEDIUM/HIGH), "ESCALATE" (CRITICAL)
    """

    should_generate_sar: bool
    """Whether SAR report should be generated"""

    # ─────── PHASE 4: EXPLANATION & SAR (STUB) ──────────────────────────

    sar_narrative: Optional[str]
    """LLM-generated SAR narrative (Phase 4)"""

    final_report: Optional[Dict[str, Any]]
    """
    Complete investigation report:
    {
        "account_id": str,
        "report_id": unique identifier,
        "risk_score": float,
        "risk_tier": str,
        "detected_patterns": list,
        "sar_narrative": str or None,
        "graph_summary": dict,
        "key_findings": list,
        "recommendations": list,
        "report_generated_at": ISO datetime,
        "execution_duration_seconds": float
    }
    """

    report_id: Optional[str]
    """Unique identifier for generated report"""

    # ─────── ERROR TRACKING & RECOVERY ───────────────────────────────────

    errors: List[ErrorContext]
    """Accumulated error contexts from all agents"""

    has_errors: bool
    """Flag: true if any agent encountered errors"""

    has_critical_errors: bool
    """Flag: true if critical errors prevent continuation"""

    error_recovery_attempts: int
    """Count of attempted error recoveries"""

    error_recovery_successful: bool
    """Flag: true if error recovery succeeded"""

    error_recovery_log: List[Dict[str, Any]]
    """Log of all recovery attempts"""

    fallback_risk_tier: Optional[str]
    """Fallback risk tier if normal scoring fails"""

    # ─────── EXECUTION METRICS & AUDIT TRAIL ─────────────────────────────

    agent_metrics: Dict[str, AgentExecutionMetrics]
    """Per-agent execution metrics"""

    overall_start_time: Optional[datetime]
    """Pipeline start timestamp"""

    overall_end_time: Optional[datetime]
    """Pipeline end timestamp"""

    total_duration_seconds: float
    """Total pipeline execution time"""

    pipeline_version: str
    """Version of orchestration pipeline"""

    execution_id: str
    """Unique execution identifier for audit trail"""

    audit_log: List[Dict[str, Any]]
    """Detailed execution audit log"""

    # ─────── FEATURE FLAGS & CONFIGURATION ────────────────────────────────

    skip_detection: bool
    """Skip Phase 1 if flagged transactions already provided"""

    skip_graph_expansion: bool
    """Skip graph context expansion (use only direct transactions)"""

    use_fallback_risk_scoring: bool
    """Use simplified risk scoring if primary scoring fails"""

    enable_debug_logging: bool
    """Enable detailed debug logging"""

    enable_recovery: bool
    """Enable automatic error recovery"""

    # ─────── METADATA & CONTEXT ──────────────────────────────────────────

    investigation_type: str
    """Type of investigation: "AUTO", "MANUAL", "BATCH", "PRIORITY" """

    priority_level: int
    """Priority level (1=highest, 10=lowest)"""

    investigation_context: Optional[Dict[str, Any]]
    """Additional context about investigation"""

    external_flags: Optional[List[str]]
    """External flags/tags from upstream systems"""


def create_initial_state(
    raw_transaction_path: str,
    account_id: str,
    hop_radius: int = 2,
    time_window_days: int = 30,
    max_neighbors: int = 50,
    contamination: float = 0.02,
    execution_id: Optional[str] = None
) -> AMLAgentState:
    """
    Factory function to create initialized AMLAgentState.

    Args:
        raw_transaction_path: Path to transaction CSV
        account_id: Target account for investigation
        hop_radius: Graph expansion depth (default: 2)
        time_window_days: Temporal window (default: 30)
        max_neighbors: Max neighbors per expansion (default: 50)
        contamination: Isolation Forest contamination (default: 0.02)
        execution_id: Optional execution ID (auto-generated if None)

    Returns:
        AMLAgentState: Initialized state ready for pipeline execution
    """
    from uuid import uuid4

    if execution_id is None:
        execution_id = f"exec_{uuid4().hex[:12]}"

    now = datetime.utcnow()

    return AMLAgentState(
        # Input
        raw_transaction_path=raw_transaction_path,
        account_id=account_id,
        hop_radius=hop_radius,
        time_window_days=time_window_days,
        max_neighbors=max_neighbors,
        contamination=contamination,
        # Phase 1
        clean_df=None,
        flagged_df=None,
        flagged_row=None,
        detection_features_used=None,
        detection_model_version=None,
        # Phase 2a
        subgraph=None,
        graph_metadata=None,
        # Phase 2b
        features=None,
        feature_statistics=None,
        # Phase 2c
        patterns=None,
        # Phase 2d
        risk_result=None,
        # Routing
        routing_decision="EXIT",
        should_generate_sar=False,
        # Phase 4 (stub)
        sar_narrative=None,
        final_report=None,
        report_id=None,
        # Error tracking
        errors=[],
        has_errors=False,
        has_critical_errors=False,
        error_recovery_attempts=0,
        error_recovery_successful=False,
        error_recovery_log=[],
        fallback_risk_tier=None,
        # Metrics
        agent_metrics={},
        overall_start_time=now,
        overall_end_time=None,
        total_duration_seconds=0.0,
        pipeline_version="3.0",
        execution_id=execution_id,
        audit_log=[],
        # Features
        skip_detection=False,
        skip_graph_expansion=False,
        use_fallback_risk_scoring=False,
        enable_debug_logging=False,
        enable_recovery=True,
        # Metadata
        investigation_type="AUTO",
        priority_level=5,
        investigation_context=None,
        external_flags=None
    )


def validate_state(state: AMLAgentState) -> tuple[bool, List[str]]:
    """
    Validate state integrity before each node execution.

    Args:
        state: AMLAgentState to validate

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    # Check required fields
    if not state.get("raw_transaction_path"):
        errors.append("Missing required field: raw_transaction_path")
    if not state.get("account_id"):
        errors.append("Missing required field: account_id")
    if state.get("hop_radius", 0) <= 0:
        errors.append("Invalid hop_radius: must be > 0")
    if state.get("time_window_days", 0) <= 0:
        errors.append("Invalid time_window_days: must be > 0")
    if not (0 < state.get("contamination", 0.02) < 1.0):
        errors.append("Invalid contamination: must be between 0 and 1")

    return len(errors) == 0, errors
