# File: src/orchestration/errors.py
"""
Phase 3 Error Handling & Recovery Utilities

Provides comprehensive error handling, recovery strategies, and validation
for the AML investigation orchestration pipeline.
"""

import logging
import traceback
from typing import Optional, Callable, Any, TypeVar, Tuple, List
from datetime import datetime
from functools import wraps
import time

from src.orchestration.state import (
    AMLAgentState, ErrorContext, AgentStatus, AgentExecutionMetrics
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OrchestrationError(Exception):
    """Base exception for orchestration pipeline errors."""
    pass


class AgentExecutionError(OrchestrationError):
    """Error during agent node execution."""
    pass


class ValidationError(OrchestrationError):
    """Data validation error."""
    pass


class RecoveryError(OrchestrationError):
    """Error during recovery attempt."""
    pass


def handle_agent_error(
    agent_name: str,
    error: Exception,
    state: AMLAgentState,
    recoverable: bool = True,
    max_retries: int = 3
) -> ErrorContext:
    """
    Structured error handling for agent execution.

    Args:
        agent_name: Name of the agent that failed
        error: Exception that occurred
        state: Current orchestration state
        recoverable: Whether error is recoverable
        max_retries: Maximum retry attempts

    Returns:
        ErrorContext with detailed error information
    """
    error_context = ErrorContext(
        agent_name=agent_name,
        error_type=type(error).__name__,
        error_message=str(error),
        timestamp=datetime.utcnow(),
        retry_count=0,
        max_retries=max_retries,
        traceback=traceback.format_exc()
    )

    logger.error(
        f"Agent '{agent_name}' failed: {error_context.error_type} - "
        f"{error_context.error_message}",
        extra={"error_context": error_context.to_dict()}
    )

    return error_context


def retry_on_error(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    recoverable_errors: Optional[List[type]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (default: 2.0)
        backoff_max: Maximum backoff time in seconds (default: 60.0)
        recoverable_errors: List of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    if recoverable_errors is None:
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            IOError,
            OSError,
            ValueError
        ]

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            retry_count = 0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    is_recoverable = any(
                        isinstance(e, error_type)
                        for error_type in recoverable_errors
                    )

                    if not is_recoverable or retry_count >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{retry_count} retries: {e}"
                        )
                        raise

                    # Calculate backoff with exponential increase
                    backoff_seconds = min(
                        backoff_base ** retry_count,
                        backoff_max
                    )

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {retry_count + 1}/"
                        f"{max_retries + 1}). Retrying in {backoff_seconds:.1f}s: {e}"
                    )

                    time.sleep(backoff_seconds)
                    retry_count += 1

            raise last_exception

        return wrapper

    return decorator


def safe_get(
    data: dict,
    key: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Safe dictionary access with validation.

    Args:
        data: Dictionary to access
        key: Key to retrieve
        default: Default value if key missing
        required: If True, raise error if key missing

    Returns:
        Value from dictionary or default

    Raises:
        ValidationError: If required key is missing
    """
    if key not in data:
        if required:
            raise ValidationError(f"Required key missing: {key}")
        return default
    return data[key]


def validate_dataframe_schema(
    df,
    required_columns: List[str],
    context: str = "dataframe"
) -> Tuple[bool, List[str]]:
    """
    Validate pandas DataFrame schema.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        context: Context for error messages

    Returns:
        Tuple of (is_valid: bool, missing_columns: List[str])
    """
    if df is None:
        return False, required_columns

    if not hasattr(df, 'columns'):
        return False, required_columns

    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        logger.error(
            f"Missing columns in {context}: {missing}. "
            f"Found: {list(df.columns)}"
        )

    return len(missing) == 0, missing


def validate_features_dict(
    features: dict,
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate feature dictionary structure.

    Args:
        features: Dictionary to validate
        required_keys: Optional list of required keys

    Returns:
        Tuple of (is_valid: bool, missing_keys: List[str])
    """
    if features is None:
        return False, required_keys or []

    if not isinstance(features, dict):
        logger.error(f"Features must be dict, got {type(features)}")
        return False, []

    if required_keys:
        missing = [k for k in required_keys if k not in features]
        if missing:
            logger.error(f"Missing feature keys: {missing}")
            return False, missing

    return True, []


def create_fallback_risk_result(
    account_id: str,
    reason: str = "Error during risk scoring"
) -> dict:
    """
    Create fallback risk result when primary scoring fails.

    Args:
        account_id: Account under investigation
        reason: Reason for fallback

    Returns:
        Simplified risk result dict
    """
    return {
        "account_id": account_id,
        "risk_score": 0.5,  # Neutral middle value
        "risk_tier": "MEDIUM",
        "score_components": {
            "anomaly_score": 0.0,
            "pattern_score": 0.0,
            "velocity_score": 0.0,
            "cross_border_score": 0.0,
            "structuring_score": 0.0,
            "temporal_score": 0.0
        },
        "routing_decision": "INVESTIGATE",
        "risk_justification": reason,
        "decision_timestamp": datetime.utcnow().isoformat(),
        "is_fallback": True,
        "fallback_reason": reason
    }


def create_fallback_final_report(
    state: AMLAgentState,
    error_summary: str
) -> dict:
    """
    Create fallback final report when pipeline fails.

    Args:
        state: Current orchestration state
        error_summary: Summary of errors encountered

    Returns:
        Fallback final report dict
    """
    return {
        "account_id": state.get("account_id", "UNKNOWN"),
        "report_id": state.get("report_id", "UNKNOWN"),
        "risk_score": 0.5,
        "risk_tier": "MEDIUM",
        "detected_patterns": [],
        "sar_narrative": None,
        "graph_summary": {
            "status": "FAILED",
            "error": error_summary
        },
        "key_findings": [
            "Investigation encountered errors during processing",
            f"Errors: {error_summary}"
        ],
        "recommendations": [
            "Manual review required",
            "Investigate system logs for detailed errors"
        ],
        "report_generated_at": datetime.utcnow().isoformat(),
        "execution_duration_seconds": state.get("total_duration_seconds", 0),
        "is_fallback": True,
        "pipeline_status": "PARTIAL_FAILURE"
    }


def add_error_to_state(
    state: AMLAgentState,
    error_context: ErrorContext,
    critical: bool = False
) -> AMLAgentState:
    """
    Add error context to state error tracking.

    Args:
        state: Current orchestration state
        error_context: Error context to add
        critical: Whether this is a critical error

    Returns:
        Updated state with error added
    """
    errors = state.get("errors", [])
    errors.append(error_context)

    state["errors"] = errors
    state["has_errors"] = True

    if critical:
        state["has_critical_errors"] = True
        logger.critical(
            f"CRITICAL ERROR in {error_context.agent_name}: "
            f"{error_context.error_message}"
        )

    return state


def log_agent_execution(
    agent_name: str,
    start_time: datetime,
    end_time: datetime,
    status: AgentStatus,
    input_data: Any = None,
    output_data: Any = None,
    error: Optional[ErrorContext] = None
) -> AgentExecutionMetrics:
    """
    Create and log agent execution metrics.

    Args:
        agent_name: Name of the agent
        start_time: Execution start time
        end_time: Execution end time
        status: Execution status
        input_data: Input data (for size estimation)
        output_data: Output data (for size estimation)
        error: Error context if execution failed

    Returns:
        AgentExecutionMetrics object
    """
    import sys

    def estimate_size_mb(obj: Any) -> float:
        """Estimate object size in MB."""
        try:
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__() / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0

    metrics = AgentExecutionMetrics(
        agent_name=agent_name,
        start_time=start_time,
        end_time=end_time,
        status=status,
        input_size_mb=estimate_size_mb(input_data),
        output_size_mb=estimate_size_mb(output_data),
        error=error
    )

    metrics.compute_duration()

    logger.info(
        f"Agent '{agent_name}' execution: {status.value} "
        f"({metrics.duration_seconds:.2f}s) - "
        f"Input: {metrics.input_size_mb:.2f}MB, Output: {metrics.output_size_mb:.2f}MB"
    )

    return metrics


def validate_risk_result(risk_result: dict) -> Tuple[bool, List[str]]:
    """
    Validate risk scoring result structure.

    Args:
        risk_result: Risk scoring output to validate

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    if not isinstance(risk_result, dict):
        errors.append(f"risk_result must be dict, got {type(risk_result)}")
        return False, errors

    required_fields = [
        "account_id",
        "risk_score",
        "risk_tier",
        "routing_decision",
        "decision_timestamp"
    ]

    for field in required_fields:
        if field not in risk_result:
            errors.append(f"Missing required field: {field}")

    # Validate risk_score range
    if "risk_score" in risk_result:
        score = risk_result["risk_score"]
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            errors.append(f"risk_score must be float in [0.0, 1.0], got {score}")

    # Validate risk_tier
    valid_tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if "risk_tier" in risk_result:
        tier = risk_result["risk_tier"]
        if tier not in valid_tiers:
            errors.append(f"risk_tier must be one of {valid_tiers}, got {tier}")

    # Validate routing_decision
    valid_routings = ["EXIT", "INVESTIGATE", "ESCALATE", "QUARANTINE"]
    if "routing_decision" in risk_result:
        routing = risk_result["routing_decision"]
        if routing not in valid_routings:
            errors.append(
                f"routing_decision must be one of {valid_routings}, got {routing}"
            )

    return len(errors) == 0, errors
