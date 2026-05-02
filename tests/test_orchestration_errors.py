# File: tests/test_orchestration_errors.py
"""
Tests for Phase 3 Error Handling & Recovery

Tests validate:
  - Error handling and logging
  - Retry logic with exponential backoff
  - Data validation functions
  - Fallback mechanisms
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.orchestration.state import AMLAgentState, create_initial_state, AgentStatus, ErrorContext
from src.orchestration.errors import (
    OrchestrationError, AgentExecutionError, ValidationError, RecoveryError,
    handle_agent_error, retry_on_error, safe_get,
    validate_dataframe_schema, validate_features_dict, validate_risk_result,
    create_fallback_risk_result, create_fallback_final_report,
    add_error_to_state, log_agent_execution
)


class TestErrorClasses:
    """Tests for custom exception classes."""
    
    def test_orchestration_error_inheritance(self):
        """Test OrchestrationError is Exception."""
        error = OrchestrationError("Test error")
        assert isinstance(error, Exception)
    
    def test_agent_execution_error_inheritance(self):
        """Test AgentExecutionError is OrchestrationError."""
        error = AgentExecutionError("Agent failed")
        assert isinstance(error, OrchestrationError)
    
    def test_validation_error_inheritance(self):
        """Test ValidationError is OrchestrationError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, OrchestrationError)


class TestErrorHandling:
    """Tests for error handling functions."""
    
    def test_handle_agent_error_creates_context(self):
        """Test handle_agent_error creates proper context."""
        state = create_initial_state("test.csv", "ACC_001")
        error = ValueError("Test error")
        
        error_ctx = handle_agent_error("test_agent", error, state)
        
        assert error_ctx.agent_name == "test_agent"
        assert error_ctx.error_type == "ValueError"
        assert error_ctx.error_message == "Test error"
        assert error_ctx.traceback is not None
    
    def test_handle_agent_error_sets_retry_count(self):
        """Test retry count is set properly."""
        state = create_initial_state("test.csv", "ACC_001")
        error = ValueError("Test error")
        
        error_ctx = handle_agent_error("test_agent", error, state, max_retries=5)
        
        assert error_ctx.max_retries == 5
        assert error_ctx.retry_count == 0
    
    def test_add_error_to_state(self):
        """Test adding error to state."""
        state = create_initial_state("test.csv", "ACC_001")
        error_ctx = ErrorContext(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Test error",
            timestamp=datetime.utcnow()
        )
        
        updated_state = add_error_to_state(state, error_ctx, critical=False)
        
        assert len(updated_state["errors"]) == 1
        assert updated_state["has_errors"] is True
        assert updated_state["has_critical_errors"] is False
    
    def test_add_error_to_state_critical(self):
        """Test adding critical error to state."""
        state = create_initial_state("test.csv", "ACC_001")
        error_ctx = ErrorContext(
            agent_name="test_agent",
            error_type="RuntimeError",
            error_message="Critical error",
            timestamp=datetime.utcnow()
        )
        
        updated_state = add_error_to_state(state, error_ctx, critical=True)
        
        assert updated_state["has_critical_errors"] is True


class TestRetryLogic:
    """Tests for retry decorator."""
    
    def test_retry_succeeds_on_first_attempt(self):
        """Test function succeeds without retry."""
        call_count = 0
        
        @retry_on_error(max_retries=3)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_succeeds_after_retries(self):
        """Test function succeeds after retries."""
        call_count = 0
        
        @retry_on_error(max_retries=3, backoff_base=0.01, backoff_max=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausts_attempts(self):
        """Test function fails after max retries."""
        @retry_on_error(max_retries=2, backoff_base=0.01, backoff_max=0.1)
        def test_function():
            raise ConnectionError("Connection failed")
        
        with pytest.raises(ConnectionError):
            test_function()
    
    def test_retry_skips_non_retryable_errors(self):
        """Test non-retryable errors are not retried."""
        call_count = 0
        
        @retry_on_error(max_retries=3, backoff_base=0.01, backoff_max=0.1)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise AssertionError("Not retryable")
        
        with pytest.raises(AssertionError):
            test_function()
        
        assert call_count == 1  # Only one attempt
    
    def test_retry_custom_retryable_errors(self):
        """Test custom retryable error list."""
        call_count = 0
        
        @retry_on_error(
            max_retries=2,
            backoff_base=0.01,
            backoff_max=0.1,
            recoverable_errors=[RuntimeError]
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Custom error")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 2


class TestDataValidation:
    """Tests for data validation functions."""
    
    def test_safe_get_existing_key(self):
        """Test safe_get with existing key."""
        data = {"key": "value"}
        result = safe_get(data, "key")
        assert result == "value"
    
    def test_safe_get_missing_key_with_default(self):
        """Test safe_get with missing key and default."""
        data = {"key": "value"}
        result = safe_get(data, "missing", default="default")
        assert result == "default"
    
    def test_safe_get_missing_key_required(self):
        """Test safe_get raises for missing required key."""
        data = {"key": "value"}
        with pytest.raises(ValidationError):
            safe_get(data, "missing", required=True)
    
    def test_validate_dataframe_schema_valid(self):
        """Test dataframe schema validation with valid schema."""
        try:
            import pandas as pd
            df = pd.DataFrame({
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"]
            })
            
            is_valid, missing = validate_dataframe_schema(
                df, ["col1", "col2"]
            )
            
            assert is_valid is True
            assert len(missing) == 0
        except ImportError:
            pytest.skip("pandas not installed")
    
    def test_validate_dataframe_schema_missing_columns(self):
        """Test dataframe schema validation with missing columns."""
        try:
            import pandas as pd
            df = pd.DataFrame({
                "col1": [1, 2, 3]
            })
            
            is_valid, missing = validate_dataframe_schema(
                df, ["col1", "col2", "col3"]
            )
            
            assert is_valid is False
            assert "col2" in missing
            assert "col3" in missing
        except ImportError:
            pytest.skip("pandas not installed")
    
    def test_validate_dataframe_schema_none_df(self):
        """Test dataframe schema validation with None."""
        is_valid, missing = validate_dataframe_schema(
            None, ["col1"]
        )
        
        assert is_valid is False
    
    def test_validate_features_dict_valid(self):
        """Test feature dict validation with valid dict."""
        features = {
            "required_key": "value",
            "another_key": 123
        }
        
        is_valid, missing = validate_features_dict(
            features, ["required_key"]
        )
        
        assert is_valid is True
        assert len(missing) == 0
    
    def test_validate_features_dict_missing_keys(self):
        """Test feature dict validation with missing keys."""
        features = {"key1": "value"}
        
        is_valid, missing = validate_features_dict(
            features, ["key1", "key2", "key3"]
        )
        
        assert is_valid is False
        assert "key2" in missing
        assert "key3" in missing
    
    def test_validate_risk_result_valid(self):
        """Test risk result validation with valid result."""
        risk_result = {
            "account_id": "ACC_001",
            "risk_score": 0.75,
            "risk_tier": "HIGH",
            "routing_decision": "INVESTIGATE",
            "decision_timestamp": datetime.utcnow().isoformat()
        }
        
        is_valid, errors = validate_risk_result(risk_result)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_risk_result_invalid_score(self):
        """Test risk result with invalid score."""
        risk_result = {
            "account_id": "ACC_001",
            "risk_score": 1.5,  # Invalid: > 1.0
            "risk_tier": "HIGH",
            "routing_decision": "INVESTIGATE",
            "decision_timestamp": datetime.utcnow().isoformat()
        }
        
        is_valid, errors = validate_risk_result(risk_result)
        
        assert is_valid is False
        assert any("risk_score" in e for e in errors)
    
    def test_validate_risk_result_invalid_tier(self):
        """Test risk result with invalid tier."""
        risk_result = {
            "account_id": "ACC_001",
            "risk_score": 0.75,
            "risk_tier": "INVALID_TIER",
            "routing_decision": "INVESTIGATE",
            "decision_timestamp": datetime.utcnow().isoformat()
        }
        
        is_valid, errors = validate_risk_result(risk_result)
        
        assert is_valid is False
        assert any("risk_tier" in e for e in errors)


class TestFallbackMechanisms:
    """Tests for fallback functions."""
    
    def test_create_fallback_risk_result(self):
        """Test creating fallback risk result."""
        fallback = create_fallback_risk_result("ACC_001", "Error occurred")
        
        assert fallback["account_id"] == "ACC_001"
        assert fallback["risk_score"] == 0.5
        assert fallback["risk_tier"] == "MEDIUM"
        assert fallback["is_fallback"] is True
    
    def test_create_fallback_final_report(self):
        """Test creating fallback final report."""
        state = create_initial_state("test.csv", "ACC_001")
        
        fallback = create_fallback_final_report(state, "Pipeline failed")
        
        assert fallback["account_id"] == "ACC_001"
        assert fallback["is_fallback"] is True
        assert "Pipeline failed" in str(fallback.get("graph_summary", {}).get("error", ""))


class TestMetricsLogging:
    """Tests for metrics logging."""
    
    def test_log_agent_execution_success(self):
        """Test logging successful agent execution."""
        start = datetime.utcnow()
        time.sleep(0.05)
        end = datetime.utcnow()
        
        metrics = log_agent_execution(
            "test_agent",
            start,
            end,
            AgentStatus.SUCCESS
        )
        
        assert metrics.agent_name == "test_agent"
        assert metrics.status == AgentStatus.SUCCESS
        assert metrics.duration_seconds >= 0.05
    
    def test_log_agent_execution_with_error(self):
        """Test logging failed agent execution with error context."""
        start = datetime.utcnow()
        end = datetime.utcnow()
        error_ctx = ErrorContext(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Test error",
            timestamp=datetime.utcnow()
        )
        
        metrics = log_agent_execution(
            "test_agent",
            start,
            end,
            AgentStatus.FAILED,
            error=error_ctx
        )
        
        assert metrics.status == AgentStatus.FAILED
        assert metrics.error == error_ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
