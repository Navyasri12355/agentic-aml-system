# File: tests/test_orchestration_state.py
"""
Tests for Phase 3 Orchestration State Management

Tests validate:
  - State initialization
  - State validation
  - Error context creation
  - Metrics tracking
"""

import pytest
from datetime import datetime

from src.orchestration.state import (
    AMLAgentState, RiskTier, RoutingDecision, AgentStatus,
    ErrorContext, AgentExecutionMetrics, create_initial_state, validate_state
)


class TestErrorContext:
    """Tests for ErrorContext data class."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        error = ErrorContext(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Test error",
            timestamp=datetime.utcnow(),
            retry_count=1,
            max_retries=3
        )
        
        assert error.agent_name == "test_agent"
        assert error.error_type == "ValueError"
        assert error.retry_count == 1
    
    def test_error_context_should_retry_true(self):
        """Test retry logic when retryable."""
        error = ErrorContext(
            agent_name="test_agent",
            error_type="ConnectionError",  # Retryable
            error_message="Connection failed",
            timestamp=datetime.utcnow(),
            retry_count=1,
            max_retries=3
        )
        
        assert error.should_retry() is True
    
    def test_error_context_should_retry_false_max_retries(self):
        """Test retry logic when max retries reached."""
        error = ErrorContext(
            agent_name="test_agent",
            error_type="ConnectionError",
            error_message="Connection failed",
            timestamp=datetime.utcnow(),
            retry_count=3,
            max_retries=3
        )
        
        assert error.should_retry() is False
    
    def test_error_context_should_retry_false_non_retryable(self):
        """Test retry logic for non-retryable errors."""
        error = ErrorContext(
            agent_name="test_agent",
            error_type="AssertionError",  # Non-retryable
            error_message="Assertion failed",
            timestamp=datetime.utcnow(),
            retry_count=0,
            max_retries=3
        )
        
        assert error.should_retry() is False
    
    def test_error_context_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.utcnow()
        error = ErrorContext(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Test error",
            timestamp=now,
            retry_count=0
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["agent_name"] == "test_agent"
        assert error_dict["error_type"] == "ValueError"
        assert error_dict["timestamp"] == now.isoformat()


class TestAgentExecutionMetrics:
    """Tests for AgentExecutionMetrics data class."""
    
    def test_metrics_creation(self):
        """Test creating metrics."""
        start = datetime.utcnow()
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            start_time=start,
            status=AgentStatus.IN_PROGRESS
        )
        
        assert metrics.agent_name == "test_agent"
        assert metrics.status == AgentStatus.IN_PROGRESS
    
    def test_metrics_compute_duration(self):
        """Test duration computation."""
        import time
        
        start = datetime.utcnow()
        time.sleep(0.1)
        end = datetime.utcnow()
        
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            start_time=start,
            end_time=end,
            status=AgentStatus.SUCCESS
        )
        
        duration = metrics.compute_duration()
        assert duration >= 0.1  # At least 0.1 seconds
    
    def test_metrics_to_dict(self):
        """Test serialization to dictionary."""
        start = datetime.utcnow()
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            start_time=start,
            status=AgentStatus.SUCCESS,
            duration_seconds=1.5,
            memory_used_mb=10.5
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["agent_name"] == "test_agent"
        assert metrics_dict["status"] == "SUCCESS"
        assert metrics_dict["duration_seconds"] == 1.5


class TestInitialState:
    """Tests for initial state creation."""
    
    def test_create_initial_state_defaults(self):
        """Test creating initial state with defaults."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        assert state["raw_transaction_path"] == "test.csv"
        assert state["account_id"] == "ACC_001"
        assert state["hop_radius"] == 2
        assert state["time_window_days"] == 30
        assert state["clean_df"] is None
        assert state["errors"] == []
        assert state["has_errors"] is False
    
    def test_create_initial_state_custom_params(self):
        """Test creating initial state with custom parameters."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001",
            hop_radius=3,
            time_window_days=60,
            max_neighbors=100,
            contamination=0.05
        )
        
        assert state["hop_radius"] == 3
        assert state["time_window_days"] == 60
        assert state["max_neighbors"] == 100
        assert state["contamination"] == 0.05
    
    def test_create_initial_state_execution_id_auto(self):
        """Test automatic execution ID generation."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        assert state["execution_id"] is not None
        assert state["execution_id"].startswith("exec_")
    
    def test_create_initial_state_execution_id_custom(self):
        """Test custom execution ID."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001",
            execution_id="custom_id_123"
        )
        
        assert state["execution_id"] == "custom_id_123"
    
    def test_create_initial_state_timestamps(self):
        """Test timestamp initialization."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        assert state["overall_start_time"] is not None
        assert isinstance(state["overall_start_time"], datetime)


class TestStateValidation:
    """Tests for state validation logic."""
    
    def test_validate_state_valid(self):
        """Test validation of valid state."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        is_valid, errors = validate_state(state)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_state_missing_transaction_path(self):
        """Test validation with missing transaction path."""
        state = create_initial_state(
            raw_transaction_path="",
            account_id="ACC_001"
        )
        
        is_valid, errors = validate_state(state)
        
        assert is_valid is False
        assert any("raw_transaction_path" in e for e in errors)
    
    def test_validate_state_missing_account_id(self):
        """Test validation with missing account ID."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id=""
        )
        
        is_valid, errors = validate_state(state)
        
        assert is_valid is False
        assert any("account_id" in e for e in errors)
    
    def test_validate_state_invalid_hop_radius(self):
        """Test validation with invalid hop radius."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001",
            hop_radius=0
        )
        
        is_valid, errors = validate_state(state)
        
        assert is_valid is False
        assert any("hop_radius" in e for e in errors)
    
    def test_validate_state_invalid_contamination(self):
        """Test validation with invalid contamination."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001",
            contamination=1.5  # Invalid: > 1.0
        )
        
        is_valid, errors = validate_state(state)
        
        assert is_valid is False
        assert any("contamination" in e for e in errors)


class TestStateTransitions:
    """Tests for state transitions through pipeline."""
    
    def test_state_tracks_error_additions(self):
        """Test adding errors to state."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        error = ErrorContext(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Test error",
            timestamp=datetime.utcnow()
        )
        
        state["errors"] = [error]
        state["has_errors"] = True
        
        assert len(state["errors"]) == 1
        assert state["has_errors"] is True
    
    def test_state_tracks_agent_metrics(self):
        """Test tracking agent metrics."""
        state = create_initial_state(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            start_time=datetime.utcnow(),
            status=AgentStatus.SUCCESS
        )
        
        state["agent_metrics"]["test_agent"] = metrics
        
        assert "test_agent" in state["agent_metrics"]
        assert state["agent_metrics"]["test_agent"].status == AgentStatus.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
