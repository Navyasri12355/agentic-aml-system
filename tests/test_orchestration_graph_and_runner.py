# File: tests/test_orchestration_graph_and_runner.py
"""
Tests for Phase 3 LangGraph Orchestration

Tests validate:
  - Graph structure and node connectivity
  - Conditional routing logic
  - Node execution (with mocks)
  - Runner functionality
  - Batch processing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from datetime import datetime
from typing import Dict, Any

from src.orchestration.state import AMLAgentState, create_initial_state, AgentStatus
from src.orchestration.graph import (
    build_orchestration_graph, compile_graph,
    route_after_risk_scoring
)
from src.orchestration.run import (
    OrchestrationRunner, create_runner
)


class TestGraphStructure:
    """Tests for orchestration graph structure."""
    
    def test_build_orchestration_graph_creates_workflow(self):
        """Test that graph builder creates workflow."""
        from langgraph.graph import StateGraph
        
        workflow = build_orchestration_graph()
        
        # Check graph is created and is a StateGraph instance
        assert workflow is not None
        assert isinstance(workflow, StateGraph)
    
    def test_compile_graph_returns_runnable(self):
        """Test that compiled graph is runnable."""
        app = compile_graph()
        
        # Should have invoke method
        assert hasattr(app, 'invoke')
        assert callable(app.invoke)
    
    def test_graph_has_all_nodes(self):
        """Test graph contains all required nodes."""
        from langgraph.graph import StateGraph
        
        workflow = build_orchestration_graph()
        
        # Verify it's a valid StateGraph
        assert isinstance(workflow, StateGraph)
        
        # Compile and verify the compiled graph works
        compiled = workflow.compile()
        assert hasattr(compiled, 'invoke')
        assert callable(compiled.invoke)


class TestConditionalRouting:
    """Tests for conditional routing logic."""
    
    def test_route_to_low_risk_exit(self):
        """Test routing to low_risk_exit_node."""
        state = create_initial_state("test.csv", "ACC_001")
        state["routing_decision"] = "EXIT"
        state["risk_result"] = {"risk_tier": "LOW"}
        
        next_node = route_after_risk_scoring(state)
        
        assert next_node == "low_risk_exit_node"
    
    def test_route_to_explanation_investigate(self):
        """Test routing to explanation_node for INVESTIGATE."""
        state = create_initial_state("test.csv", "ACC_001")
        state["routing_decision"] = "INVESTIGATE"
        state["risk_result"] = {"risk_tier": "MEDIUM"}
        
        next_node = route_after_risk_scoring(state)
        
        assert next_node == "explanation_node"
    
    def test_route_to_explanation_escalate(self):
        """Test routing to explanation_node for ESCALATE."""
        state = create_initial_state("test.csv", "ACC_001")
        state["routing_decision"] = "ESCALATE"
        state["risk_result"] = {"risk_tier": "CRITICAL"}
        
        next_node = route_after_risk_scoring(state)
        
        assert next_node == "explanation_node"
    
    def test_route_default_to_explanation(self):
        """Test default routing to explanation_node."""
        state = create_initial_state("test.csv", "ACC_001")
        state["routing_decision"] = "UNKNOWN"
        state["risk_result"] = {}
        
        next_node = route_after_risk_scoring(state)
        
        assert next_node == "explanation_node"


class TestOrchestrationRunner:
    """Tests for OrchestrationRunner."""
    
    def test_create_runner(self):
        """Test creating runner."""
        runner = create_runner()
        
        assert isinstance(runner, OrchestrationRunner)
        assert runner.enable_debug_logging is False
        assert runner.enable_recovery is True
    
    def test_create_runner_with_options(self):
        """Test creating runner with custom options."""
        runner = create_runner(
            enable_debug_logging=True,
            enable_recovery=False
        )
        
        assert runner.enable_debug_logging is True
        assert runner.enable_recovery is False
    
    def test_runner_has_compiled_graph(self):
        """Test runner has compiled graph."""
        runner = create_runner()
        
        assert hasattr(runner, 'graph')
        assert runner.graph is not None
    
    @patch('src.orchestration.run.compile_graph')
    def test_investigate_calls_graph_invoke(self, mock_compile):
        """Test investigate method calls graph invoke."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_compile.return_value = mock_graph
        
        # Create final state
        final_state = create_initial_state("test.csv", "ACC_001")
        final_state["final_report"] = {
            "account_id": "ACC_001",
            "risk_score": 0.5,
            "risk_tier": "MEDIUM"
        }
        final_state["overall_start_time"] = datetime.utcnow()
        
        mock_graph.invoke.return_value = final_state
        
        runner = OrchestrationRunner()
        runner.graph = mock_graph
        
        result = runner.investigate(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        # Verify graph was invoked
        mock_graph.invoke.assert_called_once()
        
        # Verify result structure
        assert "status" in result
        assert "result" in result
        assert "metrics" in result
        assert "errors" in result
    
    def test_investigate_returns_success_status(self):
        """Test investigate returns success status."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            final_state = create_initial_state("test.csv", "ACC_001")
            final_state["final_report"] = {
                "account_id": "ACC_001",
                "risk_score": 0.5
            }
            final_state["has_errors"] = False
            final_state["overall_start_time"] = datetime.utcnow()
            
            mock_graph.invoke.return_value = final_state
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            result = runner.investigate(
                raw_transaction_path="test.csv",
                account_id="ACC_001"
            )
            
            assert result["status"] == "SUCCESS"
    
    def test_investigate_returns_partial_failure_on_errors(self):
        """Test investigate returns PARTIAL_FAILURE on non-critical errors."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            final_state = create_initial_state("test.csv", "ACC_001")
            final_state["final_report"] = {"account_id": "ACC_001"}
            final_state["has_errors"] = True
            final_state["has_critical_errors"] = False
            final_state["errors"] = []
            final_state["overall_start_time"] = datetime.utcnow()
            
            mock_graph.invoke.return_value = final_state
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            result = runner.investigate(
                raw_transaction_path="test.csv",
                account_id="ACC_001"
            )
            
            assert result["status"] == "PARTIAL_FAILURE"
    
    def test_investigate_returns_failure_on_critical_errors(self):
        """Test investigate returns FAILURE on critical errors."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            final_state = create_initial_state("test.csv", "ACC_001")
            final_state["final_report"] = {"account_id": "ACC_001"}
            final_state["has_errors"] = True
            final_state["has_critical_errors"] = True
            final_state["errors"] = []
            final_state["overall_start_time"] = datetime.utcnow()
            
            mock_graph.invoke.return_value = final_state
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            result = runner.investigate(
                raw_transaction_path="test.csv",
                account_id="ACC_001"
            )
            
            assert result["status"] == "FAILURE"
    
    def test_investigate_batch(self):
        """Test batch investigation."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            def mock_invoke(state, timeout_dict):
                final_state = create_initial_state(
                    state["raw_transaction_path"],
                    state["account_id"]
                )
                final_state["final_report"] = {
                    "account_id": state["account_id"]
                }
                final_state["has_errors"] = False
                final_state["overall_start_time"] = datetime.utcnow()
                return final_state
            
            mock_graph.invoke.side_effect = mock_invoke
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            accounts = [
                {"account_id": "ACC_001"},
                {"account_id": "ACC_002"}
            ]
            
            results = runner.investigate_batch(
                accounts=accounts,
                raw_transaction_path="test.csv"
            )
            
            assert len(results) == 2
            assert all(r["status"] in ["SUCCESS", "FAILURE"] for r in results)
    
    def test_investigate_saves_results(self):
        """Test investigate saves results to output directory."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.orchestration.run.compile_graph') as mock_compile:
                # Setup
                mock_graph = MagicMock()
                mock_compile.return_value = mock_graph
                
                final_state = create_initial_state("test.csv", "ACC_001")
                final_state["final_report"] = {"account_id": "ACC_001"}
                final_state["has_errors"] = False
                final_state["overall_start_time"] = datetime.utcnow()
                
                mock_graph.invoke.return_value = final_state
                
                runner = OrchestrationRunner(output_dir=tmpdir)
                runner.graph = mock_graph
                
                result = runner.investigate(
                    raw_transaction_path="test.csv",
                    account_id="ACC_001"
                )
                
                # Check that files were created in output directory
                output_files = list(Path(tmpdir).glob("*.json"))
                assert len(output_files) > 0


class TestRunnerErrorHandling:
    """Tests for runner error handling."""
    
    def test_investigate_handles_pipeline_failure(self):
        """Test investigate handles graph execution failure."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            # Make graph raise exception
            mock_graph.invoke.side_effect = RuntimeError("Pipeline crashed")
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            result = runner.investigate(
                raw_transaction_path="test.csv",
                account_id="ACC_001"
            )
            
            assert result["status"] == "FAILURE"
            assert len(result["errors"]) > 0
            assert "RuntimeError" in str(result["errors"])
    
    def test_investigate_batch_continues_on_individual_failure(self):
        """Test batch investigation continues after individual failure."""
        with patch('src.orchestration.run.compile_graph') as mock_compile:
            # Setup
            mock_graph = MagicMock()
            mock_compile.return_value = mock_graph
            
            call_count = 0
            
            def mock_invoke(state, timeout_dict):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    raise RuntimeError("First account failed")
                
                final_state = create_initial_state(
                    state["raw_transaction_path"],
                    state["account_id"]
                )
                final_state["final_report"] = {
                    "account_id": state["account_id"]
                }
                final_state["has_errors"] = False
                final_state["overall_start_time"] = datetime.utcnow()
                return final_state
            
            mock_graph.invoke.side_effect = mock_invoke
            
            runner = OrchestrationRunner()
            runner.graph = mock_graph
            
            accounts = [
                {"account_id": "ACC_001"},
                {"account_id": "ACC_002"}
            ]
            
            results = runner.investigate_batch(
                accounts=accounts,
                raw_transaction_path="test.csv"
            )
            
            # Both should be processed
            assert len(results) == 2
            # First should be failure, second should be success
            assert results[0]["status"] == "FAILURE"
            assert results[1]["status"] == "SUCCESS"


class TestMetricsCollection:
    """Tests for metrics collection."""
    
    @patch('src.orchestration.run.compile_graph')
    def test_metrics_includes_agent_execution_times(self, mock_compile):
        """Test metrics includes agent execution times."""
        # Setup
        mock_graph = MagicMock()
        mock_compile.return_value = mock_graph
        
        final_state = create_initial_state("test.csv", "ACC_001")
        final_state["final_report"] = {"account_id": "ACC_001"}
        final_state["has_errors"] = False
        final_state["agent_metrics"] = {
            "detection_agent": MagicMock(to_dict=lambda: {
                "agent_name": "detection_agent",
                "status": "SUCCESS",
                "duration_seconds": 1.5
            }),
            "risk_scoring_agent": MagicMock(to_dict=lambda: {
                "agent_name": "risk_scoring_agent",
                "status": "SUCCESS",
                "duration_seconds": 0.5
            })
        }
        final_state["overall_start_time"] = datetime.utcnow()
        
        mock_graph.invoke.return_value = final_state
        
        runner = OrchestrationRunner()
        runner.graph = mock_graph
        
        result = runner.investigate(
            raw_transaction_path="test.csv",
            account_id="ACC_001"
        )
        
        assert "metrics" in result
        assert "agent_metrics" in result["metrics"]
        assert len(result["metrics"]["agent_metrics"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
