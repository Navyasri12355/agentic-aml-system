# File: src/orchestration/run.py
"""
Phase 3 Orchestration Runner

Provides execution entry points for the LangGraph AML investigation pipeline:
- Single account investigation
- Batch processing
- API integration

Features:
  - Complete execution lifecycle management
  - Error recovery and fallback strategies
  - Metrics collection and reporting
  - Audit trail logging
"""

import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.orchestration.state import AMLAgentState, create_initial_state
from src.orchestration.graph import compile_graph
from src.orchestration.errors import (
    handle_agent_error, add_error_to_state, create_fallback_final_report
)

logger = logging.getLogger(__name__)


class OrchestrationRunner:
    """
    Runner for Phase 3 LangGraph orchestration pipeline.
    
    Manages:
      - Graph compilation and execution
      - State management
      - Error recovery
      - Metrics collection
      - Audit logging
    """
    
    def __init__(
        self,
        enable_debug_logging: bool = False,
        enable_recovery: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize orchestration runner.
        
        Args:
            enable_debug_logging: Enable detailed debug logging
            enable_recovery: Enable automatic error recovery
            output_dir: Directory for saving reports (optional)
        """
        self.enable_debug_logging = enable_debug_logging
        self.enable_recovery = enable_recovery
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.graph = compile_graph()
        logger.info("Orchestration runner initialized")
    
    def investigate(
        self,
        raw_transaction_path: str,
        account_id: str,
        hop_radius: int = 2,
        time_window_days: int = 30,
        max_neighbors: int = 50,
        contamination: float = 0.02,
        execution_id: Optional[str] = None,
        investigation_type: str = "AUTO",
        priority_level: int = 5
    ) -> Dict[str, Any]:
        """
        Run investigation for single account.
        
        Args:
            raw_transaction_path: Path to transaction CSV
            account_id: Target account ID
            hop_radius: Graph expansion depth
            time_window_days: Temporal lookback window
            max_neighbors: Max neighbors per expansion
            contamination: Isolation Forest contamination
            execution_id: Optional execution identifier
            investigation_type: Type of investigation
            priority_level: Priority level (1=highest)
        
        Returns:
            Dictionary with investigation results:
            {
                "status": "SUCCESS" | "PARTIAL_FAILURE" | "FAILURE",
                "result": final_report dict,
                "metrics": execution_metrics dict,
                "errors": list of error summaries,
                "execution_id": unique execution ID
            }
        """
        logger.info(f"Starting investigation for account: {account_id}")
        
        # Create initial state
        initial_state = create_initial_state(
            raw_transaction_path=raw_transaction_path,
            account_id=account_id,
            hop_radius=hop_radius,
            time_window_days=time_window_days,
            max_neighbors=max_neighbors,
            contamination=contamination,
            execution_id=execution_id
        )
        
        # Set feature flags
        initial_state["enable_debug_logging"] = self.enable_debug_logging
        initial_state["enable_recovery"] = self.enable_recovery
        initial_state["investigation_type"] = investigation_type
        initial_state["priority_level"] = priority_level
        
        try:
            # Execute graph
            logger.info("Invoking LangGraph pipeline...")
            final_state = self._execute_graph(initial_state)
            
            # Process results
            return self._process_results(final_state, initial_state)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return self._handle_pipeline_failure(
                initial_state, e, investigation_type
            )
    
    def _execute_graph(self, initial_state: AMLAgentState) -> AMLAgentState:
        """
        Execute the compiled graph with timeout and error handling.
        
        Args:
            initial_state: Initial orchestration state
        
        Returns:
            Final orchestration state after execution
        """
        try:
            final_state = self.graph.invoke(
                initial_state,
                {"timeout": 300}  # 5 minute timeout
            )
            return final_state
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            raise
    
    def _process_results(
        self,
        final_state: AMLAgentState,
        initial_state: AMLAgentState
    ) -> Dict[str, Any]:
        """
        Process final state into result dictionary.
        
        Args:
            final_state: Final orchestration state
            initial_state: Initial orchestration state
        
        Returns:
            Processed results dictionary
        """
        # Compute execution metrics
        overall_end_time = datetime.utcnow()
        overall_start_time = initial_state.get("overall_start_time", overall_end_time)
        duration_seconds = (overall_end_time - overall_start_time).total_seconds()
        
        # Determine status
        has_errors = final_state.get("has_errors", False)
        has_critical_errors = final_state.get("has_critical_errors", False)
        
        if has_critical_errors:
            status = "FAILURE"
        elif has_errors:
            status = "PARTIAL_FAILURE"
        else:
            status = "SUCCESS"
        
        # Prepare error summaries
        error_summaries = []
        for error_ctx in final_state.get("errors", []):
            error_summaries.append({
                "agent": error_ctx.agent_name,
                "error_type": error_ctx.error_type,
                "message": error_ctx.error_message
            })
        
        # Prepare metrics
        metrics = {
            "execution_id": final_state.get("execution_id"),
            "account_id": final_state.get("account_id"),
            "total_duration_seconds": duration_seconds,
            "start_time": overall_start_time.isoformat(),
            "end_time": overall_end_time.isoformat(),
            "pipeline_version": final_state.get("pipeline_version"),
            "agent_metrics": {
                name: metrics.to_dict()
                for name, metrics in final_state.get("agent_metrics", {}).items()
            }
        }
        
        # Update final report with metrics
        final_report = final_state.get("final_report", {})
        if final_report:
            final_report["execution_duration_seconds"] = duration_seconds
            final_report["pipeline_version"] = final_state.get("pipeline_version")
            final_report["agent_count"] = len(final_state.get("agent_metrics", {}))
            final_report["has_errors"] = has_errors
        
        result = {
            "status": status,
            "result": final_report,
            "metrics": metrics,
            "errors": error_summaries,
            "execution_id": final_state.get("execution_id")
        }
        
        # Save to output directory if configured
        if self.output_dir:
            self._save_results(result)
        
        logger.info(
            f"Investigation complete: status={status}, "
            f"duration={duration_seconds:.2f}s, errors={len(error_summaries)}"
        )
        
        return result
    
    def _handle_pipeline_failure(
        self,
        initial_state: AMLAgentState,
        error: Exception,
        investigation_type: str
    ) -> Dict[str, Any]:
        """
        Handle catastrophic pipeline failure.
        
        Args:
            initial_state: Initial state
            error: Exception that caused failure
            investigation_type: Investigation type
        
        Returns:
            Failure result dictionary
        """
        logger.error(f"Pipeline catastrophic failure: {error}", exc_info=True)
        
        overall_end_time = datetime.utcnow()
        overall_start_time = initial_state.get("overall_start_time", overall_end_time)
        duration_seconds = (overall_end_time - overall_start_time).total_seconds()
        
        fallback_report = create_fallback_final_report(
            initial_state,
            f"Pipeline execution failed: {str(error)}"
        )
        fallback_report["execution_duration_seconds"] = duration_seconds
        
        result = {
            "status": "FAILURE",
            "result": fallback_report,
            "metrics": {
                "execution_id": initial_state.get("execution_id"),
                "account_id": initial_state.get("account_id"),
                "total_duration_seconds": duration_seconds,
                "pipeline_status": "CRASHED"
            },
            "errors": [{
                "agent": "orchestration_runner",
                "error_type": type(error).__name__,
                "message": str(error)
            }],
            "execution_id": initial_state.get("execution_id")
        }
        
        if self.output_dir:
            self._save_results(result)
        
        return result
    
    def investigate_batch(
        self,
        accounts: List[Dict[str, Any]],
        raw_transaction_path: str,
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Investigate multiple accounts.
        
        Args:
            accounts: List of account dicts with "account_id" key
            raw_transaction_path: Path to transaction CSV
            max_workers: Number of parallel workers (currently: 1=sequential)
        
        Returns:
            List of investigation results
        """
        logger.info(f"Starting batch investigation of {len(accounts)} accounts")
        
        results = []
        for i, account_info in enumerate(accounts, 1):
            account_id = account_info.get("account_id")
            logger.info(f"[{i}/{len(accounts)}] Investigating {account_id}")
            
            try:
                result = self.investigate(
                    raw_transaction_path=raw_transaction_path,
                    account_id=account_id,
                    hop_radius=account_info.get("hop_radius", 2),
                    time_window_days=account_info.get("time_window_days", 30),
                    investigation_type=account_info.get("investigation_type", "AUTO")
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch account {account_id} failed: {e}")
                results.append({
                    "status": "FAILURE",
                    "account_id": account_id,
                    "error": str(e)
                })
        
        logger.info(
            f"Batch investigation complete: "
            f"{len([r for r in results if r.get('status') == 'SUCCESS'])} successful, "
            f"{len([r for r in results if r.get('status') != 'SUCCESS'])} failed"
        )
        
        return results
    
    def _save_results(self, result: Dict[str, Any]) -> None:
        """
        Save investigation results to output directory.
        
        Args:
            result: Investigation result dictionary
        """
        try:
            execution_id = result.get("execution_id", "unknown")
            account_id = result.get("result", {}).get("account_id", "unknown")
            
            # Create filename
            filename = f"{account_id}_{execution_id}.json"
            filepath = self.output_dir / filename
            
            # Save JSON
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def create_runner(
    enable_debug_logging: bool = False,
    enable_recovery: bool = True,
    output_dir: Optional[str] = None
) -> OrchestrationRunner:
    """
    Factory function to create orchestration runner.
    
    Args:
        enable_debug_logging: Enable debug logging
        enable_recovery: Enable error recovery
        output_dir: Output directory for reports
    
    Returns:
        Initialized OrchestrationRunner
    """
    return OrchestrationRunner(
        enable_debug_logging=enable_debug_logging,
        enable_recovery=enable_recovery,
        output_dir=output_dir
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Standalone Execution
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Command-line interface for orchestration runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 3 AML Investigation Orchestration Pipeline"
    )
    parser.add_argument(
        "--transaction-file",
        required=True,
        help="Path to raw transaction CSV"
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Account ID to investigate"
    )
    parser.add_argument(
        "--hop-radius",
        type=int,
        default=2,
        help="Graph expansion radius (default: 2)"
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=30,
        help="Temporal window in days (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save reports"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create runner
    runner = create_runner(
        enable_debug_logging=args.debug,
        output_dir=args.output_dir
    )
    
    # Run investigation
    result = runner.investigate(
        raw_transaction_path=args.transaction_file,
        account_id=args.account_id,
        hop_radius=args.hop_radius,
        time_window_days=args.time_window
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"INVESTIGATION RESULTS: {args.account_id}")
    print("="*80)
    print(f"Status: {result['status']}")
    print(f"Duration: {result['metrics']['total_duration_seconds']:.2f}s")
    print(f"\nRisk Score: {result['result'].get('risk_score', 'N/A'):.2f}")
    print(f"Risk Tier: {result['result'].get('risk_tier', 'N/A')}")
    print(f"Patterns: {result['result'].get('detected_patterns', [])}")
    
    if result.get("errors"):
        print(f"\nErrors encountered: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"  - {error['agent']}: {error['error_type']}")
    
    print("="*80 + "\n")
    
    return 0 if result["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    exit(main())
