"""Phase 3 runner (phase-wise testing wrapper)

This script compiles the production `src.orchestration.graph` once and
invokes it per flagged transaction slice. It preserves the original
CLI interface but delegates execution to the production LangGraph
implementation so tests exercise the true pipeline.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.orchestration.state import create_initial_state
from src.orchestration.run import create_runner


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 using production orchestration graph")
    parser.add_argument("--clean-path", default="data/processed/phase1_full_results.csv")
    parser.add_argument("--flagged-path", default="data/processed/flagged_hybrid_final.csv")
    parser.add_argument("--output-path", default="data/processed/phase3_risk_results.json")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=100)
    parser.add_argument("--hop-radius", type=int, default=2)
    parser.add_argument("--time-window-days", type=int, default=30)
    parser.add_argument("--max-neighbors", type=int, default=50)
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()

    clean_path = args.clean_path
    flagged_path = args.flagged_path
    output_path = args.output_path

    print("Loading data...")
    clean_df = pd.read_csv(clean_path)
    flagged_df = pd.read_csv(flagged_path).iloc[args.start_idx:args.end_idx]

    # Ensure timestamps are datetime
    if "timestamp" in flagged_df.columns:
        flagged_df["timestamp"] = pd.to_datetime(flagged_df["timestamp"], errors="coerce")
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")

    print(f"Flagged slice: {len(flagged_df)} transactions (rows {args.start_idx}–{args.end_idx})")

    # Create an orchestration runner which wraps the compiled graph
    runner = create_runner(enable_debug_logging=False, output_dir=os.path.dirname(output_path) or None)

    flat_results: list[Any] = []
    errors: list[dict[str, Any]] = []

    print("Running Phase 3 pipeline (orchestration)...")
    for _, row in tqdm(flagged_df.iterrows(), total=len(flagged_df), unit="tx"):
        row_data = row.to_dict()
        account_id = str(row_data.get("sender_id") or row_data.get("receiver_id") or "")
        transaction_id = str(row_data.get("transaction_id", f"UNK_{account_id}"))

        # Create a minimal initial state using the orchestration factory
        state = create_initial_state(
            raw_transaction_path=clean_path,
            account_id=account_id,
            hop_radius=args.hop_radius,
            time_window_days=args.time_window_days,
            max_neighbors=args.max_neighbors,
            contamination=args.contamination,
        )

        # Inject the flagged row so the graph can use it directly
        state["flagged_row"] = row_data
        state["transaction_id"] = transaction_id
        state["overall_start_time"] = datetime.utcnow()

        try:
            result = runner.investigate(
                raw_transaction_path=clean_path,
                account_id=account_id,
                hop_radius=args.hop_radius,
                time_window_days=args.time_window_days,
                max_neighbors=args.max_neighbors,
                contamination=args.contamination,
            )

            final_report = result.get("result") or {}
            if final_report:
                final_report["transaction_id"] = transaction_id
                final_report["status"] = result.get("status")
                flat_results.append(final_report)
            else:
                errors.append({"transaction_id": transaction_id, "error": result.get("errors") or "No report produced"})

        except Exception as e:
            errors.append({"transaction_id": transaction_id, "error": str(e)})

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(flat_results, f, indent=2, default=str)

    # Summary
    print(f"\n✅ Phase 3 orchestration run complete")
    print(f"   Processed : {len(flat_results) + len(errors)}")
    print(f"   Succeeded : {len(flat_results)}")
    print(f"   Errors    : {len(errors)}")
    print(f"   Output    : {output_path}")


if __name__ == "__main__":
    main()