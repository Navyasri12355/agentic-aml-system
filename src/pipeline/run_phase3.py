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

# ─── Routing Logic ────────────────────────────────────────────────────────────

def route_after_risk(state: InvestigationState) -> str:
    if state.get("routing_decision") == "INVESTIGATE":
        return "explanation_node"
    return END


# ─── Graph Builder ────────────────────────────────────────────────────────────

def build_investigation_graph(graph_agent, feature_agent, pattern_agent, risk_agent):
    builder = StateGraph(InvestigationState)

    builder.add_node("graph_node",       make_graph_node(graph_agent))
    builder.add_node("feature_node",     make_feature_node(feature_agent))
    builder.add_node("pattern_node",     make_pattern_node(pattern_agent))
    builder.add_node("risk_node",        make_risk_node(risk_agent))
    builder.add_node("explanation_node", explanation_node)

    builder.set_entry_point("graph_node")

    builder.add_edge("graph_node",    "feature_node")
    builder.add_edge("feature_node",  "pattern_node")
    builder.add_edge("pattern_node",  "risk_node")

    builder.add_conditional_edges(
        "risk_node",
        route_after_risk,
        {
            "explanation_node": "explanation_node",
            END: END,
        },
    )

    builder.add_edge("explanation_node", END)

    return builder.compile()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    START_IDX = 575
    END_IDX   = 595

    clean_path   = "data/processed/phase1_full_results.csv"
    flagged_path = "data/processed/flagged_hybrid_final.csv"
    output_path  = "data/processed/phase3_risk_results.json"

    print("Loading data...")
    clean_df   = pd.read_csv(clean_path)
    flagged_df = pd.read_csv(flagged_path).iloc[START_IDX:END_IDX]
    flagged_df["timestamp"] = pd.to_datetime(flagged_df["timestamp"])
    clean_df["timestamp"]   = pd.to_datetime(clean_df["timestamp"])

    print(f"Flagged slice: {len(flagged_df)} transactions (rows {START_IDX}–{END_IDX})")

    global_stats = build_global_stats(clean_df)

    print("Initialising agents...")
    graph_agent   = GraphAgent(clean_df)
    feature_agent = FeatureAgent(global_stats)
    pattern_agent = PatternAgent(global_stats)
    risk_agent    = RiskAgent(global_stats)

    print("Compiling LangGraph...")
    graph = build_investigation_graph(
        graph_agent, feature_agent, pattern_agent, risk_agent
    )

    flat_results = []
    errors       = []

    print("Running Phase 3 pipeline...")
    for row in tqdm(flagged_df.itertuples(index=False), total=len(flagged_df), unit="tx"):
        account_id     = str(getattr(row, "sender_id", None) or getattr(row, "receiver_id", ""))
        transaction_id = str(getattr(row, "transaction_id", f"UNK_{account_id}"))

        initial_state: InvestigationState = {
            "transaction_id":    transaction_id,
            "account_id":        account_id,
            "flagged_row":       row._asdict(),
            "graph_result":      None,
            "feature_result":    None,
            "pattern_result":    None,
            "risk_result":       None,
            "routing_decision":  "EXIT",
            "error":             None,
        }

        try:
            final_state = graph.invoke(initial_state)
            risk = final_state.get("risk_result")
            if risk:
                risk["_error"] = final_state.get("error")
                risk["_feature_result"] = final_state.get("feature_result")
                risk["_pattern_result"] = final_state.get("pattern_result")
                flat_results.append(risk)
            elif final_state.get("error"):
                errors.append({"transaction_id": transaction_id, "error": final_state["error"]})
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