# File: src/pipeline/run_phase3.py

import os
import json
import pandas as pd
from tqdm import tqdm
from langgraph.graph import StateGraph, END

from src.pipeline.state_phase3 import InvestigationState
from src.agents.graph_agent import GraphAgent
from src.agents.feature_agent import FeatureAgent
from src.agents.pattern_agent import PatternAgent
from src.agents.risk_agent import RiskAgent
from src.utils.global_stats import build_global_stats


# ─── Node Factories ───────────────────────────────────────────────────────────

def make_graph_node(graph_agent: GraphAgent):
    def graph_node(state: InvestigationState) -> dict:
        try:
            result = graph_agent.build_subgraph(
                account_id=state["account_id"],
                flag_date=state["flagged_row"]["timestamp"],
                hop_radius=2,
                time_window_days=30,
                max_neighbors=50,
            )
            return {"graph_result": result}
        except Exception as e:
            return {"error": f"GraphAgent failed: {e}"}
    return graph_node


def make_feature_node(feature_agent: FeatureAgent):
    def feature_node(state: InvestigationState) -> dict:
        if state.get("error") or state.get("graph_result") is None:
            return {}
        try:
            result = feature_agent.extract_features(state["graph_result"])
            return {"feature_result": result}
        except Exception as e:
            return {"error": f"FeatureAgent failed: {e}"}
    return feature_node


def make_pattern_node(pattern_agent: PatternAgent):
    def pattern_node(state: InvestigationState) -> dict:
        if state.get("error") or state.get("feature_result") is None:
            return {}
        try:
            result = pattern_agent.detect_patterns(state["feature_result"])
            return {"pattern_result": result}
        except Exception as e:
            return {"error": f"PatternAgent failed: {e}"}
    return pattern_node


def make_risk_node(risk_agent: RiskAgent):
    def risk_node(state: InvestigationState) -> dict:
        if state.get("error") or state.get("pattern_result") is None:
            return {}
        try:
            result = risk_agent.compute_risk(
                flagged_row=state["flagged_row"],
                feature_result=state["feature_result"],
                pattern_result=state["pattern_result"],
                graph_result=state["graph_result"],
                transaction_id=state["transaction_id"],
            )
            routing = result.get("routing_decision", "EXIT")
            return {"risk_result": result, "routing_decision": routing}
        except Exception as e:
            return {"error": f"RiskAgent failed: {e}", "routing_decision": "EXIT"}
    return risk_node


def explanation_node(state: InvestigationState) -> dict:
    """
    Placeholder for Phase 4 Explanation Agent + SAR generation.
    Currently just marks the case as pending explanation.
    """
    risk = state.get("risk_result", {})
    risk["explanation_status"] = "pending_phase4"
    return {"risk_result": risk}


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
                flat_results.append(risk)
            elif final_state.get("error"):
                errors.append({"transaction_id": transaction_id, "error": final_state["error"]})
        except Exception as e:
            errors.append({"transaction_id": transaction_id, "error": str(e)})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(flat_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n✅ Phase 3 complete")
    print(f"   Processed : {len(flat_results) + len(errors)}")
    print(f"   Succeeded : {len(flat_results)}")
    print(f"   Errors    : {len(errors)}")
    print(f"   Output    : {output_path}")

    if flat_results:
        df_res = pd.DataFrame(flat_results)
        print("\n📊 Risk tier distribution:")
        for tier, cnt in df_res["risk_tier"].value_counts().items():
            print(f"   {tier}: {cnt}  ({cnt/len(df_res)*100:.1f}%)")

        inv = df_res[df_res["routing_decision"] == "INVESTIGATE"]
        print(f"\n🔍 Sent to Explanation Node (HIGH risk): {len(inv)}")
        print(f"   explanation_status: pending_phase4")

    # ── Evaluation ───────────────────────────────────────────────────────────
    label_col = next(
        (c for c in flagged_df.columns if c in ("is_laundering", "Is Laundering", "label")),
        None,
    )
    if label_col and flat_results:
        df_res = pd.DataFrame(flat_results)
        merged = flagged_df.merge(
            df_res[["transaction_id", "routing_decision"]],
            on="transaction_id", how="left",
        )
        actual   = int(merged[label_col].sum())
        inv_df   = merged[merged["routing_decision"] == "INVESTIGATE"]
        tp       = int(inv_df[label_col].sum())
        fp       = len(inv_df) - tp
        missed   = actual - tp

        print(f"\n{'='*55}")
        print(f"EVALUATION  (rows {START_IDX}:{END_IDX})")
        print(f"{'='*55}")
        print(f"  Actual laundering    : {actual}")
        print(f"  Sent to INVESTIGATE  : {len(inv_df)}")
        print(f"  True positives       : {tp}")
        print(f"  False positives      : {fp}")
        print(f"  Missed               : {missed}")
        if actual:
            print(f"  Recall               : {tp/actual*100:.1f}%")
        if len(inv_df):
            print(f"  Precision            : {tp/len(inv_df)*100:.1f}%")


if __name__ == "__main__":
    main()