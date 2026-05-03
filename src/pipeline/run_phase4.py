# File: src/pipeline/run_phase4.py

import os
import json
import random
from dotenv import load_dotenv

load_dotenv()

from src.agents.explanation_agent import generate_sar_report

def main():
    phase3_path  = "data/processed/phase3_risk_results.json"
    reports_dir  = "data/reports"

    print("Loading Phase 3 risk results to select samples...")
    if not os.path.exists(phase3_path):
        print(f"Error: {phase3_path} not found. Run run_phase3.py first.")
        return

    with open(phase3_path, "r") as f:
        phase3_results = json.load(f)

    # Group by risk tier
    tier_groups = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for res in phase3_results:
        tier = res.get("risk_tier", "LOW")
        if tier in tier_groups:
            tier_groups[tier].append(res)

    selected_results = []
    
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        tier_list = tier_groups[tier]
        sampled = random.sample(tier_list, min(3, len(tier_list)))
        selected_results.extend(sampled)
        print(f"Selected {len(sampled)} samples for {tier} risk.")

    print(f"\nRunning Phase 4 Explanation Agent on {len(selected_results)} selected transactions...")
    
    results_by_tier = {"HIGH": [], "MEDIUM": [], "LOW": []}
    errors = []

    for state in selected_results:
        transaction_id = state.get("transaction_id", "UNKNOWN")
        account_id = state.get("account_id", "UNKNOWN")
        risk_score = state.get("risk_score", 0.0)
        risk_tier = state.get("risk_tier", "LOW")
        
        # Extract the features and patterns saved by Phase 3
        features = state.get("_feature_result", {}).get("features", {}) if state.get("_feature_result") else {}
        pattern_result = state.get("_pattern_result", {}) if state.get("_pattern_result") else {}
        
        # Clean up the risk_result to only pass what's expected (remove the temp _ fields)
        risk_result = {k: v for k, v in state.items() if not k.startswith("_")}

        try:
            report = generate_sar_report(
                account_id=account_id,
                risk_score=risk_score,
                risk_tier=risk_tier,
                features=features,
                pattern_result=pattern_result,
                risk_result=risk_result,
            )
            
            if report:
                tier = report.get("risk_tier", "LOW")
                if tier in results_by_tier:
                    results_by_tier[tier].append(report)
                    
        except Exception as e:
            errors.append({"transaction_id": transaction_id, "error": str(e)})
            print(f"Exception processing {transaction_id}: {e}")

    # Export reports
    os.makedirs(reports_dir, exist_ok=True)
    
    for tier, reports in results_by_tier.items():
        if reports:
            tier_lower = tier.lower()
            output_path = os.path.join(reports_dir, f"sample_sar_{tier_lower}_risk.json")
            with open(output_path, "w") as f:
                json.dump(reports, f, indent=2)
            print(f"Saved {len(reports)} {tier} reports to {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[DONE] Phase 4 complete")
    for tier, reports in results_by_tier.items():
        print(f"   {tier} Risk SARs : {len(reports)}")
    print(f"   Errors        : {len(errors)}")

if __name__ == "__main__":
    main()
