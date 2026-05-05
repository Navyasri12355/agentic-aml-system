from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, cast
import pandas as pd, json, os, io, base64, tempfile
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]

from src.orchestration.run import create_runner


app = FastAPI(
	title="AML Investigation API",
	version="1.0.0",
	description="API for Phase 3 AML investigations.",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


# -----------------------------
# Phase 3 Orchestration API
# -----------------------------
# Initialize a singleton orchestration runner for the FastAPI process.
runner = create_runner(
	enable_debug_logging=False,
	enable_recovery=True,
	output_dir=os.getenv("PHASE3_REPORT_DIR", "reports/")
)

@app.get("/")
def root():
    return {"message": "AML Investigation API"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/investigate")
async def investigate(account_id: str, hop_radius: int = 2, time_window_days: int = 30, file: UploadFile = File(None)):
    
    # 1. First, try to load pre-computed Phase 3 results for 100% accuracy
    phase3_path = "data/processed/phase3_risk_results.json"
    if os.path.exists(phase3_path):
        with open(phase3_path, "r") as f:
            phase3_results = json.load(f)
            
        for state in phase3_results:
            if state.get("account_id") == account_id:
                # Pre-computed hit! Extract exact metrics
                risk_score = state.get("risk_score", 0.0)
                risk_tier = state.get("risk_tier", "LOW")
                features = state.get("_feature_result", {}).get("features", {})
                pattern_result = state.get("_pattern_result", {})
                
                # Clean risk_result
                risk_result = {k: v for k, v in state.items() if not k.startswith("_")}
                
                # Generate SAR narrative directly
                from src.agents.explanation_agent import generate_sar_report
                report = generate_sar_report(
                    account_id=account_id,
                    risk_score=risk_score,
                    risk_tier=risk_tier,
                    features=features,
                    pattern_result=pattern_result,
                    risk_result=risk_result
                )
                
                subgraph = state.get("graph_data", {"nodes": [], "edges": []})
                
                # Enrich subgraph nodes with their individual risk scores
                if subgraph and "nodes" in subgraph:
                    # Create a quick lookup map from all phase 3 results
                    risk_lookup = {r.get("account_id"): r.get("risk_score", 0.0) for r in phase3_results}
                    for node in subgraph["nodes"]:
                        node["risk_score"] = risk_lookup.get(node["id"], 0.0)
                        
                return {**report, "subgraph": subgraph, "features": features}

    # 2. If not found in pre-computed results, run live pipeline fallback
    if file:
        contents = await file.read()
        tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(tmp_path, "wb") as f: f.write(contents)
    else:
        tmp_path = "data/processed/phase1_full_results.csv"
        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=400, detail="Default dataset not found and no file uploaded.")
            
    state = initial_state(tmp_path, account_id, hop_radius, time_window_days)
    result = cast(Any, aml_pipeline).invoke(input=state)
    if result.get("errors"):
        raise HTTPException(status_code=500, detail=result["errors"])
    report = result.get("final_report", {})
    subgraph = result.get("subgraph", {"nodes": [], "edges": []})
    return {**report, "subgraph": subgraph, "features": result.get("features", {})}

