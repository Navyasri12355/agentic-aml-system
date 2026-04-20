from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd, json, os, io, base64
from dotenv import load_dotenv
load_dotenv()

from src.orchestration.state import initial_state
from src.orchestration.graph import aml_pipeline

app = FastAPI(title="AML Investigation API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS","*").split(","), allow_methods=["*"], allow_headers=["*"])

class InvestigateRequest(BaseModel):
    account_id: str
    hop_radius: int = 2
    time_window_days: int = 30

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/investigate")
async def investigate(account_id: str, hop_radius: int = 2, time_window_days: int = 30, file: UploadFile = File(...)):
    contents = await file.read()
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f: f.write(contents)
    state = initial_state(tmp_path, account_id, hop_radius, time_window_days)
    result = aml_pipeline.invoke(state)
    if result.get("errors"):
        raise HTTPException(status_code=500, detail=result["errors"])
    report = result.get("final_report", {})
    subgraph = result.get("subgraph", {"nodes": [], "edges": []})
    return {**report, "subgraph": subgraph, "features": result.get("features", {})}