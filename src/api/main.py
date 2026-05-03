from __future__ import annotations

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


@app.post("/investigate")
async def investigate(
	file: UploadFile = File(...),
	account_id: str = Query(...),
	hop_radius: int = Query(2, ge=1, le=4),
	time_window_days: int = Query(30, ge=1, le=365),
	max_neighbors: int = Query(50, ge=1, le=500),
	contamination: float = Query(0.02, ge=0.0, le=1.0),
	priority_level: int = Query(5, ge=1, le=10),
) -> Dict[str, Any]:
	"""Phase 3 comprehensive investigation endpoint wired to the orchestration runner.

	Saves uploaded file to a temporary path and invokes the Phase 3 runner.
	"""
	import tempfile

	content = await file.read()
	if not content:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
		f.write(content)
		temp_path = f.name

	try:
		result = runner.investigate(
			raw_transaction_path=temp_path,
			account_id=account_id,
			hop_radius=hop_radius,
			time_window_days=time_window_days,
			max_neighbors=max_neighbors,
			contamination=contamination,
			priority_level=priority_level,
		)

		# Return a concise view for API clients
		return {
			"status": result.get("status"),
			"execution_id": result.get("execution_id"),
			"investigation": result.get("result"),
			"metrics": result.get("metrics"),
			"errors": result.get("errors"),
		}
	finally:
		try:
			os.unlink(temp_path)
		except Exception:
			pass


@app.post("/investigate/batch")
async def investigate_batch(
	file: UploadFile = File(...),
	account_ids: str = Query(...),  # comma-separated list
	hop_radius: int = Query(2, ge=1, le=4),
	time_window_days: int = Query(30, ge=1, le=365),
	max_neighbors: int = Query(50, ge=1, le=500),
) -> Dict[str, Any]:
	"""Batch investigation endpoint using the Phase 3 runner."""
	import tempfile

	content = await file.read()
	if not content:
		raise HTTPException(status_code=400, detail="Uploaded file is empty.")

	with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
		f.write(content)
		temp_path = f.name

	accounts = [ {"account_id": aid.strip()} for aid in account_ids.split(",") if aid.strip() ]

	try:
		results = runner.investigate_batch(
			accounts=accounts,
			raw_transaction_path=temp_path,
			max_workers=1
		)

		return {
			"status": "ok",
			"total": len(results),
			"results": results
		}
	finally:
		try:
			os.unlink(temp_path)
		except Exception:
			pass


@app.get("/")
def root() -> Dict[str, str]:
	return {"message": "AML Investigation API is running."}


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "healthy"}


if __name__ == "__main__":
	import uvicorn  # pyright: ignore[reportMissingImports]

	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)
