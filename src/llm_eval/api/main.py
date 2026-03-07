"""FastAPI app: leaderboard, runs, model items; static dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from llm_eval.storage.sqlite import (
    init_db,
    get_leaderboard,
    run_to_report,
    get_all_run_ids,
)
from llm_eval.storage.sqlite import get_run_items
import sqlite3

# Default DB path: cwd/data/eval.db when run from project root, or env LLM_EVAL_DB
DEFAULT_DB_PATH = Path.cwd() / "data" / "eval.db"
DB_PATH = Path(os.environ.get("LLM_EVAL_DB", str(DEFAULT_DB_PATH)))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
init_db(DB_PATH)

app = FastAPI(title="LLM Evaluation Harness")


def _conn():
    return sqlite3.connect(str(DB_PATH))


@app.get("/leaderboard")
def leaderboard():
    """Models ranked by average score per rubric (from SQLite history)."""
    conn = _conn()
    try:
        rows = get_leaderboard(conn)
        return {"leaderboard": rows}
    finally:
        conn.close()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Full EvaluationRunReport for a run (including cost_metadata)."""
    conn = _conn()
    try:
        report = run_to_report(conn, run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return report.model_dump()
    finally:
        conn.close()


@app.get("/models/{model_name}/items")
def get_model_items(model_name: str):
    """All evaluated items for a model (all runs for that model)."""
    conn = _conn()
    try:
        runs = conn.execute(
            "SELECT run_id FROM runs WHERE model_name = ? ORDER BY timestamp DESC",
            (model_name,),
        ).fetchall()
        items_by_question: dict[str, dict] = {}
        for (run_id,) in runs:
            items = get_run_items(conn, run_id)
            for it in items:
                key = it.question
                if key not in items_by_question:
                    items_by_question[key] = {
                        "question": it.question,
                        "response": it.response,
                        "scores": it.scores,
                        "reasoning": it.reasoning,
                        "overall_score": it.overall_score,
                        "runs": [],
                    }
                items_by_question[key]["runs"].append(
                    {"run_id": run_id, "overall_score": it.overall_score}
                )
        return {"model_name": model_name, "items": list(items_by_question.values())}
    finally:
        conn.close()


@app.get("/runs")
def list_runs():
    """List all run IDs (for drilldown)."""
    conn = _conn()
    try:
        ids = get_all_run_ids(conn)
        runs_meta = []
        for run_id in ids:
            report = run_to_report(conn, run_id)
            if report:
                runs_meta.append({
                    "run_id": report.run_id,
                    "model_name": report.model_name,
                    "rubric_name": report.rubric_name,
                    "timestamp": report.timestamp,
                    "position_bias_rate": report.position_bias_rate,
                    "verbosity_correlation": report.verbosity_correlation,
                    "cohens_kappa": report.cohens_kappa,
                    "cost_metadata": report.cost_metadata.model_dump(),
                })
        return {"runs": runs_meta}
    finally:
        conn.close()


# Mount static dashboard
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {"message": "Dashboard static files not found. Mount /static or add api/static/."}
