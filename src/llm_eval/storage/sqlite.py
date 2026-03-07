"""SQLite schema and operations: runs (with cost columns), run_items."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from llm_eval.contracts import (
    CostMetadata,
    EvaluationItemResult,
    EvaluationRunReport,
)


def init_db(path: str | Path) -> None:
    """Create tables if they do not exist."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                rubric_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                position_bias_rate REAL NOT NULL,
                verbosity_correlation REAL NOT NULL,
                cohens_kappa REAL,
                total_tokens_in INTEGER NOT NULL DEFAULT 0,
                total_tokens_out INTEGER NOT NULL DEFAULT 0,
                total_estimated_cost REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_items (
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                question TEXT NOT NULL,
                response TEXT NOT NULL,
                response_length_tokens INTEGER NOT NULL DEFAULT 0,
                scores_json TEXT NOT NULL,
                reasoning_json TEXT NOT NULL,
                overall_score REAL NOT NULL,
                failed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, item_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    model_name: str,
    rubric_name: str,
    timestamp: str,
    position_bias_rate: float,
    verbosity_correlation: float,
    cohens_kappa: float | None,
    cost_metadata: CostMetadata,
) -> None:
    """Insert one run row (including cost columns)."""
    conn.execute(
        """INSERT INTO runs (
            run_id, model_name, rubric_name, timestamp,
            position_bias_rate, verbosity_correlation, cohens_kappa,
            total_tokens_in, total_tokens_out, total_estimated_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            model_name,
            rubric_name,
            timestamp,
            position_bias_rate,
            verbosity_correlation,
            cohens_kappa,
            cost_metadata.total_tokens_in,
            cost_metadata.total_tokens_out,
            cost_metadata.total_estimated_cost,
        ),
    )


def insert_run_items(
    conn: sqlite3.Connection,
    run_id: str,
    items: list[EvaluationItemResult],
) -> None:
    """Insert all item rows for a run."""
    for it in items:
        conn.execute(
            """INSERT INTO run_items (
                run_id, item_id, question, response, response_length_tokens,
                scores_json, reasoning_json, overall_score, failed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                it.item_id,
                it.question,
                it.response,
                it.response_length_tokens,
                json.dumps(it.scores),
                json.dumps(it.reasoning),
                it.overall_score,
                1 if it.failed else 0,
            ),
        )


def get_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    """Load run metadata by run_id. Returns dict with cost fields."""
    row = conn.execute(
        """SELECT run_id, model_name, rubric_name, timestamp,
                  position_bias_rate, verbosity_correlation, cohens_kappa,
                  total_tokens_in, total_tokens_out, total_estimated_cost
           FROM runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "run_id": row[0],
        "model_name": row[1],
        "rubric_name": row[2],
        "timestamp": row[3],
        "position_bias_rate": row[4],
        "verbosity_correlation": row[5],
        "cohens_kappa": row[6],
        "total_tokens_in": row[7],
        "total_tokens_out": row[8],
        "total_estimated_cost": row[9],
    }


def get_run_items(conn: sqlite3.Connection, run_id: str) -> list[EvaluationItemResult]:
    """Load all items for a run."""
    rows = conn.execute(
        """SELECT item_id, question, response, response_length_tokens,
                  scores_json, reasoning_json, overall_score, failed
           FROM run_items WHERE run_id = ? ORDER BY item_id""",
        (run_id,),
    ).fetchall()
    return [
        EvaluationItemResult(
            item_id=r[0],
            question=r[1],
            response=r[2],
            response_length_tokens=r[3],
            scores=json.loads(r[4]) if r[4] else {},
            reasoning=json.loads(r[5]) if r[5] else {},
            overall_score=r[6],
            failed=bool(r[7]),
        )
        for r in rows
    ]


def run_to_report(conn: sqlite3.Connection, run_id: str) -> EvaluationRunReport | None:
    """Build EvaluationRunReport from DB for a run."""
    run = get_run(conn, run_id)
    if not run:
        return None
    items = get_run_items(conn, run_id)
    return EvaluationRunReport(
        run_id=run["run_id"],
        model_name=run["model_name"],
        rubric_name=run["rubric_name"],
        items=items,
        position_bias_rate=run["position_bias_rate"],
        verbosity_correlation=run["verbosity_correlation"],
        cohens_kappa=run["cohens_kappa"],
        cost_metadata=CostMetadata(
            total_tokens_in=run["total_tokens_in"],
            total_tokens_out=run["total_tokens_out"],
            total_estimated_cost=run["total_estimated_cost"],
        ),
        timestamp=run["timestamp"],
    )


def get_leaderboard(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Aggregate by (model_name, rubric_name): avg overall_score, run_count. Sorted by rubric then score desc."""
    rows = conn.execute("""
        SELECT r.model_name, r.rubric_name,
               AVG(ri.overall_score) AS avg_score,
               COUNT(DISTINCT r.run_id) AS run_count
        FROM run_items ri
        JOIN runs r ON r.run_id = ri.run_id
        GROUP BY r.model_name, r.rubric_name
        ORDER BY r.rubric_name, avg_score DESC
    """).fetchall()
    return [
        {"model_name": r[0], "rubric_name": r[1], "avg_score": r[2], "run_count": r[3]}
        for r in rows
    ]


def get_all_run_ids(conn: sqlite3.Connection) -> list[str]:
    """Return all run_ids (for listing runs)."""
    return [r[0] for r in conn.execute("SELECT run_id FROM runs ORDER BY timestamp DESC").fetchall()]
