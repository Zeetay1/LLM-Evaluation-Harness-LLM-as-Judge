"""Phase 3: SQLite schema, leaderboard from history, cost persistence."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from llm_eval.contracts import CostMetadata, EvaluationItemResult, EvaluationRunReport
from llm_eval.storage.sqlite import (
    init_db,
    insert_run,
    insert_run_items,
    get_run,
    get_run_items,
    get_leaderboard,
    run_to_report,
)


def test_init_db_creates_tables():
    """Schema creates runs and run_items tables."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        init_db(path)
        conn = sqlite3.connect(path)
        r = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('runs', 'run_items')"
        ).fetchall()
        conn.close()
        assert len(r) == 2
    finally:
        Path(path).unlink(missing_ok=True)


def test_cost_fields_persisted_and_retrievable():
    """Run row includes total_tokens_in, total_tokens_out, total_estimated_cost; retrieve matches."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        init_db(path)
        conn = sqlite3.connect(path)
        cost = CostMetadata(total_tokens_in=1000, total_tokens_out=500, total_estimated_cost=0.012)
        insert_run(
            conn,
            run_id="run-1",
            model_name="m1",
            rubric_name="accuracy",
            timestamp="2024-01-01T00:00:00Z",
            position_bias_rate=0.0,
            verbosity_correlation=0.1,
            cohens_kappa=None,
            cost_metadata=cost,
        )
        insert_run_items(
            conn,
            "run-1",
            [
                EvaluationItemResult(
                    item_id="0",
                    question="Q?",
                    response="R.",
                    response_length_tokens=5,
                    scores={"x": 3},
                    reasoning={"x": "ok"},
                    overall_score=3.0,
                    failed=False,
                )
            ],
        )
        conn.commit()
        run = get_run(conn, "run-1")
        conn.close()
        assert run is not None
        assert run["total_tokens_in"] == 1000
        assert run["total_tokens_out"] == 500
        assert run["total_estimated_cost"] == pytest.approx(0.012)
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except PermissionError:
            pass


def test_leaderboard_from_history():
    """Leaderboard is built from SQLite; insert known results and verify ranking."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        init_db(path)
        conn = sqlite3.connect(path)
        for run_id, model, rubric, score in [
            ("r1", "model_a", "accuracy", 4.0),
            ("r2", "model_b", "accuracy", 3.0),
            ("r3", "model_a", "helpfulness", 3.5),
        ]:
            insert_run(
                conn,
                run_id=run_id,
                model_name=model,
                rubric_name=rubric,
                timestamp="2024-01-01T00:00:00Z",
                position_bias_rate=0.0,
                verbosity_correlation=0.0,
                cohens_kappa=None,
                cost_metadata=CostMetadata(total_tokens_in=0, total_tokens_out=0, total_estimated_cost=0.0),
            )
            insert_run_items(
                conn,
                run_id,
                [
                    EvaluationItemResult(
                        item_id="0",
                        question="Q",
                        response="R",
                        response_length_tokens=0,
                        scores={},
                        reasoning={},
                        overall_score=score,
                        failed=False,
                    )
                ],
            )
        conn.commit()
        lb = get_leaderboard(conn)
        conn.close()
        assert len(lb) >= 2
        accuracy_rows = [r for r in lb if r["rubric_name"] == "accuracy"]
        assert len(accuracy_rows) == 2
        assert accuracy_rows[0]["avg_score"] >= accuracy_rows[1]["avg_score"]
        assert accuracy_rows[0]["model_name"] == "model_a"
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except PermissionError:
            pass


def test_run_to_report_includes_cost():
    """run_to_report returns EvaluationRunReport with cost_metadata."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        init_db(path)
        conn = sqlite3.connect(path)
        insert_run(
            conn,
            run_id="run-1",
            model_name="m",
            rubric_name="accuracy",
            timestamp="2024-01-01T00:00:00Z",
            position_bias_rate=0.0,
            verbosity_correlation=0.0,
            cohens_kappa=None,
            cost_metadata=CostMetadata(total_tokens_in=100, total_tokens_out=50, total_estimated_cost=0.001),
        )
        insert_run_items(conn, "run-1", [])
        conn.commit()
        report = run_to_report(conn, "run-1")
        conn.close()
        assert report is not None
        assert report.cost_metadata.total_tokens_in == 100
        assert report.cost_metadata.total_estimated_cost == pytest.approx(0.001)
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except PermissionError:
            pass
