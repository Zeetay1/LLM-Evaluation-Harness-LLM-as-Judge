"""Phase 3: Batch evaluator, cost aggregation, report JSON, no overwrite."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_eval.contracts import CostMetadata, EvaluationRunReport
from llm_eval.batch.evaluator import run_batch_evaluation
from llm_eval.metrics.cost_tracking import call_cost, aggregate_costs
from llm_eval.metrics.cohen_kappa import cohens_kappa
from llm_eval.rubrics.definitions import ACCURACY_RUBRIC
from tests.conftest import make_mock_response, valid_judge_json


def test_batch_processes_all_items_and_stores():
    """Batch with mock judge: all items processed, stored in SQLite; report has run_id, items, bias, cost_metadata."""
    client = MagicMock()
    client.messages.create.return_value = make_mock_response(
        valid_judge_json(ACCURACY_RUBRIC),
        input_tokens=100,
        output_tokens=50,
    )
    items = [("Q1?", "R1."), ("Q2?", "R2.")]
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        report = run_batch_evaluation(
            items,
            "test_model",
            ACCURACY_RUBRIC,
            db_path,
            client=client,
        )
        assert report.run_id
        assert len(report.items) == 2
        assert report.position_bias_rate == 0.0
        assert report.verbosity_correlation is not None
        assert report.cost_metadata.total_tokens_in > 0
        assert report.cost_metadata.total_estimated_cost > 0
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT run_id, total_tokens_in, total_estimated_cost FROM runs WHERE run_id = ?", (report.run_id,)).fetchone()
        conn.close()
        assert row is not None
        assert row[1] == report.cost_metadata.total_tokens_in
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_cost_aggregation_to_run_total():
    """Per-item costs aggregate correctly to run total."""
    costs = [
        call_cost(100, 50, "claude-3-5-sonnet-20241022"),
        call_cost(200, 100, "claude-3-5-sonnet-20241022"),
    ]
    meta = aggregate_costs(costs)
    assert meta.total_tokens_in == 300
    assert meta.total_tokens_out == 150
    assert meta.total_estimated_cost == sum(c.estimated_cost for c in costs)


def test_cohens_kappa_hand_computed():
    """Cohen's kappa matches hand-computed expected for a small list."""
    judge = [1, 2, 3, 4, 5]
    human = [1, 2, 3, 4, 5]
    k = cohens_kappa(judge, human)
    assert k == pytest.approx(1.0, abs=0.01)


def test_report_round_trip_json():
    """EvaluationRunReport serializes to JSON and deserializes with all fields including cost_metadata."""
    report = EvaluationRunReport(
        run_id="r1",
        model_name="m",
        rubric_name="accuracy",
        items=[],
        position_bias_rate=0.1,
        verbosity_correlation=0.2,
        cohens_kappa=0.5,
        cost_metadata=CostMetadata(total_tokens_in=100, total_tokens_out=50, total_estimated_cost=0.001),
        timestamp="2024-01-01T00:00:00Z",
    )
    js = report.model_dump_json()
    data = json.loads(js)
    assert data["run_id"] == "r1"
    assert data["cost_metadata"]["total_tokens_in"] == 100
    assert data["cost_metadata"]["total_estimated_cost"] == 0.001
    report2 = EvaluationRunReport.model_validate(data)
    assert report2.cost_metadata.total_estimated_cost == report.cost_metadata.total_estimated_cost


def test_two_runs_distinct_run_ids():
    """Re-running same inputs produces new run_id; both present in DB."""
    client = MagicMock()
    client.messages.create.return_value = make_mock_response(
        valid_judge_json(ACCURACY_RUBRIC),
        input_tokens=10,
        output_tokens=5,
    )
    items = [("Q?", "R.")]
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        r1 = run_batch_evaluation(items, "m", ACCURACY_RUBRIC, db_path, client=client)
        r2 = run_batch_evaluation(items, "m", ACCURACY_RUBRIC, db_path, client=client)
        assert r1.run_id != r2.run_id
        import sqlite3
        conn = sqlite3.connect(db_path)
        ids = [row[0] for row in conn.execute("SELECT run_id FROM runs").fetchall()]
        conn.close()
        assert r1.run_id in ids and r2.run_id in ids
    finally:
        Path(db_path).unlink(missing_ok=True)
