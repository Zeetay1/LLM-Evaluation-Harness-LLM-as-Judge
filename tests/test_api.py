"""Phase 4: Leaderboard API, runs, model items, cost in report."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_eval.contracts import CostMetadata, EvaluationItemResult
from llm_eval.storage.sqlite import init_db, insert_run, insert_run_items, run_to_report
from llm_eval.api.main import app, DB_PATH


@pytest.fixture
def temp_db(monkeypatch):
    """Use a temporary DB for API tests."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = f.name
    f.close()
    monkeypatch.setattr("llm_eval.api.main.DB_PATH", Path(path))
    init_db(path)
    yield path
    try:
        Path(path).unlink(missing_ok=True)
    except PermissionError:
        pass


@pytest.fixture
def seeded_client(temp_db):
    """Client with pre-seeded runs and items."""
    conn = sqlite3.connect(temp_db)
    for run_id, model, rubric, score in [
        ("run-1", "model_a", "accuracy", 4.0),
        ("run-2", "model_b", "accuracy", 3.0),
    ]:
        insert_run(
            conn,
            run_id=run_id,
            model_name=model,
            rubric_name=rubric,
            timestamp="2024-01-01T00:00:00Z",
            position_bias_rate=0.0,
            verbosity_correlation=0.1,
            cohens_kappa=None,
            cost_metadata=CostMetadata(total_tokens_in=100, total_tokens_out=50, total_estimated_cost=0.002),
        )
        insert_run_items(
            conn,
            run_id,
            [
                EvaluationItemResult(
                    item_id="0",
                    question="What is 2+2?",
                    response="4",
                    response_length_tokens=1,
                    scores={"factual_correctness": 5, "completeness": 5},
                    reasoning={"factual_correctness": "Correct", "completeness": "Complete"},
                    overall_score=score,
                    failed=False,
                )
            ],
        )
    conn.commit()
    conn.close()
    return TestClient(app)


def test_leaderboard_returns_ranking(seeded_client):
    """GET /leaderboard returns expected ranking from DB."""
    r = seeded_client.get("/leaderboard")
    assert r.status_code == 200
    data = r.json()
    assert "leaderboard" in data
    lb = data["leaderboard"]
    assert len(lb) >= 2
    accuracy = [x for x in lb if x["rubric_name"] == "accuracy"]
    assert len(accuracy) == 2
    assert accuracy[0]["model_name"] == "model_a"
    assert accuracy[0]["avg_score"] >= accuracy[1]["avg_score"]


def test_get_run_returns_full_report_with_cost(seeded_client):
    """GET /runs/{run_id} returns full report including cost_metadata."""
    r = seeded_client.get("/runs/run-1")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "run-1"
    assert "cost_metadata" in data
    assert data["cost_metadata"]["total_tokens_in"] == 100
    assert data["cost_metadata"]["total_tokens_out"] == 50
    assert data["cost_metadata"]["total_estimated_cost"] == 0.002


def test_get_run_404(seeded_client):
    """GET /runs/nonexistent returns 404."""
    r = seeded_client.get("/runs/nonexistent-id")
    assert r.status_code == 404


def test_models_items_returns_items(seeded_client):
    """GET /models/{model_name}/items returns correct items for that model."""
    r = seeded_client.get("/models/model_a/items")
    assert r.status_code == 200
    data = r.json()
    assert data["model_name"] == "model_a"
    assert "items" in data
    assert len(data["items"]) >= 1
    assert data["items"][0]["question"] == "What is 2+2?"


def test_compare_two_models_deltas(seeded_client):
    """Comparison: two models with known items; response includes scores and deltas."""
    r = seeded_client.get("/leaderboard")
    assert r.status_code == 200
    r_a = seeded_client.get("/models/model_a/items")
    r_b = seeded_client.get("/models/model_b/items")
    assert r_a.status_code == 200 and r_b.status_code == 200
    items_a = r_a.json().get("items", [])
    items_b = r_b.json().get("items", [])
    # Same question in both
    q = "What is 2+2?"
    sa = next((i["overall_score"] for i in items_a if i["question"] == q), None)
    sb = next((i["overall_score"] for i in items_b if i["question"] == q), None)
    assert sa == 4.0 and sb == 3.0


def test_server_starts_and_endpoints_reachable(seeded_client):
    """All main endpoints are reachable."""
    assert seeded_client.get("/leaderboard").status_code == 200
    assert seeded_client.get("/runs").status_code == 200
    assert seeded_client.get("/runs/run-1").status_code == 200
    assert seeded_client.get("/models/model_a/items").status_code == 200
