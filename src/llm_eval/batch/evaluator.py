"""Batch evaluator: process items, aggregate cost and bias, persist to SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_eval.contracts import (
    CostMetadata,
    EvaluationItemResult,
    EvaluationRunReport,
    Rubric,
)
from llm_eval.judge.executor import execute_judge, execute_comparative_judge
from llm_eval.judge.bias import count_tokens_simple, verbosity_correlation, position_bias_rate
from llm_eval.metrics.cost_tracking import aggregate_costs, CallCost
from llm_eval.storage.sqlite import init_db, insert_run, insert_run_items


def run_batch_evaluation(
    items: list[tuple[str, str]],
    model_name: str,
    rubric: Rubric,
    db_path: str | Path,
    *,
    client: Any | None = None,
    position_bias_pairs: list[tuple[str, str, str]] | None = None,
    calibration_path: str | Path | None = None,
) -> EvaluationRunReport:
    """
    Run judge on each (question, response), aggregate costs and bias, persist to SQLite.
    Returns EvaluationRunReport with run_id, items, position_bias_rate, verbosity_correlation,
    cost_metadata, and cohens_kappa (if calibration_path provided).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    call_costs: list[CallCost] = []
    results: list[EvaluationItemResult] = []
    for i, (question, response) in enumerate(items):
        result, cost = execute_judge(question, response, rubric, client=client)
        call_costs.append(cost)
        length = count_tokens_simple(response)
        results.append(
            EvaluationItemResult(
                item_id=str(i),
                question=question,
                response=response,
                response_length_tokens=length,
                scores=result.scores,
                reasoning=result.reasoning,
                overall_score=result.overall_score,
                failed=result.failed,
            )
        )

    cost_metadata = aggregate_costs(call_costs)
    lengths = [r.response_length_tokens for r in results]
    scores = [r.overall_score for r in results]
    verb_corr = verbosity_correlation(lengths, scores)

    position_bias = 0.0
    if position_bias_pairs:
        def run_comp(q: str, a: str, b: str) -> str:
            data, cost = execute_comparative_judge(q, a, b, rubric, client=client)
            call_costs.append(cost)
            w = (data.get("winner") or "").upper()
            return "A" if w == "A" else "B"
        position_bias = position_bias_rate(position_bias_pairs, run_comp)
        cost_metadata = aggregate_costs(call_costs)

    cohens_kappa: float | None = None
    if calibration_path and Path(calibration_path).exists():
        cal_data = _load_calibration(calibration_path)
        if cal_data:
            judge_scores_list: list[float] = []
            human_scores_list: list[float] = []
            for item in cal_data:
                q = item.get("question", "")
                r = item.get("response", "")
                human_overall = item.get("human_overall")
                if human_overall is None:
                    continue
                j_result, c_cost = execute_judge(q, r, rubric, client=client)
                call_costs.append(c_cost)
                judge_scores_list.append(j_result.overall_score)
                human_scores_list.append(float(human_overall))
            cost_metadata = aggregate_costs(call_costs)
            if len(judge_scores_list) == len(human_scores_list) and len(judge_scores_list) > 0:
                from llm_eval.metrics.cohen_kappa import cohens_kappa as compute_kappa
                cohens_kappa = compute_kappa(judge_scores_list, human_scores_list)

    conn = sqlite3.connect(str(db_path))
    try:
        insert_run(
            conn,
            run_id=run_id,
            model_name=model_name,
            rubric_name=rubric.name,
            timestamp=timestamp,
            position_bias_rate=position_bias,
            verbosity_correlation=verb_corr,
            cohens_kappa=cohens_kappa,
            cost_metadata=cost_metadata,
        )
        insert_run_items(conn, run_id, results)
        conn.commit()
    finally:
        conn.close()

    return EvaluationRunReport(
        run_id=run_id,
        model_name=model_name,
        rubric_name=rubric.name,
        items=results,
        position_bias_rate=position_bias,
        verbosity_correlation=verb_corr,
        cohens_kappa=cohens_kappa,
        cost_metadata=cost_metadata,
        timestamp=timestamp,
    )


def _load_calibration(path: str | Path) -> list[dict[str, Any]] | None:
    """Load calibration JSON; return list of items with human_overall and human_scores."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except Exception:
        return None
