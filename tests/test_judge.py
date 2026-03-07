"""Phase 2: Judge executor, parsing, retries, cost, bias."""

import json
from unittest.mock import MagicMock

import pytest

from llm_eval.config import get_model_pricing, DEFAULT_JUDGE_MODEL
from llm_eval.contracts import CallCost, JudgeResult
from llm_eval.judge.executor import execute_judge, _parse_json, _retry_prompt, _dict_to_judge_result
from llm_eval.judge.bias import verbosity_correlation, count_tokens_simple, position_bias_rate
from llm_eval.metrics.cost_tracking import estimate_cost, call_cost, aggregate_costs
from llm_eval.rubrics.definitions import ACCURACY_RUBRIC
from tests.conftest import make_mock_response, valid_judge_json


def test_executor_returns_typed_judge_result():
    """Executor returns JudgeResult (Pydantic), not raw JSON."""
    client = MagicMock()
    client.messages.create.return_value = make_mock_response(
        valid_judge_json(ACCURACY_RUBRIC),
        input_tokens=10,
        output_tokens=20,
    )
    result, cost = execute_judge("Q?", "R.", ACCURACY_RUBRIC, client=client)
    assert isinstance(result, JudgeResult)
    assert result.scores
    assert result.reasoning
    assert result.overall_score == 3.5
    assert result.failed is False


def test_executor_retry_prompt_includes_exact_parse_error():
    """When mock returns invalid JSON, retry prompt includes the exact parse error."""
    client = MagicMock()
    client.messages.create.side_effect = [
        make_mock_response("not json at all", 10, 5),
        make_mock_response(valid_judge_json(ACCURACY_RUBRIC), 10, 20),
    ]
    result, _ = execute_judge("Q?", "R.", ACCURACY_RUBRIC, client=client, max_retries=2)
    assert result.failed is False
    calls = client.messages.create.call_args_list
    assert len(calls) >= 2
    retry_content = calls[1][1]["messages"][0]["content"]
    assert "Parse error" in retry_content or "parse" in retry_content.lower()
    assert "not json" in retry_content or "Expecting" in retry_content or "error" in retry_content.lower()


def test_executor_marks_failed_after_n_retries():
    """After N failures the item is marked failed, not retried indefinitely."""
    client = MagicMock()
    client.messages.create.return_value = make_mock_response("{ invalid }", 10, 5)
    result, cost = execute_judge("Q?", "R.", ACCURACY_RUBRIC, client=client, max_retries=2)
    assert result.failed is True
    assert result.overall_score == 0.0
    assert client.messages.create.call_count == 3  # initial + 2 retries


def test_cost_metadata_present_and_non_zero_after_mock_call():
    """Mock returns usage -> executor returns cost metadata; non-zero when mock supplies usage."""
    client = MagicMock()
    client.messages.create.return_value = make_mock_response(
        valid_judge_json(ACCURACY_RUBRIC),
        input_tokens=1000,
        output_tokens=500,
    )
    result, cost = execute_judge("Q?", "R.", ACCURACY_RUBRIC, client=client)
    assert isinstance(cost, CallCost)
    assert cost.tokens_in == 1000
    assert cost.tokens_out == 500
    assert cost.estimated_cost > 0


def test_estimated_cost_matches_config_pricing():
    """estimated_cost matches config: tokens_in * price_in + tokens_out * price_out."""
    price_in, price_out = get_model_pricing(DEFAULT_JUDGE_MODEL)
    expected = (1000 / 1000.0) * price_in + (500 / 1000.0) * price_out
    assert estimate_cost(1000, 500, DEFAULT_JUDGE_MODEL) == pytest.approx(expected)
    c = call_cost(1000, 500, DEFAULT_JUDGE_MODEL)
    assert c.estimated_cost == pytest.approx(expected)


def test_position_bias_rate_known_flip():
    """Hand-built set where winner flips -> position bias rate as expected."""
    def judge_ab(q: str, a: str, b: str) -> str:
        return "A" if (a, b) == ("short", "long") else "B"
    # When (resp_a, resp_b) = (short, long) -> A; when (long, short) -> B, so flip.
    pairs = [
        ("q1", "short", "long"),
        ("q2", "short", "long"),
    ]
    # Our run_comparative_judge takes (question, resp_a, resp_b) and returns winner for that order.
    # So we need a callable: (q, a, b) -> winner. If we return A for (short, long) and B for (long, short), we get flip.
    def run_judge(q: str, a: str, b: str) -> str:
        return "A" if (a, b) == ("short", "long") else "B"
    rate = position_bias_rate(pairs, run_judge)
    assert rate == 1.0  # both pairs flip


def test_position_bias_rate_no_flip():
    """When winner never flips, rate is 0."""
    def run_judge(q: str, a: str, b: str) -> str:
        return "A"  # always A
    pairs = [("q1", "x", "y"), ("q2", "a", "b")]
    assert position_bias_rate(pairs, run_judge) == 0.0


def test_verbosity_correlation_known():
    """Hand-built lengths and scores -> Pearson matches expected."""
    lengths = [10, 20, 30, 40, 50]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = verbosity_correlation(lengths, scores)
    assert r == pytest.approx(1.0)


def test_verbosity_correlation_negative():
    """Longer -> lower score gives negative correlation."""
    lengths = [10, 20, 30]
    scores = [5.0, 3.0, 1.0]
    r = verbosity_correlation(lengths, scores)
    assert r == pytest.approx(-1.0)


def test_count_tokens_simple():
    """Simple token count is deterministic."""
    assert count_tokens_simple("one two three") == 3
    assert count_tokens_simple("") == 0


def test_parse_json_extracts_from_markdown():
    """JSON can be extracted from markdown code fence."""
    data = _parse_json('```json\n{"scores": {"x": 1}, "reasoning": {"x": "y"}, "overall_score": 2.0}\n```')
    assert data["scores"]["x"] == 1
    assert data["overall_score"] == 2.0


def test_retry_prompt_references_error():
    """Retry prompt includes the exact parse error string."""
    prompt = _retry_prompt("Original", "Previous response", "Expecting value: line 1 column 1")
    assert "Expecting value" in prompt
    assert "line 1 column 1" in prompt


def test_aggregate_costs():
    """Per-item costs aggregate to run total."""
    costs = [
        CallCost(tokens_in=100, tokens_out=50, estimated_cost=0.001),
        CallCost(tokens_in=200, tokens_out=100, estimated_cost=0.002),
    ]
    meta = aggregate_costs(costs)
    assert meta.total_tokens_in == 300
    assert meta.total_tokens_out == 150
    assert meta.total_estimated_cost == pytest.approx(0.003)
