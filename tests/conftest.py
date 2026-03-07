"""Shared fixtures: mock Anthropic client, sample rubrics/items."""

import json
from unittest.mock import MagicMock

import pytest

from llm_eval.contracts import Rubric
from llm_eval.rubrics.definitions import ACCURACY_RUBRIC


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client that returns configurable content and usage."""
    client = MagicMock()
    return client


def make_mock_response(
    content: str,
    input_tokens: int = 100,
    output_tokens: int = 50,
):
    """Build a mock Message-like response with content and usage."""
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text=content)]
    msg.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return msg


def valid_judge_json(rubric: Rubric, overall: float = 3.5) -> str:
    """Produce valid judge JSON for a rubric."""
    scores = {c.name: 3 for c in rubric.criteria}
    reasoning = {c.name: "OK" for c in rubric.criteria}
    return json.dumps({"scores": scores, "reasoning": reasoning, "overall_score": overall})
