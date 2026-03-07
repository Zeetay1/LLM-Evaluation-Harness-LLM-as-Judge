"""Judge executor: call LLM, parse JSON, retry on failure with exact error."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_eval.config import DEFAULT_JUDGE_MODEL
from llm_eval.contracts import CallCost, JudgeResult, Rubric
from llm_eval.metrics.cost_tracking import call_cost
from llm_eval.rubrics.prompt import build_judge_prompt, build_comparative_judge_prompt


# Default max retries before marking item failed
DEFAULT_MAX_RETRIES = 2


def execute_judge(
    question: str,
    response: str,
    rubric: Rubric,
    *,
    client: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[JudgeResult, CallCost]:
    """
    Run the judge on (question, response). Returns (JudgeResult, CallCost).
    On parse failure, retries with exact error in prompt; after max_retries returns failed result.
    """
    if client is None:
        from anthropic import Anthropic
        client = Anthropic()
    prompt = build_judge_prompt(rubric, question, response)
    return _call_and_parse(client, model, prompt, rubric, max_retries)


def execute_comparative_judge(
    question: str,
    response_a: str,
    response_b: str,
    rubric: Rubric,
    *,
    client: Any | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[dict[str, Any], CallCost]:
    """
    Run comparative judge (A vs B). Returns (dict with 'winner': 'A'|'B', 'scores_a', 'scores_b', 'reasoning'), CallCost).
    On parse failure, retries with exact error; after max_retries returns failed result (winner empty or arbitrary).
    """
    if client is None:
        from anthropic import Anthropic
        client = Anthropic()
    prompt = build_comparative_judge_prompt(rubric, question, response_a, response_b)
    return _call_and_parse_comparative(client, model, prompt, max_retries)


def _call_and_parse(
    client: Any,
    model: str,
    prompt: str,
    rubric: Rubric,
    max_retries: int,
) -> tuple[JudgeResult, CallCost]:
    last_cost = CallCost()
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        text = ""
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in = getattr(usage, "input_tokens", 0) or 0
                tokens_out = getattr(usage, "output_tokens", 0) or 0
            else:
                tokens_in = tokens_out = 0
            last_cost = call_cost(tokens_in, tokens_out, model)
            text = _extract_text(resp)
            data = _parse_json(text)
            result = _dict_to_judge_result(data, rubric)
            return (result, last_cost)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                prompt = _retry_prompt(prompt, text, last_error)
                continue
            return (
                JudgeResult(
                    scores={c.name: 0 for c in rubric.criteria},
                    reasoning={c.name: f"Parse failed: {last_error}" for c in rubric.criteria},
                    overall_score=0.0,
                    failed=True,
                ),
                last_cost,
            )
    return (
        JudgeResult(
            scores={c.name: 0 for c in rubric.criteria},
            reasoning={c.name: f"Parse failed: {last_error}" for c in rubric.criteria},
            overall_score=0.0,
            failed=True,
        ),
        last_cost,
    )


def _call_and_parse_comparative(
    client: Any,
    model: str,
    prompt: str,
    max_retries: int,
) -> tuple[dict[str, Any], CallCost]:
    last_cost = CallCost()
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        text = ""
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(resp, "usage", None)
            if usage:
                tokens_in = getattr(usage, "input_tokens", 0) or 0
                tokens_out = getattr(usage, "output_tokens", 0) or 0
            else:
                tokens_in = tokens_out = 0
            last_cost = call_cost(tokens_in, tokens_out, model)
            text = _extract_text(resp)
            data = _parse_json(text)
            winner = data.get("winner") or ""
            if winner.upper() not in ("A", "B"):
                raise ValueError(f"Invalid winner: {winner!r}")
            return (data, last_cost)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                prompt = _retry_prompt(prompt, text, last_error)
                continue
            return ({"winner": "", "reasoning": f"Parse failed: {last_error}", "failed": True}, last_cost)
    return ({"winner": "", "reasoning": f"Parse failed: {last_error}", "failed": True}, last_cost)


def _extract_text(resp: Any) -> str:
    """Extract assistant message text from Anthropic response."""
    content = getattr(resp, "content", None) or []
    if isinstance(content, str):
        return content
    for block in content:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text (strip markdown code fence if present)."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _retry_prompt(original_prompt: str, previous_response: str, parse_error: str) -> str:
    """Build retry prompt that references the exact parse error."""
    return f"""{original_prompt}

---

Your previous response failed to parse as JSON. Parse error: {parse_error}

Please respond with valid JSON only, using the exact schema specified. No markdown, no prose."""


def _dict_to_judge_result(data: dict[str, Any], rubric: Rubric) -> JudgeResult:
    """Convert parsed dict to JudgeResult; ensure all criteria present."""
    scores = data.get("scores") or {}
    reasoning = data.get("reasoning") or {}
    overall = data.get("overall_score")
    if overall is None:
        criterion_scores = [scores.get(c.name, 0) for c in rubric.criteria]
        overall = sum(criterion_scores) / len(rubric.criteria) if criterion_scores else 0.0
    else:
        overall = float(overall)
    for c in rubric.criteria:
        if c.name not in scores:
            scores[c.name] = 0
        if c.name not in reasoning:
            reasoning[c.name] = ""
    return JudgeResult(scores=scores, reasoning=reasoning, overall_score=overall, failed=False)
