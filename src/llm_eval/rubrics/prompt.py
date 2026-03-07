"""Prompt generator: converts any rubric into a consistent judge prompt with JSON output."""

from __future__ import annotations

from llm_eval.contracts import Rubric


def build_judge_prompt(rubric: Rubric, question: str, response: str) -> str:
    """
    Build a judge prompt for scoring a single response.
    Prompt includes rubric name, all criteria with 1/3/5 descriptions, and explicit JSON-only output instruction.
    """
    criteria_block = _format_criteria(rubric)
    schema_block = _format_schema(rubric)
    return f"""You are an expert evaluator. Score the following model response using the rubric below.

## Rubric: {rubric.name}

{criteria_block}

## Task

**Question:** {question}

**Model response to evaluate:** {response}

## Instructions

Score the response on each criterion using the scale above (1-5). Provide a brief justification for each score. Then provide an overall score (1-5) that reflects the aggregate.

You must respond with valid JSON only—no other text, no markdown code fences, no prose. Use exactly this structure:

{schema_block}

Respond with only the JSON object, nothing else."""


def build_comparative_judge_prompt(
    rubric: Rubric,
    question: str,
    response_a: str,
    response_b: str,
) -> str:
    """
    Build a judge prompt for comparing two responses (for position bias).
    Asks which response is better and requests JSON with winner and optional scores.
    """
    criteria_block = _format_criteria(rubric)
    return f"""You are an expert evaluator. Compare the following two model responses using the rubric below and decide which is better.

## Rubric: {rubric.name}

{criteria_block}

## Task

**Question:** {question}

**Response A:** {response_a}

**Response B:** {response_b}

## Instructions

Which response is better according to the rubric? Respond with valid JSON only—no other text, no markdown, no prose. Use exactly this structure:

{{"winner": "A" or "B", "scores_a": {{"<criterion_name>": 1-5, ...}}, "scores_b": {{"<criterion_name>": 1-5, ...}}, "reasoning": "Brief explanation"}}

Respond with only the JSON object, nothing else."""


def _format_criteria(rubric: Rubric) -> str:
    """Format rubric criteria with score descriptions for 1, 3, 5."""
    parts = []
    for c in rubric.criteria:
        desc = c.description
        levels = c.score_descriptions
        one = levels.get(1, "Low")
        three = levels.get(3, "Medium")
        five = levels.get(5, "High")
        parts.append(
            f"- **{c.name}**: {desc}\n  - 1: {one}\n  - 3: {three}\n  - 5: {five}"
        )
    return "\n\n".join(parts)


def _format_schema(rubric: Rubric) -> str:
    """Format the required JSON schema with criterion names."""
    names = [c.name for c in rubric.criteria]
    scores_ex = ", ".join(f'"{n}": 1|2|3|4|5' for n in names)
    reasoning_ex = ", ".join(f'"{n}": "..."' for n in names)
    return f'{{"scores": {{{scores_ex}}}, "reasoning": {{{reasoning_ex}}}, "overall_score": number}}'
