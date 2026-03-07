"""Phase 1: Prompt generator and built-in rubrics."""

import pytest

from llm_eval.contracts import Criterion, Rubric
from llm_eval.rubrics.definitions import (
    ACCURACY_RUBRIC,
    CONCISENESS_RUBRIC,
    HELPFULNESS_RUBRIC,
    SAFETY_RUBRIC,
    get_builtin_rubric,
    list_builtin_rubrics,
)
from llm_eval.rubrics.prompt import build_judge_prompt, build_comparative_judge_prompt


def test_builtin_rubrics_exist():
    """All four built-in rubrics are defined and have at least 2 criteria."""
    rubrics = [ACCURACY_RUBRIC, HELPFULNESS_RUBRIC, SAFETY_RUBRIC, CONCISENESS_RUBRIC]
    for r in rubrics:
        assert r.name
        assert len(r.criteria) >= 2
        for c in r.criteria:
            assert c.name
            assert 1 in c.score_descriptions
            assert 3 in c.score_descriptions
            assert 5 in c.score_descriptions


def test_prompt_contains_rubric_name_and_criteria():
    """Generated prompt includes rubric name, every criterion name, and score descriptions 1/3/5."""
    for rubric in [ACCURACY_RUBRIC, HELPFULNESS_RUBRIC, SAFETY_RUBRIC, CONCISENESS_RUBRIC]:
        prompt = build_judge_prompt(rubric, "What is 2+2?", "Four.")
        assert rubric.name in prompt
        for c in rubric.criteria:
            assert c.name in prompt
            assert "1:" in prompt or "- 1:" in prompt
            assert "3:" in prompt or "- 3:" in prompt
            assert "5:" in prompt or "- 5:" in prompt
            assert c.score_descriptions[1] in prompt
            assert c.score_descriptions[3] in prompt
            assert c.score_descriptions[5] in prompt


def test_prompt_contains_json_instruction():
    """Generated prompt explicitly instructs judge to return JSON only."""
    prompt = build_judge_prompt(ACCURACY_RUBRIC, "Q?", "R.")
    assert "JSON" in prompt
    assert "scores" in prompt
    assert "reasoning" in prompt
    assert "overall_score" in prompt
    assert "only" in prompt.lower() or "valid" in prompt.lower()


def test_fifth_rubric_works_without_changing_prompt_generator():
    """Adding a fifth rubric requires no changes to the prompt generator."""
    custom = Rubric(
        name="custom_rubric",
        criteria=[
            Criterion(
                name="criterion_one",
                description="First custom criterion.",
                score_descriptions={1: "Bad", 3: "OK", 5: "Good"},
            ),
            Criterion(
                name="criterion_two",
                description="Second custom criterion.",
                score_descriptions={1: "Low", 3: "Mid", 5: "High"},
            ),
        ],
    )
    prompt = build_judge_prompt(custom, "Question?", "Response.")
    assert "custom_rubric" in prompt
    assert "criterion_one" in prompt
    assert "criterion_two" in prompt
    assert "Bad" in prompt and "OK" in prompt and "Good" in prompt
    assert "Low" in prompt and "Mid" in prompt and "High" in prompt
    assert "JSON" in prompt and "scores" in prompt and "reasoning" in prompt


def test_comparative_prompt_includes_both_responses_and_winner():
    """Comparative prompt includes question, response A, response B, and winner schema."""
    prompt = build_comparative_judge_prompt(
        ACCURACY_RUBRIC, "Q?", "Answer A", "Answer B"
    )
    assert "Answer A" in prompt and "Answer B" in prompt
    assert "winner" in prompt
    assert "A" in prompt and "B" in prompt
    assert ACCURACY_RUBRIC.name in prompt


def test_list_and_get_builtin_rubrics():
    """list_builtin_rubrics and get_builtin_rubric work."""
    names = list_builtin_rubrics()
    assert set(names) == {"accuracy", "helpfulness", "safety", "conciseness"}
    assert get_builtin_rubric("accuracy") == ACCURACY_RUBRIC
    assert get_builtin_rubric("nonexistent") is None
