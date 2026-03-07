"""Pydantic data contracts for rubrics, judge output, reports, and cost."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Rubric ---


class Criterion(BaseModel):
    """Single evaluation criterion with score level descriptions."""

    name: str
    description: str
    score_descriptions: dict[int, str] = Field(
        ...,
        description="Score level descriptions; keys 1, 3, 5 required",
    )


class Rubric(BaseModel):
    """Evaluation rubric: name and list of criteria."""

    name: str
    criteria: list[Criterion]


# --- Judge output ---


class JudgeResult(BaseModel):
    """Structured result from the judge LLM: per-criterion scores and reasoning."""

    scores: dict[str, int] = Field(..., description="Criterion name -> score 1-5")
    reasoning: dict[str, str] = Field(..., description="Criterion name -> justification")
    overall_score: float = Field(..., description="Aggregate score")
    failed: bool = Field(default=False, description="True if parsing failed after retries")


# --- Cost ---


class CallCost(BaseModel):
    """Per-call token and cost metadata from one API call."""

    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost: float = 0.0


class CostMetadata(BaseModel):
    """Run-level aggregated cost metadata."""

    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_estimated_cost: float = 0.0


# --- Evaluation run ---


class EvaluationItemResult(BaseModel):
    """Result for one (question, response) item."""

    item_id: str
    question: str
    response: str
    response_length_tokens: int = 0
    scores: dict[str, int] = Field(default_factory=dict)
    reasoning: dict[str, str] = Field(default_factory=dict)
    overall_score: float = 0.0
    failed: bool = False


class EvaluationRunReport(BaseModel):
    """Full report for an evaluation run; always includes bias and cost metrics."""

    run_id: str
    model_name: str
    rubric_name: str
    items: list[EvaluationItemResult] = Field(default_factory=list)
    position_bias_rate: float = 0.0
    verbosity_correlation: float = 0.0
    cohens_kappa: float | None = None
    cost_metadata: CostMetadata = Field(default_factory=CostMetadata)
    timestamp: str = ""

    model_config = {"arbitrary_types_allowed": False}

    def model_dump_json(self, **kwargs: Any) -> str:
        return super().model_dump_json(**kwargs)
