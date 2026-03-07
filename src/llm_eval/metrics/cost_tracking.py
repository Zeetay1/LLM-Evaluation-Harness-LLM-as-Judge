"""Per-call token/cost logging and run-level aggregation."""

from __future__ import annotations

from llm_eval.config import get_model_pricing
from llm_eval.contracts import CallCost, CostMetadata


def estimate_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    """Compute estimated cost (USD) from token counts using config pricing."""
    price_in, price_out = get_model_pricing(model)
    return (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out


def call_cost(tokens_in: int, tokens_out: int, model: str) -> CallCost:
    """Build a CallCost for one API call."""
    return CallCost(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        estimated_cost=estimate_cost(tokens_in, tokens_out, model),
    )


def aggregate_costs(costs: list[CallCost]) -> CostMetadata:
    """Aggregate per-call costs into run-level CostMetadata."""
    total_in = sum(c.tokens_in for c in costs)
    total_out = sum(c.tokens_out for c in costs)
    total_cost = sum(c.estimated_cost for c in costs)
    return CostMetadata(
        total_tokens_in=total_in,
        total_tokens_out=total_out,
        total_estimated_cost=total_cost,
    )
