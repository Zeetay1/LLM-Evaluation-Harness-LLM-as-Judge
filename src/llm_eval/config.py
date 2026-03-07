"""Model pricing constants for cost estimation (per-token input/output)."""

# Anthropic Claude pricing (USD per 1K tokens) - update as needed from published pricing
# Example: Claude 3.5 Sonnet as of 2024
DEFAULT_JUDGE_MODEL = "claude-3-5-sonnet-20241022"

# Per 1K input tokens, per 1K output tokens (USD)
MODEL_PRICE_INPUT_PER_1K: dict[str, float] = {
    "claude-3-5-sonnet-20241022": 0.003,
    "claude-3-5-sonnet-latest": 0.003,
}
MODEL_PRICE_OUTPUT_PER_1K: dict[str, float] = {
    "claude-3-5-sonnet-20241022": 0.015,
    "claude-3-5-sonnet-latest": 0.015,
}


def get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_per_1k, output_per_1k) for the model. Defaults to 0 if unknown."""
    return (
        MODEL_PRICE_INPUT_PER_1K.get(model, 0.0),
        MODEL_PRICE_OUTPUT_PER_1K.get(model, 0.0),
    )
