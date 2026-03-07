# LLM Evaluation Harness (LLM-as-Judge)

An evaluation framework where an LLM judges the outputs of other LLMs, with bias metrics and cost tracking.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest tests/ -v
```

No `ANTHROPIC_API_KEY` required; all LLM calls are mocked in tests.

## Start API and dashboard

```bash
uvicorn src.llm_eval.api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 for the dashboard.
