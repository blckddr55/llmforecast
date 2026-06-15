# llmforecast — Bayesian Linguistic Forecaster

A web-grounded agent that estimates the probability of a forecasting question by
reasoning like a superforecaster: it starts from a base rate and performs
explicit Bayesian updates as it gathers evidence from the web.

It uses **Google Gemini** (function calling) for reasoning and **Tavily** for
web search.

## How it works

At every step the model is *forced* to call a single function,
`update_belief_and_act`, which records:

- `probability` — the current posterior, calibrated to `[0.05, 0.95]`
- `confidence` — `low` / `medium` / `high`
- `evidence_for` / `evidence_against` — concrete supporting / contradicting evidence
- `update_reasoning` — the Bayesian update just performed
- `action` — `web_search` or `submit`
- `action_input` — a search query, or a final justification

When the action is `web_search`, the agent runs Tavily, feeds the results back as
a function response, and updates again — up to `MAX_STEPS` (10). When it
`submit`s (or the step budget runs out), the run returns its final probability.

To reduce variance, each question is forecast over `NUM_TRIALS` (5) independent
runs, and the results are combined with a **logit-space mean** — averaging in
log-odds space, which is symmetric around 0.5 and treats evidence additively
rather than averaging raw probabilities.

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)
- A Google Gemini API key and a Tavily API key

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` from the template and add your keys (`.env` is git-ignored):

```bash
cp .env.example .env
# then edit .env:
#   GEMINI_API_KEY=...
#   TAVILY_API_KEY=...
```

## Usage

```bash
uv run forecaster.py
```

This forecasts the example question in `main()`. To forecast your own, edit that
function or import the API directly:

```python
from forecaster import aggregate_forecasts

p = aggregate_forecasts("Will event X happen before date Y?")
print(p)
```

### Logging

Progress is logged via the standard `logging` module. The default `INFO` level
shows each step's probability / action / reasoning, every search query and
result summary, and per-trial timing. For full detail (evidence lists and
per-source hits):

```bash
LOG_LEVEL=DEBUG uv run forecaster.py
```

## Configuration

The knobs live at the top of `forecaster.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `gemini-2.5-flash` | Gemini model |
| `MAX_STEPS` | `10` | Max agent steps per run |
| `NUM_TRIALS` | `5` | Independent runs aggregated per question |
| `MAX_OUTPUT_TOKENS` | `4096` | Output token cap per call |
| `TEMPERATURE` | `1.0` | Sampling temperature (so trials diverge) |
| `THINKING_BUDGET` | `0` | Gemini thinking budget; `0` disables it (use `-1` or a positive value for `*-pro` models) |
| `TAVILY_MAX_RESULTS` | `5` | Results per search |
