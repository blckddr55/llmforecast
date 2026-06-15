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
Prediction-market and betting sites are excluded from the search, and the model
is told not to treat their odds as evidence — so the forecast rests on primary
sources rather than echoing a market price.

To reduce variance, each question is forecast over `NUM_TRIALS` (5) independent
runs, and the results are combined with a **logit-space mean** — averaging in
log-odds space, which is symmetric around 0.5 and treats evidence additively
rather than averaging raw probabilities. A final model call then synthesizes the
trials into a short briefing: the headline probability and spread, the strongest
evidence for and against, and a bottom line.

### Prior injection (optional)

You can seed the agent with an external **prior anchor** — a market-implied
probability for market questions, or a historical base rate for dataset
questions. Pass `prior=<float in [0, 1]>` to `aggregate_forecasts` (or
`run_agent`) and the agent starts from that anchor and updates away from it only
as far as the evidence justifies. Omit it (the default) and the agent forms its
own base rate.

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
# Default example question, no prior:
uv run forecaster.py

# Inject a prior anchor (a market price or historical base rate):
uv run forecaster.py --prior 0.10

# Your own question, with a prior and a custom number of trials:
uv run forecaster.py "Will X happen before 2027?" --prior 0.62 --trials 3
```

The question is an optional positional argument; `--prior` and `--trials` are
optional flags. You can also call the API directly from Python:

```python
from forecaster import aggregate_forecasts

result = aggregate_forecasts("Will event X happen before date Y?")
print(result.probability)   # aggregated probability (logit-space mean)
print(result.summary)       # synthesized briefing of the argument

# Anchor on an external prior (e.g. a prediction-market price of 62%):
result = aggregate_forecasts("Will event X happen before date Y?", prior=0.62)
```

### Logging

Progress is logged via the standard `logging` module. The default `INFO` level
shows each step's probability / action / reasoning, every search query with its
result titles and links, and per-trial timing. For full detail (evidence lists):

```bash
LOG_LEVEL=DEBUG uv run forecaster.py
```

## Saved runs

Every run is written to `runs/` as a timestamped JSON record — the question, any
prior, the per-trial final beliefs (probability, evidence, reasoning), the
aggregated probability, and the synthesized briefing — so decisions aren't lost
when the process exits.

## Configuration

The knobs live at the top of `forecaster.py`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `gemini-3.5-flash` | Gemini model |
| `MAX_STEPS` | `10` | Max agent steps per run |
| `NUM_TRIALS` | `5` | Independent runs aggregated per question |
| `MAX_OUTPUT_TOKENS` | `8192` | Output token cap per call (headroom for thinking) |
| `TEMPERATURE` | `1.0` | Sampling temperature (so trials diverge) |
| `THINKING_LEVEL` | `"high"` | Gemini 3 thinking depth — `"low"` or `"high"` |
| `TAVILY_MAX_RESULTS` | `5` | Results per search |
