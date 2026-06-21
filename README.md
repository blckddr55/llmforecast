# llmforecast — Bayesian Linguistic Forecaster

A web-grounded agent that estimates the probability of a forecasting question by
reasoning like a superforecaster: it starts from a base rate and performs
explicit Bayesian updates as it gathers evidence from the web.

It uses **Google Gemini** (function calling) for reasoning and the **Brave Search
API** for web search.

## How it works

At every step the model is *forced* to call a single function,
`update_belief_and_act`, which records:

- `probability` — the current posterior, calibrated to `[0.05, 0.95]`
- `confidence` — `low` / `medium` / `high`
- `evidence_for` / `evidence_against` — concrete supporting / contradicting evidence
- `update_reasoning` — the Bayesian update just performed
- `action` — `web_search` or `submit`
- `action_input` — a search query, or a final justification

When the action is `web_search`, the agent runs a Brave search, feeds the results
back as a function response, and updates again — up to `MAX_STEPS` (10). When it
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
- A Google Gemini API key and a Brave Search API key

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
#   BRAVE_API_KEY=...
```

## Usage

```bash
# Default example question, no prior:
uv run forecaster.py

# Inject a prior anchor (a market price or historical base rate):
uv run forecaster.py --prior 0.10

# Your own question, with a prior and a custom number of trials:
uv run forecaster.py "Will X happen before 2027?" --prior 0.62 --trials 3

# Add structured context (shown under "Background and resolution criteria"):
uv run forecaster.py "Will X happen before 2027?" \
  --background "X has been attempted 3 times since 2020." \
  --resolution-criteria "Resolves YES if X is officially confirmed before 2027-01-01."
```

The question is an optional positional argument; `--prior`, `--trials`,
`--background`, and `--resolution-criteria` are optional flags. `background` and
`resolution_criteria` are shown to the forecaster and the search summarizer. You can also call the API directly from Python:

```python
from forecaster import aggregate_forecasts

result = aggregate_forecasts("Will event X happen before date Y?")
print(result.probability)   # aggregated probability (logit-space mean)
print(result.summary)       # synthesized briefing of the argument

# Anchor on an external prior (e.g. a prediction-market price of 62%):
result = aggregate_forecasts("Will event X happen before date Y?", prior=0.62)
```

### Finding markets to forecast

`polymarket.py` lists open Polymarket events by liquidity and time-to-resolution
(a candidate list to forecast). It reads the public Gamma API — no key required.

```bash
# Events with >= $50k liquidity resolving within 7 days, most liquid first:
uv run polymarket.py --min-liquidity 50000 --days 7

# Restrict to a tag (filtered server-side), cap the count, emit raw JSON:
uv run polymarket.py --min-liquidity 50000 --days 30 --tag politics --limit 20 --json
```

Polymarket has no single `category` field; events are labelled with multiple
**tags** (e.g. `Politics`, `Sports`, `Crypto`). `--tag <slug>` filters by one
server-side; the readable listing and the `tags` field show each event's labels.

```python
from polymarket import fetch_markets

events = fetch_markets(min_liquidity=50000, within_days=7, tag_slug="politics")
# each event: id, slug, title, tags, endDate, liquidity, volume,
# and markets[] = {question, prices: [(outcome, implied_probability), ...]}
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
aggregated probability, and the synthesized briefing — plus an `outcome` field you
fill in when the question resolves — so decisions aren't lost when the process
exits.

Each trial also records the full **decision process**: a `steps` trajectory (the
probability, action, and reasoning at every step), a `sources` map
(`search_X_result_Y → {title, url, query}`, so evidence citations stay resolvable
after the scratch files are deleted), a `stats` block (`steps_used`, `n_searches`,
`n_reads`, `terminated_by`, `seconds`), and per-trial `usage` (token counts). A
run-level `usage` sums all trials plus the briefing, giving the exact token cost.

## Calibration

Forecasts are calibrated with hierarchical Platt scaling (`calibration.py`), keyed
on the **question category** (e.g. a Polymarket tag) as the source:

```bash
# 1. Tag a forecast with its category (recorded on the saved run):
uv run forecaster.py "Will X happen before 2027?" --category politics

# 2. Record the actual outcome once the question resolves:
uv run forecaster.py --resolve runs/<file>.json --outcome 1

# 3. Fit the calibrator across all resolved runs, report it, and save the fit:
uv run forecaster.py --calibrate
```

The fit learns a global slope/intercept plus a per-category offset that is L2-
regularized toward zero (`--lam`, default 1.0) — categories with little data
shrink to the pooled global fit. It reports leave-one-out log loss and Brier
before vs. after, and writes the parameters to `calibration_fit.json`.

Once that fit exists, every new forecast is **also** calibrated for its
`--category`: the run records both the raw `probability` (which feeds future
fits — calibration is never applied on top of itself) and a `calibrated_probability`.
A category unseen at fit time falls back to the global fit (offset 0).
`calibration.py` also runs standalone (`uv run calibration.py`) on synthetic data.

### Building calibration data from Polymarket

Calibration needs *resolved* forecasts, and most questions take months to settle.
Two drivers forecast Polymarket markets to build a calibration set, then
auto-resolve them from Polymarket once they settle. By default each market's
price is injected as the **prior anchor** (the crowd signal): the model starts
from it and updates away only as evidence justifies. Pass `--no-use-market-prior`
to forecast **independently** instead (the agent forms its own base rate; markets
are ignored as evidence). Either way a run is saved per market, tagged
`--category`.

**`train_markets.py`** — general, selects markets by Polymarket **tag**. Pool
several tags into one calibration category (e.g. politics + geopolitics):

```bash
# Preview (no LLM calls); tags are fetched server-side and de-duplicated:
uv run train_markets.py --tags politics,geopolitics --category politics --dry-run

# Forecast competitive binary markets (price within --min/--max-price), most
# liquid first, capped per event and overall:
uv run train_markets.py --tags politics,geopolitics --category politics --limit 10

# Restrict to a liquidity band (both bounds optional, in USD):
uv run train_markets.py --tags politics,geopolitics --min-liquidity 100000 --max-liquidity 500000
```

Novelty markets are filtered out by default: events tagged `tweets-markets`
(`--exclude-tags <slugs>`) and markets whose question matches `publicly insult`
(`--exclude-pattern <regex>`, a case-insensitive regex on the question/title).
Pass `--exclude-tags ''` / `--exclude-pattern ''` to keep them, or extend the
regex (e.g. `--exclude-pattern 'publicly insult|speak to'`) to drop more.

To discover tag slugs to pass to `--tags` / `--exclude-tags`, run
`uv run train_markets.py --list-tags` — it prints every tag on active, liquid
events with event counts (add `--tags politics` to see the tags that co-occur
with a given one).

**`train_worldcup.py`** — a sports preset that forecasts each upcoming match's
binary "Will {team} win?" markets (match-winner markets resolve in *hours*, so
they build a calibration set fast):

```bash
uv run train_worldcup.py --days 3 --limit 4
```

Then, for either driver:

```bash
# After the markets settle, record outcomes automatically from Gamma:
uv run train_markets.py --resolve        # resolves ALL pending Polymarket runs

# Fit once enough runs are resolved:
uv run forecaster.py --calibrate
```

Each run stores a `market` reference (event slug + condition id) and the
`market_price` (and, when anchoring is on, that same value as the run's `prior`),
so `--resolve` maps it back to the settled outcome. Cost scales as markets ×
`--trials`, so start with `--dry-run` and a small `--limit`. Pooling tags under
one `--category` calibrates them together (their own bias offset in the shared
hierarchical fit); see [Calibration](#calibration).

**Re-runs skip what's done.** Both drivers de-duplicate against `runs/` by market
condition id, so a market already forecast (whether pending or resolved) is not
forecast again — only `--resolve` revisits it. `--limit` therefore counts *new*
markets, so you can build the set incrementally: forecast a batch, resolve, run
again to pick up the next batch.

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
| `BRAVE_MAX_RESULTS` | `5` | Results per search |
