"""Bayesian Linguistic Forecaster (Google Gemini + Tavily).

An agent that estimates the probability of a forecasting question by reasoning
like a superforecaster: it starts from a base rate and performs explicit Bayesian
updates as it gathers web evidence via Tavily. At every step the model is forced
to call a single function (`update_belief_and_act`) that records its current
posterior probability, the supporting/contradicting evidence, and the next
action (search again, or submit).

To reduce variance, each question is forecast with several independent agent
runs, and the resulting probabilities are combined with a logit-space mean
(averaging in log-odds space rather than probability space).

API keys are loaded from a local .env file (git-ignored) and must include:
    GEMINI_API_KEY
    TAVILY_API_KEY

Run with:  uv run forecaster.py
Verbose:   LOG_LEVEL=DEBUG uv run forecaster.py   (adds the full evidence lists)
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.special import expit, logit
from tavily import TavilyClient

logger = logging.getLogger("forecaster")

# Load API keys from this project's local .env file (git-ignored).
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# Saved forecast records (one JSON per run) so decisions aren't lost.
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# --- Configuration -----------------------------------------------------------

MODEL = "gemini-3.5-flash"  # latest GA Gemini generation (newer than any 3.x Pro preview)

MAX_STEPS = 10  # max agent steps per run before forcing a submission
NUM_TRIALS = 5  # independent runs aggregated per question
MAX_OUTPUT_TOKENS = 8192  # headroom for high-effort thinking + the structured function call
TEMPERATURE = 1.0  # >0 so the NUM_TRIALS runs genuinely diverge
TAVILY_MAX_RESULTS = 5

# Prediction-market / betting platforms excluded from search — their odds are
# derivative, and reasoning from them would be circular.
EXCLUDE_DOMAINS = [
    "polymarket.com",
    "kalshi.com",
    "predictit.org",
    "metaculus.com",
    "manifold.markets",
    "futuur.com",
    "insightprediction.com",
    "electionbettingodds.com",
    "predictionhunt.com",
    "oddschecker.com",
    "betfair.com",
    "smarkets.com",
    "pinnacle.com",
]

# Gemini 3 "thinking level" applied to each forced function call. "high"
# maximizes reasoning depth; use "low" for cheaper, faster runs. On
# gemini-3.5-flash this stays modest (a few hundred thinking tokens); give
# MAX_OUTPUT_TOKENS more headroom if you switch to a heavier model.
THINKING_LEVEL = "high"

TOOL_NAME = "update_belief_and_act"

# --- Function (tool) schema --------------------------------------------------

UPDATE_BELIEF_FUNCTION = types.FunctionDeclaration(
    name=TOOL_NAME,
    description=(
        "Record your current Bayesian posterior belief about the forecasting "
        "question and choose the next action. Call this at every step: update "
        "your probability in light of the latest evidence, then either search "
        "the web for more evidence or submit your final estimate."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "probability": types.Schema(
                type="NUMBER",
                minimum=0.05,
                maximum=0.95,
                description=(
                    "Your current posterior probability that the claim is true / "
                    "the event occurs, as a calibrated float between 0.05 and 0.95."
                ),
            ),
            "confidence": types.Schema(
                type="STRING",
                enum=["low", "medium", "high"],
                description="How confident you are in this probability estimate.",
            ),
            "evidence_for": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING"),
                description="Concrete pieces of evidence that raise the probability.",
            ),
            "evidence_against": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING"),
                description="Concrete pieces of evidence that lower the probability.",
            ),
            "update_reasoning": types.Schema(
                type="STRING",
                description=(
                    "Explain how the latest evidence moved your estimate from the "
                    "previous step (the Bayesian update you just performed)."
                ),
            ),
            "action": types.Schema(
                type="STRING",
                enum=["web_search", "submit"],
                description=(
                    "Choose 'web_search' to gather more evidence, or 'submit' to "
                    "finalize the current probability as your answer."
                ),
            ),
            "action_input": types.Schema(
                type="STRING",
                description=(
                    "If action is 'web_search', a focused search query. If action "
                    "is 'submit', a brief justification of the final probability."
                ),
            ),
        },
        required=[
            "probability",
            "confidence",
            "evidence_for",
            "evidence_against",
            "update_reasoning",
            "action",
            "action_input",
        ],
    ),
)

UPDATE_BELIEF_TOOL = types.Tool(function_declarations=[UPDATE_BELIEF_FUNCTION])

NO_MARKETS_RULE = (
    "Do NOT use prediction markets or betting odds as evidence — not Polymarket, "
    "Kalshi, PredictIt, Metaculus, Manifold, or sportsbook odds, and not even when "
    "a news article quotes them. Ignore any market-implied probability or odds that "
    "appear in search results, and never cite them. Base the forecast solely on "
    "primary evidence: events, official data, fundamentals, and expert analysis."
)

SYSTEM_PROMPT = (
    "You are a Bayesian Linguistic Forecaster. You estimate the probability that "
    "a given claim is true or a given event will happen, expressed as a calibrated "
    "number between 0.05 and 0.95.\n\n"
    "Reason like a superforecaster: begin from a sensible base rate, then update "
    "your belief incrementally as evidence arrives. At every step you MUST call "
    "the `update_belief_and_act` function to record your current posterior "
    "probability, the evidence for and against, your confidence, and the reasoning "
    "behind your update.\n\n"
    f"{NO_MARKETS_RULE}\n\n"
    "- If you still need information, set action='web_search' and put a focused, "
    "specific query in action_input. You will receive search results and update "
    "again.\n"
    "- When further searching is unlikely to change your estimate, set "
    "action='submit' and use action_input to briefly justify your final number.\n\n"
    "Calibrate carefully and avoid overconfidence: reserve probabilities near 0.05 "
    "or 0.95 for cases backed by strong, corroborated evidence. You have at most "
    f"{MAX_STEPS} steps."
)

def build_initial_message(question: str, prior: float | None = None) -> str:
    """Build the opening user message, optionally seeded with a prior anchor.

    `prior` is an external probability anchor in [0, 1] — e.g. a market-implied
    price for a market question, or a historical base rate for a dataset
    question. When omitted, the agent forms its own base rate.
    """
    if prior is not None:
        anchor = (
            f"You are given an external prior anchor of {prior:.0%} for this "
            "question (for example, the market-implied probability for a market "
            "question, or the historical base rate for a dataset question). Begin "
            "from this anchor and update away from it only as far as the evidence "
            "justifies."
        )
    else:
        anchor = "Establish a reasonable base rate before gathering evidence."
    return (
        f"Forecasting question: {question}\n\n"
        f"{anchor}\n\n"
        "Decide whether to search for evidence or submit. Call the "
        "update_belief_and_act function now."
    )

# --- Clients (lazily constructed so the module is import-safe) ---------------

_genai_client: genai.Client | None = None
_tavily_client: TavilyClient | None = None


def get_genai_client() -> genai.Client:
    """Return a cached Gemini client (reads GEMINI_API_KEY from the env)."""
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"GEMINI_API_KEY is not set (expected in {ENV_PATH}).")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def get_tavily_client() -> TavilyClient:
    """Return a cached Tavily client (reads TAVILY_API_KEY from the env)."""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(f"TAVILY_API_KEY is not set (expected in {ENV_PATH}).")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# --- Search integration ------------------------------------------------------


def tavily_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> str:
    """Run a Tavily web search and return content formatted for LLM context.

    Uses an "advanced" search with Tavily's synthesized answer plus the top
    source snippets, assembled into a compact, readable string suitable for a
    function response. (Tavily also offers `client.get_search_context(...)`,
    which returns a token-budgeted context string directly, if you prefer that.)
    """
    client = get_tavily_client()
    logger.info("search: %s", query)
    start = time.perf_counter()
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            exclude_domains=EXCLUDE_DOMAINS,
        )
    except Exception as exc:  # don't let a transient search failure kill the run
        logger.warning("search failed for %r: %s", query, exc)
        return f"Search failed for query {query!r}: {exc}"

    parts = [f"Search query: {query}"]

    answer = response.get("answer")
    if answer:
        parts.append(f"\nSummary answer:\n{answer}")

    results = response.get("results", [])
    if not results:
        parts.append("\nNo results found.")
    else:
        parts.append("\nSources:")
        for i, result in enumerate(results, start=1):
            title = result.get("title", "(untitled)")
            url = result.get("url", "")
            content = (result.get("content") or "").strip()
            parts.append(f"\n[{i}] {title}\n{url}\n{content}")
            logger.info("  [%d] %s — %s", i, title, url)

    formatted = "\n".join(parts)
    logger.info(
        "search done: %d result(s), %d chars (%.1fs)",
        len(results),
        len(formatted),
        time.perf_counter() - start,
    )
    return formatted


# --- Agent loop --------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def run_agent(
    question: str, prior: float | None = None, max_steps: int = MAX_STEPS
) -> dict:
    """Run a single forecasting agent loop and return its final belief.

    The model is forced to call `update_belief_and_act` at every step (function
    calling mode 'ANY'). On a 'web_search' action we run Tavily, append the
    result as a function_response, and continue. On 'submit' (or when the step
    budget is exhausted) we return the last belief — the tool-call arguments
    (probability, evidence_for/against, reasoning, ...) with `probability`
    clamped to [0.05, 0.95].

    If `prior` (a probability in [0, 1]) is given, the agent starts anchored on
    it instead of forming its own base rate (prior injection).
    """
    if prior is not None and not 0.0 <= prior <= 1.0:
        raise ValueError(f"prior must be a probability in [0, 1], got {prior!r}")

    client = get_genai_client()
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    config = types.GenerateContentConfig(
        system_instruction=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Today's date is {today}; do not search for the current date."
        ),
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        tools=[UPDATE_BELIEF_TOOL],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=[TOOL_NAME],
            )
        ),
    )

    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=build_initial_message(question, prior))],
        )
    ]
    probability = 0.5  # fallback prior if the loop exits unexpectedly
    belief: dict = {
        "probability": probability,
        "confidence": "low",
        "evidence_for": [],
        "evidence_against": [],
        "update_reasoning": "",
        "action": "submit",
        "action_input": "",
    }

    for step in range(1, max_steps + 1):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )

        calls = response.function_calls or []
        if not calls:
            # Should not happen under forced function calling; bail with current estimate.
            logger.warning("step %d: no function call returned; stopping early", step)
            break
        call = calls[0]

        belief = dict(call.args or {})
        probability = _clamp(float(belief.get("probability", probability)), 0.05, 0.95)
        belief["probability"] = probability  # store the clamped value
        action = belief.get("action", "submit")

        logger.info(
            "step %2d/%d | p=%.3f | confidence=%s | action=%s",
            step,
            max_steps,
            probability,
            belief.get("confidence", "?"),
            action,
        )
        reasoning = belief.get("update_reasoning")
        if reasoning:
            logger.info("  reasoning: %s", reasoning)
        logger.debug("  evidence_for: %s", belief.get("evidence_for", []))
        logger.debug("  evidence_against: %s", belief.get("evidence_against", []))

        # Preserve the model's function-call turn.
        contents.append(response.candidates[0].content)

        if action == "submit" or step == max_steps:
            if action == "submit":
                logger.info("  submit: %s", belief.get("action_input", ""))
            else:
                logger.info(
                    "  step budget (%d) reached; submitting current estimate", max_steps
                )
            break

        # action == "web_search": run the search and feed the result back.
        query = belief.get("action_input") or question
        result = tavily_search(query)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=call.name, response={"result": result}
                    )
                ],
            )
        )

    return belief


# --- Multi-trial aggregation -------------------------------------------------


@dataclass
class ForecastResult:
    """Aggregated forecast plus a model-written briefing of the argument."""

    probability: float  # aggregated (logit-space mean), in [0.05, 0.95]
    trials: list[dict]  # per-trial final beliefs (probability, evidence, reasoning)
    summary: str  # synthesized "case for / case against / bottom line"

    @property
    def trial_probabilities(self) -> list[float]:
        return [float(b["probability"]) for b in self.trials]


def summarize_forecast(
    question: str,
    beliefs: list[dict],
    aggregated: float,
    prior: float | None = None,
) -> str:
    """Synthesize the trials into one short briefing via a final model call."""
    trial_blocks = []
    for i, belief in enumerate(beliefs, start=1):
        rationale = belief.get("action_input") or belief.get("update_reasoning") or "n/a"
        trial_blocks.append(
            f"Trial {i} — probability {belief.get('probability', 0.0):.2f} "
            f"(confidence: {belief.get('confidence', '?')})\n"
            f"  For: {'; '.join(belief.get('evidence_for') or []) or 'none'}\n"
            f"  Against: {'; '.join(belief.get('evidence_against') or []) or 'none'}\n"
            f"  Rationale: {rationale}"
        )
    trials_text = "\n\n".join(trial_blocks)
    spread = ", ".join(f"{b.get('probability', 0.0):.2f}" for b in beliefs)
    prior_line = (
        f"An external prior anchor of {prior:.0%} was supplied.\n\n"
        if prior is not None
        else ""
    )
    prompt = (
        f"Forecasting question: {question}\n\n"
        f"{prior_line}"
        f"{len(beliefs)} independent forecasts were run; their probabilities were "
        f"[{spread}], combined (logit-space mean) into {aggregated:.2f}.\n\n"
        f"Per-trial evidence and reasoning:\n{trials_text}\n\n"
        "Write a concise briefing for a decision-maker that synthesizes ACROSS the "
        "trials (do not just list them), using exactly these sections:\n"
        f"Forecast: the aggregate probability ({aggregated:.0%}) and the spread.\n"
        "The case for: the strongest, best-corroborated evidence raising the probability.\n"
        "The case against: the strongest evidence lowering it.\n"
        "Bottom line: one or two sentences on the overall judgement and key uncertainty."
        "\n\nDo not mention, cite, or rely on prediction markets or betting odds "
        "anywhere in the briefing, even if the per-trial notes above reference them. "
        "Build the case for and against from substantive evidence only."
    )
    response = get_genai_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return (response.text or "").strip()


def aggregate_forecasts(
    question: str, prior: float | None = None, num_trials: int = NUM_TRIALS
) -> ForecastResult:
    """Run the agent `num_trials` times and combine results via a logit-space mean.

    Averaging in log-odds (logit) space is more principled than averaging raw
    probabilities: it treats evidence additively and is symmetric around 0.5.
    An optional `prior` anchor (a probability in [0, 1]) is forwarded to every
    run. Returns a `ForecastResult` with the aggregated probability, the per-trial
    spread, and a synthesized briefing of the overall argument.
    """
    logger.info(
        "Forecasting in %d trials | model=%s | prior=%s | question: %s",
        num_trials,
        MODEL,
        f"{prior:.0%}" if prior is not None else "none",
        question,
    )
    overall_start = time.perf_counter()

    beliefs = []
    for trial in range(1, num_trials + 1):
        logger.info("=== Trial %d/%d ===", trial, num_trials)
        trial_start = time.perf_counter()
        belief = run_agent(question, prior=prior)
        beliefs.append(belief)
        logger.info(
            "Trial %d/%d result: p=%.3f (%.1fs)",
            trial,
            num_trials,
            belief["probability"],
            time.perf_counter() - trial_start,
        )

    probabilities = [float(b["probability"]) for b in beliefs]
    aggregated = float(expit(np.mean(logit(np.asarray(probabilities)))))

    logger.info(
        "All trials %s | logit-space mean=%.3f | elapsed %.1fs",
        [round(p, 3) for p in probabilities],
        aggregated,
        time.perf_counter() - overall_start,
    )

    logger.info("Synthesizing final briefing...")
    summary = summarize_forecast(question, beliefs, aggregated, prior)
    return ForecastResult(
        probability=aggregated,
        trials=beliefs,
        summary=summary,
    )


# --- Persistence -------------------------------------------------------------


def save_run(
    question: str,
    result: ForecastResult,
    prior: float | None = None,
    num_trials: int | None = None,
) -> Path:
    """Write a forecast run to RUNS_DIR as JSON so the decision is not lost."""
    RUNS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "forecast"
    path = RUNS_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    payload = {
        "timestamp": now.isoformat(),
        "question": question,
        "prior": prior,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "num_trials": num_trials if num_trials is not None else len(result.trials),
        "probability": result.probability,
        "trial_probabilities": result.trial_probabilities,
        "summary": result.summary,
        "trials": result.trials,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- Example execution -------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian Linguistic Forecaster")
    parser.add_argument(
        "question",
        nargs="?",
        default="Will Volodymyr Zelenskyy win the Nobel Peace Price in 2026?",
        help="Forecasting question (defaults to a built-in example).",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=None,
        help="Optional prior anchor in [0, 1] — a market price or historical base rate.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=NUM_TRIALS,
        help=f"Independent runs to aggregate (default: {NUM_TRIALS}).",
    )
    args = parser.parse_args()
    if args.prior is not None and not 0.0 <= args.prior <= 1.0:
        parser.error("--prior must be a probability between 0 and 1")

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    missing = [
        key for key in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)} "
            f"(expected in {ENV_PATH})"
        )

    result = aggregate_forecasts(
        args.question, prior=args.prior, num_trials=args.trials
    )
    saved_path = save_run(args.question, result, prior=args.prior, num_trials=args.trials)

    spread = ", ".join(f"{p:.2f}" for p in result.trial_probabilities)
    print("\n" + "=" * 72)
    print(result.summary)
    print("=" * 72)
    print(
        f"Aggregated probability: {result.probability:.3f}   "
        f"({args.trials} trials: {spread})"
    )
    print(f"Saved to: {saved_path.relative_to(Path.cwd()) if saved_path.is_relative_to(Path.cwd()) else saved_path}")


if __name__ == "__main__":
    main()
