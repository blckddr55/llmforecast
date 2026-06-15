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

import logging
import os
import time
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

# --- Configuration -----------------------------------------------------------

MODEL = "gemini-2.5-flash"  # matches polyscrape's workhorse model

MAX_STEPS = 10  # max agent steps per run before forcing a submission
NUM_TRIALS = 5  # independent runs aggregated per question
MAX_OUTPUT_TOKENS = 4096  # generous ceiling for a single structured function call
TEMPERATURE = 1.0  # >0 so the NUM_TRIALS runs genuinely diverge
TAVILY_MAX_RESULTS = 5

# Disable Gemini "thinking" so each forced function call is cheap and fits the
# token budget, mirroring the no-thinking design of the original. 0 works on
# gemini-2.5-flash / flash-lite; use -1 (dynamic) or a positive budget if you
# switch MODEL to a *-pro model, which cannot fully disable thinking.
THINKING_BUDGET = 0

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

SYSTEM_PROMPT = (
    "You are a Bayesian Linguistic Forecaster. You estimate the probability that "
    "a given claim is true or a given event will happen, expressed as a calibrated "
    "number between 0.05 and 0.95.\n\n"
    "Reason like a superforecaster: begin from a sensible base rate, then update "
    "your belief incrementally as evidence arrives. At every step you MUST call "
    "the `update_belief_and_act` function to record your current posterior "
    "probability, the evidence for and against, your confidence, and the reasoning "
    "behind your update.\n\n"
    "- If you still need information, set action='web_search' and put a focused, "
    "specific query in action_input. You will receive search results and update "
    "again.\n"
    "- When further searching is unlikely to change your estimate, set "
    "action='submit' and use action_input to briefly justify your final number.\n\n"
    "Calibrate carefully and avoid overconfidence: reserve probabilities near 0.05 "
    "or 0.95 for cases backed by strong, corroborated evidence. You have at most "
    f"{MAX_STEPS} steps."
)

INITIAL_USER_TEMPLATE = (
    "Forecasting question: {question}\n\n"
    "Establish a reasonable base rate, then decide whether to search for evidence "
    "or submit. Call the update_belief_and_act function now."
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


def run_agent(question: str, max_steps: int = MAX_STEPS) -> float:
    """Run a single forecasting agent loop and return its final probability.

    The model is forced to call `update_belief_and_act` at every step (function
    calling mode 'ANY'). On a 'web_search' action we run Tavily, append the
    result as a function_response, and continue. On 'submit' (or when the step
    budget is exhausted) we return the last recorded probability, clamped to
    [0.05, 0.95].
    """
    client = get_genai_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
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
            parts=[types.Part(text=INITIAL_USER_TEMPLATE.format(question=question))],
        )
    ]
    probability = 0.5  # fallback prior if the loop exits unexpectedly

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

        belief = call.args or {}
        probability = _clamp(float(belief.get("probability", probability)), 0.05, 0.95)
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

    return probability


# --- Multi-trial aggregation -------------------------------------------------


def aggregate_forecasts(question: str, num_trials: int = NUM_TRIALS) -> float:
    """Run the agent `num_trials` times and combine results via a logit-space mean.

    Averaging in log-odds (logit) space is more principled than averaging raw
    probabilities: it treats evidence additively and is symmetric around 0.5.
    Returns the aggregated probability in [0.05, 0.95].
    """
    logger.info(
        "Forecasting in %d trials | model=%s | question: %s", num_trials, MODEL, question
    )
    overall_start = time.perf_counter()

    probabilities = []
    for trial in range(1, num_trials + 1):
        logger.info("=== Trial %d/%d ===", trial, num_trials)
        trial_start = time.perf_counter()
        probability = run_agent(question)
        probabilities.append(probability)
        logger.info(
            "Trial %d/%d result: p=%.3f (%.1fs)",
            trial,
            num_trials,
            probability,
            time.perf_counter() - trial_start,
        )

    probs = np.asarray(probabilities, dtype=float)
    aggregated = float(expit(np.mean(logit(probs))))

    logger.info(
        "All trials %s | logit-space mean=%.3f | elapsed %.1fs",
        [round(p, 3) for p in probabilities],
        aggregated,
        time.perf_counter() - overall_start,
    )
    return aggregated


# --- Example execution -------------------------------------------------------


def main() -> None:
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

    question = (
        "Will Volodymyr Zelenskyy win the Nobel Peace Price in 2026?"
    )

    final_probability = aggregate_forecasts(question, num_trials=NUM_TRIALS)

    print(
        f"\nAggregated probability (logit-space mean of {NUM_TRIALS} trials): "
        f"{final_probability:.3f}"
    )


if __name__ == "__main__":
    main()
