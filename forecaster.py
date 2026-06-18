"""Bayesian Linguistic Forecaster (Google Gemini + Brave Search).

An agent that estimates the probability of a forecasting question by reasoning
like a superforecaster: it starts from a base rate and performs explicit Bayesian
updates as it gathers web evidence via Brave Search. At every step the model is forced
to call a single function (`update_belief_and_act`) that records its current
posterior probability, the supporting/contradicting evidence, and the next
action (search again, or submit).

To reduce variance, each question is forecast with several independent agent
runs, and the resulting probabilities are combined with a logit-space mean
(averaging in log-odds space rather than probability space).

API keys are loaded from a local .env file (git-ignored) and must include:
    GEMINI_API_KEY
    BRAVE_API_KEY

Run with:  uv run forecaster.py
Verbose:   LOG_LEVEL=DEBUG uv run forecaster.py   (adds the full evidence lists)
"""

import argparse
import html
import json
import logging
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.special import expit, logit

import calibration

logger = logging.getLogger("forecaster")

# Load API keys from this project's local .env file (git-ignored).
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# Saved forecast records (one JSON per run) so decisions aren't lost.
RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Persisted hierarchical-Platt calibration fit (written by --calibrate, applied
# to new forecasts keyed on question category).
CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration_fit.json"

# --- Configuration -----------------------------------------------------------

MODEL = "gemini-3.5-flash"  # latest GA Gemini generation (newer than any 3.x Pro preview)

MAX_STEPS = 14  # max agent steps per run before forcing a submission (read_files adds round-trips)
NUM_TRIALS = 5  # independent runs aggregated per question
MAX_OUTPUT_TOKENS = 8192  # headroom for high-effort thinking + the structured function call
TEMPERATURE = 1.0  # >0 so the NUM_TRIALS runs genuinely diverge
BRAVE_MAX_RESULTS = 10  # snippets shown per search (full text goes to files, not context)
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Progressive disclosure: search results are written to local files; the agent
# reads chosen ones through a cheaper sub-LLM summarizer (Gemini Flash) so the
# main context only ever sees short snippets plus the summaries it requested.
SUMMARIZER_MODEL = MODEL  # the sub-LLM that summarizes read files
SUMMARIZER_MAX_TOKENS = 1536  # output cap for a file summary
SUMMARIZER_MAX_DOC_CHARS = 20000  # cap on concatenated file text sent to the summarizer

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
                enum=["web_search", "read_files", "submit"],
                description=(
                    "Choose 'web_search' to run a new search (you get back short "
                    "snippets, each with a file id); 'read_files' to pull the full "
                    "content of chosen results (by id) back as a focused summary; "
                    "or 'submit' to finalize the current probability as your answer."
                ),
            ),
            "action_input": types.Schema(
                type="STRING",
                description=(
                    "If action is 'web_search', a focused search query. If action is "
                    "'read_files', the instruction for the summarizer — the specific "
                    "facts to extract from the chosen files. If action is 'submit', a "
                    "brief justification of the final probability."
                ),
            ),
            "read_file_ids": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING"),
                description=(
                    "Only for action='read_files': the file ids (e.g. "
                    "'search_1_result_3') from earlier search snippets to read in "
                    "full. Pick the few most promising; their content is summarized "
                    "with your action_input and returned to you."
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
    "You gather evidence through a two-stage search:\n"
    "- set action='web_search' with a focused query in action_input. You get back "
    "a list of short snippets, each tagged with a file id (e.g. "
    "'search_1_result_3'). Snippets are deliberately brief — scan them to spot the "
    "most promising sources.\n"
    "- set action='read_files' with the chosen ids in read_file_ids and, in "
    "action_input, a specific instruction for what to extract. The full text of "
    "those results is summarized for you and returned. Read before trusting a "
    "snippet; do not submit on snippets alone when a result looks decisive.\n"
    "- when further evidence is unlikely to change your estimate, set "
    "action='submit' and use action_input to briefly justify your final number.\n\n"
    "Calibrate carefully and avoid overconfidence: reserve probabilities near 0.05 "
    "or 0.95 for cases backed by strong, corroborated evidence. You have at most "
    f"{MAX_STEPS} steps, so spend reads on the results that matter."
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
        "Decide whether to search for evidence (web_search), read promising "
        "results in full (read_files), or submit. Call the update_belief_and_act "
        "function now."
    )

# --- Clients (lazily constructed so the module is import-safe) ---------------

_genai_client: genai.Client | None = None
_brave_api_key: str | None = None


def get_genai_client() -> genai.Client:
    """Return a cached Gemini client (reads GEMINI_API_KEY from the env)."""
    global _genai_client
    if _genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(f"GEMINI_API_KEY is not set (expected in {ENV_PATH}).")
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def get_brave_api_key() -> str:
    """Return the cached Brave Search API key (reads BRAVE_API_KEY from the env)."""
    global _brave_api_key
    if _brave_api_key is None:
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            raise RuntimeError(f"BRAVE_API_KEY is not set (expected in {ENV_PATH}).")
        _brave_api_key = api_key
    return _brave_api_key


# --- Search integration ------------------------------------------------------


def _strip_html(text: str) -> str:
    """Strip HTML tags and unescape entities from a Brave result snippet."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _is_excluded(url: str) -> bool:
    """True if the URL's host is (a subdomain of) an EXCLUDE_DOMAINS entry."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith(f".{d}") for d in EXCLUDE_DOMAINS)


def brave_search(
    query: str, max_results: int = BRAVE_MAX_RESULTS
) -> tuple[list[dict], str | None]:
    """Run a Brave web search; return (results, error_message).

    Each result dict has `title`, `url`, `short_snippet` (the one-line Brave
    description shown to the agent in context) and `full_content` (description +
    all extra excerpts, written to a file for on-demand reading). Requesting
    `extra_snippets=true` yields up to 5 richer excerpts per result, subject to
    the Brave plan tier. Prediction-market / betting sites (EXCLUDE_DOMAINS) are
    filtered out client-side — the API has no server-side exclusion — so we
    over-fetch to still leave up to `max_results`. On failure returns ([], msg).
    """
    api_key = get_brave_api_key()
    logger.info("search: %s", query)
    start = time.perf_counter()
    count = min(20, max_results + len(EXCLUDE_DOMAINS))
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": count,
            "result_filter": "web",
            "extra_snippets": "true",
        }
    )
    request_url = f"{BRAVE_ENDPOINT}?{params}"
    logger.debug("brave request: %s", request_url)
    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            response = json.load(resp)
    except Exception as exc:  # don't let a transient search failure kill the run
        logger.warning("search failed for %r: %s", query, exc)
        return [], f"Search failed for query {query!r}: {exc}"

    raw = [
        r
        for r in (response.get("web") or {}).get("results", [])
        if not _is_excluded(r.get("url", ""))
    ][:max_results]

    results = []
    for r in raw:
        title = _strip_html(r.get("title", "")) or "(untitled)"
        url = r.get("url", "")
        description = _strip_html(r.get("description", ""))
        extras = [_strip_html(s) for s in r.get("extra_snippets") or []]
        if not extras:
            logger.debug("no extra_snippets for %s (plan tier may not include them)", url)
        body = "\n".join(filter(None, [description, *extras]))
        results.append(
            {
                "title": title,
                "url": url,
                "short_snippet": description,
                "full_content": f"{title}\n{url}\n\n{body}".strip(),
            }
        )
        logger.info("  %s — %s", title, url)

    logger.info(
        "search done: %d result(s) (%.1fs)", len(results), time.perf_counter() - start
    )
    return results, None


def register_search_results(
    results: list[dict],
    search_index: int,
    scratch: Path,
    registry: dict[str, Path],
) -> str:
    """Write each result's full content to a file, register its id, and return
    the short snippet list (ids + title + url + one-line snippet) for context.

    This is the progressive-disclosure split: only the short snippets enter the
    agent's context; the full text lives in `<scratch>/<id>.md` until the agent
    asks to read it.
    """
    if not results:
        return "No results found."
    lines = []
    for j, result in enumerate(results, start=1):
        file_id = f"search_{search_index}_result_{j}"
        path = scratch / f"{file_id}.md"
        path.write_text(result["full_content"], encoding="utf-8")
        registry[file_id] = path
        snippet = result["short_snippet"] or "(no snippet)"
        lines.append(f"[{file_id}] {result['title']}\n{result['url']}\n{snippet}")
    header = (
        f"{len(results)} result(s). To read the full content of the most relevant "
        "ones, call action='read_files' with their ids in read_file_ids and a "
        "focused extraction instruction in action_input.\n"
    )
    return header + "\n\n".join(lines)


def summarize(question: str, instruction: str, docs: list[str]) -> str:
    """Summarize selected search-result files with the sub-LLM (Gemini Flash).

    `instruction` is the main agent's extraction prompt (what facts to pull). The
    original `question` and the no-markets rule are injected independently so the
    summarizer stays on task and never surfaces market odds the main agent must
    ignore. Returns an error string (never raises) so a failure can't kill a run.
    """
    if not docs:
        return "No documents to summarize."
    instruction = instruction.strip() or (
        f"Extract any facts relevant to the forecasting question: {question}"
    )
    combined = "\n\n---\n\n".join(docs)[:SUMMARIZER_MAX_DOC_CHARS]
    try:
        response = get_genai_client().models.generate_content(
            model=SUMMARIZER_MODEL,
            contents=f"{instruction}\n\nDocuments:\n\n{combined}",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a research assistant extracting facts to help answer "
                    f"this forecasting question: {question}\n\n{NO_MARKETS_RULE}\n\n"
                    "Summarize only what the documents actually say that bears on the "
                    "extraction request. Be concise and concrete; keep dates and "
                    "figures. If the documents are irrelevant, say so plainly."
                ),
                temperature=0.3,
                max_output_tokens=SUMMARIZER_MAX_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return (response.text or "").strip() or "Summary was empty."
    except Exception as exc:
        logger.warning("summarization failed: %s", exc)
        return f"Summarization failed: {exc}"


# --- Agent loop --------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def run_read_files(question: str, belief: dict, registry: dict[str, Path]) -> str:
    """Resolve the agent's requested file ids, summarize them, return the result.

    Unknown ids are skipped and reported so the model self-corrects; an empty or
    all-unknown request returns the list of valid ids rather than failing.
    """
    requested = belief.get("read_file_ids") or []
    docs, valid, unknown = [], [], []
    for file_id in requested:
        path = registry.get(file_id)
        if path is None:
            unknown.append(file_id)
        else:
            docs.append(path.read_text(encoding="utf-8"))
            valid.append(file_id)

    if not docs:
        available = ", ".join(sorted(registry)) or "none yet — run a web_search first"
        return f"No readable file ids in {requested or '[]'}. Available ids: {available}."

    notes = f" (ignored unknown ids: {', '.join(unknown)})" if unknown else ""
    summary = summarize(question, belief.get("action_input") or "", docs)
    logger.info("  read_files: %s%s", ", ".join(valid), notes)
    return f"Summary of {', '.join(valid)}{notes}:\n\n{summary}"


def run_agent(
    question: str, prior: float | None = None, max_steps: int = MAX_STEPS
) -> dict:
    """Run a single forecasting agent loop and return its final belief.

    The model is forced to call `update_belief_and_act` at every step (function
    calling mode 'ANY'). On a 'web_search' action we run Brave Search, write each
    result's full text to a per-run scratch file, and feed back only short
    snippets (each tagged with a file id). On a 'read_files' action we summarize
    the chosen files via the sub-LLM and feed back the summary (progressive
    disclosure — the heavy text never floods the main context). On 'submit' (or
    when the step budget is exhausted) we return the last belief — the tool-call
    arguments (probability, evidence_for/against, reasoning, ...) with
    `probability` clamped to [0.05, 0.95].

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

    # Per-run scratch dir + id->path registry for the search files (progressive
    # disclosure). Trials run sequentially, so this state is naturally isolated.
    scratch = Path(tempfile.mkdtemp(prefix="forecast_"))
    registry: dict[str, Path] = {}
    search_index = 0

    try:
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

            if action == "read_files":
                result = run_read_files(question, belief, registry)
            else:  # 'web_search' (and any unexpected action): run a new search
                search_index += 1
                results, error = brave_search(belief.get("action_input") or question)
                result = error or register_search_results(
                    results, search_index, scratch, registry
                )

            # The function_response must echo call.name (the only tool) so Gemini
            # pairs it with the preceding call — true for both action branches.
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
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return belief


# --- Multi-trial aggregation -------------------------------------------------


@dataclass
class ForecastResult:
    """Aggregated forecast plus a model-written briefing of the argument."""

    probability: float  # raw aggregate (logit-space mean), in [0.05, 0.95]
    trials: list[dict]  # per-trial final beliefs (probability, evidence, reasoning)
    summary: str  # synthesized "case for / case against / bottom line"
    # Calibrated aggregate (hierarchical Platt, keyed on `category`); None when no
    # calibration fit has been saved yet. Kept SEPARATE from `probability` so the
    # raw value still feeds future calibration fits (calibrating on an already-
    # calibrated number would double-correct).
    calibrated_probability: float | None = None
    category: str | None = None  # question category — the calibration source key

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


def _apply_calibration(probability: float, category: str | None) -> float | None:
    """Apply the persisted calibration fit to a raw aggregate, if one exists.

    Returns the calibrated probability, or None when no fit has been saved yet.
    The source key is the question's `category`; a category unseen at fit time
    shrinks to the global fit (offset 0) inside `calibration.apply`.
    """
    if not CALIBRATION_PATH.exists():
        return None
    fit = calibration.load_fit(CALIBRATION_PATH)
    return float(calibration.apply(probability, category or "uncategorized", fit))


def aggregate_forecasts(
    question: str,
    prior: float | None = None,
    num_trials: int = NUM_TRIALS,
    category: str | None = None,
) -> ForecastResult:
    """Run the agent `num_trials` times and combine results via a logit-space mean.

    Averaging in log-odds (logit) space is more principled than averaging raw
    probabilities: it treats evidence additively and is symmetric around 0.5.
    An optional `prior` anchor (a probability in [0, 1]) is forwarded to every
    run. `category` (e.g. a Polymarket tag like "politics") is the source key for
    calibration: if a fit has been saved, the aggregate is also calibrated for it.
    Returns a `ForecastResult` with the raw aggregate, the calibrated aggregate
    (when available), the per-trial spread, and a synthesized briefing.
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

    calibrated = _apply_calibration(aggregated, category)
    if calibrated is not None:
        logger.info(
            "Calibrated (category=%s): %.3f -> %.3f",
            category or "uncategorized",
            aggregated,
            calibrated,
        )

    logger.info("Synthesizing final briefing...")
    summary = summarize_forecast(question, beliefs, aggregated, prior)
    return ForecastResult(
        probability=aggregated,
        trials=beliefs,
        summary=summary,
        calibrated_probability=calibrated,
        category=category,
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
        "category": result.category,  # calibration source key
        "outcome": None,  # fill in via --resolve once the question resolves
        "prior": prior,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "num_trials": num_trials if num_trials is not None else len(result.trials),
        "probability": result.probability,  # raw aggregate (feeds calibration fits)
        "calibrated_probability": result.calibrated_probability,
        "trial_probabilities": result.trial_probabilities,
        "summary": result.summary,
        "trials": result.trials,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --- Calibration across sources ----------------------------------------------


def load_resolved_runs() -> list[dict]:
    """Load saved runs that carry a recorded binary outcome (0 or 1)."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("outcome") in (0, 1):
            runs.append(data)
    return runs


def resolve_run(run_path: str | Path, outcome: int) -> Path:
    """Record the actual binary outcome (0/1) on a saved run file."""
    path = Path(run_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["outcome"] = int(outcome)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_category(run: dict) -> str:
    """The calibration source key for a run — its question category."""
    return run.get("category") or "uncategorized"


def calibrate_across_sources(lam: float = 1.0) -> dict:
    """Fit hierarchical Platt scaling across all resolved runs, keyed on category.

    Each run's question `category` is the calibration "source", so the fit learns
    a global slope/intercept plus a per-category offset (L2-regularized toward
    zero — categories with little data shrink to the pooled fit). The fitted
    parameters are written to CALIBRATION_PATH so new forecasts can be calibrated.
    Returns a report with the parameters, the leave-one-out calibrated
    probabilities, and the underlying arrays for scoring.
    """
    runs = load_resolved_runs()
    if len(runs) < 2:
        raise SystemExit(
            f"Need at least 2 resolved runs to calibrate; found {len(runs)}. "
            "Resolve runs first: forecaster.py --resolve <runs/...json> --outcome 0|1"
        )
    sources = sorted({_run_category(r) for r in runs})
    source_of = {category: i for i, category in enumerate(sources)}
    p_hat = np.array([float(r["probability"]) for r in runs])
    y = np.array([int(r["outcome"]) for r in runs])
    source_idx = np.array([source_of[_run_category(r)] for r in runs])

    calibrated = calibration.loocv_calibrate(p_hat, y, source_idx, lam=lam)
    a, b, delta = calibration.fit(
        calibration.safe_logit(p_hat), y, source_idx, len(sources), lam
    )

    # Persist the fit so aggregate_forecasts can calibrate new predictions. The
    # source_of map is saved with the offsets so categories stay aligned.
    fit = {
        "a": float(a),
        "b": float(b),
        "delta": [float(d) for d in delta],
        "source_of": source_of,
        "sources": sources,
        "lam": lam,
        "eps": calibration.EPS,
        "n": len(runs),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_PATH.write_text(
        json.dumps(fit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "n": len(runs),
        "sources": sources,
        "p_hat": p_hat,
        "y": y,
        "calibrated": calibrated,
        "a": float(a),
        "b": float(b),
        "delta": [float(d) for d in delta],
        "path": CALIBRATION_PATH,
    }


def print_calibration_report(report: dict) -> None:
    """Print the per-category offsets and leave-one-out calibration metrics."""
    print(f"Resolved runs: {report['n']} | sources (categories): {len(report['sources'])}")
    print(f"Global fit: slope a={report['a']:.3f}, intercept b={report['b']:.3f}")
    print("Per-category offsets (delta_s):")
    for category, d in zip(report["sources"], report["delta"]):
        print(f"  {category:<26} {d:+.3f}")
    p_hat, y, cal = report["p_hat"], report["y"], report["calibrated"]
    print("Leave-one-out quality (lower is better):")
    print(
        f"  log loss : raw {calibration.log_loss(p_hat, y):.4f}  ->  "
        f"calibrated {calibration.log_loss(cal, y):.4f}"
    )
    print(
        f"  Brier    : raw {calibration.brier(p_hat, y):.4f}  ->  "
        f"calibrated {calibration.brier(cal, y):.4f}"
    )
    print(f"Saved fit to: {report['path'].name}")


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
    parser.add_argument(
        "--category",
        default=None,
        help="Question category (e.g. a Polymarket tag) — the calibration source "
        "key. Used to calibrate the aggregate and recorded on the saved run.",
    )
    parser.add_argument(
        "--resolve",
        metavar="RUN_JSON",
        help="Record an outcome on a saved run (pair with --outcome), then exit.",
    )
    parser.add_argument(
        "--outcome",
        type=int,
        choices=(0, 1),
        help="Actual binary outcome for --resolve.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit hierarchical Platt scaling across resolved runs (per question "
        "category), save the fit, then exit.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="L2 weight on per-source offsets for --calibrate (default: 1.0).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Offline modes (no API keys required).
    if args.calibrate:
        print_calibration_report(calibrate_across_sources(lam=args.lam))
        return
    if args.resolve is not None:
        if args.outcome is None:
            parser.error("--resolve requires --outcome 0 or 1")
        path = resolve_run(args.resolve, args.outcome)
        print(f"Recorded outcome={args.outcome} on {path}")
        return

    # Forecast mode.
    if args.prior is not None and not 0.0 <= args.prior <= 1.0:
        parser.error("--prior must be a probability between 0 and 1")

    missing = [
        key for key in ("GEMINI_API_KEY", "BRAVE_API_KEY") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)} "
            f"(expected in {ENV_PATH})"
        )

    result = aggregate_forecasts(
        args.question, prior=args.prior, num_trials=args.trials, category=args.category
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
    if result.calibrated_probability is not None:
        print(
            f"Calibrated probability: {result.calibrated_probability:.3f}   "
            f"(category: {result.category or 'uncategorized'})"
        )
    print(f"Saved to: {saved_path.relative_to(Path.cwd()) if saved_path.is_relative_to(Path.cwd()) else saved_path}")


if __name__ == "__main__":
    main()
