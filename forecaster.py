"""Bayesian Linguistic Forecaster (pluggable LLM + Brave Search).

An agent that estimates the probability of a forecasting question by reasoning
like a superforecaster: it starts from a base rate and performs explicit Bayesian
updates as it gathers web evidence via Brave Search. At every step the model
records its current posterior probability, the supporting/contradicting evidence,
and the next action (search again, read files, or submit) as a single structured
object (`update_belief_and_act`).

The LLM backend is pluggable (see --provider) so cost and quality can be compared
head-to-head: `gemini` uses Google's google-genai SDK with forced function
calling; `deepseek` uses DeepSeek's OpenAI-compatible API with JSON structured
output (DeepSeek V4 rejects forced tool_choice, so we constrain the output shape
instead). The agent loop is provider-agnostic.

To reduce variance, each question is forecast with several independent agent
runs, and the resulting probabilities are combined with a logit-space mean
(averaging in log-odds space rather than probability space).

API keys are loaded from a local .env file (git-ignored) and must include
BRAVE_API_KEY plus the key for the selected provider:
    GEMINI_API_KEY    (--provider gemini, the default)
    DEEPSEEK_API_KEY  (--provider deepseek)

Run with:  uv run forecaster.py
DeepSeek:  uv run forecaster.py --provider deepseek "<question>"
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
#
# The LLM backend (model names, thinking level, per-run step budget, and output
# token cap) is selected at runtime via the provider layer below (see Provider /
# --provider); the knobs here are shared across providers. The step budget is
# per-provider because the cost of a deep run differs sharply by backend — on
# Gemini the per-step context grows so cost climbs steeply with the budget, while
# the cheaper DeepSeek tier can afford to hunt far longer (see Provider.max_steps).

NUM_TRIALS = 5  # independent runs aggregated per question
TEMPERATURE = 1.0  # >0 so the NUM_TRIALS runs genuinely diverge
BRAVE_MAX_RESULTS = 10  # snippets shown per search (full text goes to files, not context)
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Progressive disclosure: search results are written to local files; the agent
# reads chosen ones through a cheaper sub-LLM summarizer (each provider's cheap
# tier) so the main context only ever sees short snippets plus the summaries it
# requested. The summarizer model is per-provider (see Provider.summarizer_model).
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
                    "the event occurs, as a calibrated float between 0.05 and 0.95. "
                    "Be precise (e.g. 0.78, not 'likely')."
                ),
            ),
            "comparison_class": types.Schema(
                type="STRING",
                description=(
                    "Outside view (CHAMPS KNOW: Comparison classes / Outside view): "
                    "the reference class of similar past cases your base rate is "
                    "drawn from, e.g. 'first-term incumbents facing a recession'. "
                    "Set it at step 1 and carry it forward (refine only if you find "
                    "a better reference class)."
                ),
            ),
            "base_rate": types.Schema(
                type="NUMBER",
                minimum=0.05,
                maximum=0.95,
                description=(
                    "The outside-view base rate: the share of your comparison_class "
                    "that resolved YES, BEFORE case-specific evidence. This is your "
                    "anchor; the posterior should only move away from it as far as "
                    "the evidence justifies."
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
                    "facts to extract from the chosen files. If action is 'submit', "
                    "your pre-mortem (the strongest case your forecast is WRONG) plus "
                    "a brief justification of the final probability."
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
            "comparison_class",
            "base_rate",
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
    "You are an elite superforecaster. Predict the probability that a binary "
    "question resolves YES, working with the CHAMPS KNOW discipline from Philip "
    "Tetlock's research on what makes forecasters accurate.\n\n"
    "METHOD — CHAMPS KNOW:\n"
    "- Comparison classes & Outside view (C, O): START from the outside. Identify "
    "a reference class of similar past cases and the rate at which they resolved "
    "YES, BEFORE weighing case-specific detail. Record these as `comparison_class` "
    "and `base_rate`, and treat the base rate as the anchor you adjust away from.\n"
    "- Hunt for information (H): actively dig for primary, hard-to-find evidence — "
    "official data, original documents, domain-specific facts — not punditry or "
    "commentary. Go past the obvious sources.\n"
    "- Adjust often (A): update incrementally and frequently. Make many small, "
    "evidence-proportioned Bayesian updates rather than a few big jumps; a large "
    "swing demands correspondingly strong, corroborated evidence.\n"
    "- Make precise estimates (M): commit to a specific number (e.g. 0.78), never "
    "vague language. Granularity carries information.\n"
    "- Pre-mortem / Post-mortem (P): before submitting, run a pre-mortem — state "
    "the strongest case that your forecast is WRONG and confirm your probability "
    "already prices it in.\n"
    "- Synthesize (S): combine multiple independent perspectives and source types "
    "into one balanced estimate; don't anchor on a single narrative.\n"
    "- No sacred cows (N): set aside ideology, hopes, and prior commitments — "
    "follow only the evidence, even when it cuts against what you'd like to be "
    "true.\n"
    "Your forecast is one of several independent runs that are aggregated and "
    "scored over time by Brier score (Keep score, Wisdom of crowds) — so be "
    "calibrated, not dramatic.\n\n"
    "LOOP:\n"
    "1. Form your `comparison_class` and `base_rate` (the outside view).\n"
    "2. Then loop: at each step call `update_belief_and_act` to record your "
    "current belief state and choose ONE action. After each action your belief "
    "state is updated with what you learned.\n"
    "3. Submit (action='submit') once the estimate has stabilized, further "
    "evidence is unlikely to move it, AND you have run the pre-mortem.\n\n"
    "Belief-state rules:\n"
    "- Carry `comparison_class` and `base_rate` forward on every step (refine the "
    "base rate only if you find a better reference class).\n"
    "- Evidence lists (evidence_for / evidence_against) must ACCUMULATE across "
    "steps: carry forward what still holds and add to it — do not start fresh.\n"
    "- Each evidence item MUST cite its source by file id (e.g. search_1_result_3).\n"
    "- In update_reasoning, explain WHY the latest evidence changed your "
    "probability and by roughly how much (the Bayesian update you just performed).\n"
    "- Weigh the RECENCY and AUTHORITATIVENESS of each source.\n\n"
    f"{NO_MARKETS_RULE}\n\n"
    "Actions (set `action` on update_belief_and_act):\n"
    "- 'web_search': run a focused query (in action_input). You get back short "
    "snippets, each tagged with a file id (e.g. search_1_result_3) — scan them to "
    "spot the most promising sources.\n"
    "- 'read_files': read chosen results in full — put their ids in read_file_ids "
    "and a specific extraction instruction in action_input. Their text is "
    "summarized and returned. Read before trusting a snippet when a result looks "
    "decisive.\n"
    "- 'submit': finalize — put your pre-mortem and a brief justification in "
    "action_input.\n\n"
    "Rules:\n"
    "- You work within a fixed step budget (stated below) and MUST submit before "
    "it runs out.\n"
    "- Keep hunting and adjusting until the estimate stabilizes; on a genuinely "
    "uncertain question, do NOT submit after a shallow look — gather more "
    "evidence first.\n"
    "- Probabilities must be between 0.05 and 0.95; reserve the extremes for "
    "claims backed by strong, corroborated evidence."
)

# Appended to the system prompt for providers that express update_belief_and_act
# as constrained JSON output (rather than a forced tool call) — see DeepSeekSession.
JSON_OUTPUT_INSTRUCTION = (
    "\n\nOUTPUT FORMAT — at every step respond with a SINGLE JSON object and "
    "NOTHING else (no prose, no markdown fences), with exactly these keys:\n"
    '  "probability": number between 0.05 and 0.95 (be precise, e.g. 0.78),\n'
    '  "comparison_class": string — the reference class for your base rate,\n'
    '  "base_rate": number 0.05–0.95 — the outside-view base rate (your anchor),\n'
    '  "confidence": one of "low", "medium", "high",\n'
    '  "evidence_for": array of strings (each citing a file id),\n'
    '  "evidence_against": array of strings (each citing a file id),\n'
    '  "update_reasoning": string — the Bayesian update you just made,\n'
    '  "action": one of "web_search", "read_files", "submit",\n'
    '  "action_input": string — the search query, extraction instruction, or '
    "(for submit) your pre-mortem plus final justification,\n"
    '  "read_file_ids": array of strings — only when action is "read_files".\n'
    "Return only the JSON object."
)


def build_initial_message(
    question: str,
    prior: float | None = None,
    background: str | None = None,
    resolution_criteria: str | None = None,
) -> str:
    """Build the opening user message in the superforecaster question format.

    `background` and `resolution_criteria` are optional context shown under a
    dedicated heading (omitted when both are absent). `prior` is an optional
    external anchor in [0, 1] — a market-implied price or a historical base rate;
    when given the model starts from it and updates only as far as the evidence
    justifies, otherwise it forms its own base rate by reference-class reasoning.
    """
    parts = [f"# Question\n{question}"]

    if background or resolution_criteria:
        section = ["## Background and resolution criteria"]
        if background:
            section.append(background.strip())
        if resolution_criteria:
            section.append(resolution_criteria.strip())
        parts.append("\n".join(section))

    if prior is not None:
        parts.append(
            "## Prior estimate\n"
            f"An external prior estimate for this question is {prior:.0%} (a "
            "market-implied probability or a historical base rate — the wisdom of "
            "the crowd). Use it to inform your `base_rate`, then adjust on "
            "question-specific evidence from search and tools."
        )
    else:
        parts.append(
            "## Prior estimate\n"
            "No prior is given. Build your own `base_rate` from a comparison class "
            "of similar past cases (the outside view) before gathering evidence."
        )

    parts.append(
        "Begin with the outside view: state your comparison_class and base_rate, "
        "then call update_belief_and_act."
    )
    return "\n\n".join(parts)

# --- Providers (pluggable LLM backends) --------------------------------------
#
# The agent needs two operations from a backend: a structured "belief" step (the
# arguments to update_belief_and_act) within a multi-turn Session, and a one-shot
# plain-text completion (file summaries + the final briefing). Each Provider
# implements both; the agent loop never touches an SDK directly, so adding a
# backend means adding a Provider, not editing the loop.

# Token pricing (USD per 1M tokens) as (input, output). Thinking/reasoning tokens
# bill as output; cache-hit discounts are ignored (prompts rarely repeat). Used
# only to estimate run cost — keep roughly in step with each provider's pricing.
PRICING_PER_MTOK = {
    "gemini-3.5-flash": (1.50, 9.00),
    "deepseek-v4-pro": (0.435, 0.87),    # launch promo (~75% off); list ~ 1.74 / 3.48
    "deepseek-v4-flash": (0.14, 0.28),
}


def _new_usage() -> dict:
    return {"prompt": 0, "output": 0, "thinking": 0, "total": 0}


def _merge_usage(acc: dict, delta: dict) -> None:
    """Add one call's normalized usage delta into an accumulator."""
    for key in acc:
        acc[key] += delta.get(key, 0) or 0


def cost_usd(usage: dict | None, model: str) -> float | None:
    """Estimate the USD cost of `usage` tokens billed at `model`'s rate.

    Thinking/reasoning tokens bill as output. Returns None for an unpriced model.
    Note: across a run, summarizer/briefing calls use the provider's cheaper
    `summarizer_model`, so charging everything at the main model's rate is a
    (usually slight) upper bound.
    """
    rates = PRICING_PER_MTOK.get(model)
    if rates is None or not usage:
        return None
    in_rate, out_rate = rates
    billed_output = (usage.get("output", 0) or 0) + (usage.get("thinking", 0) or 0)
    return ((usage.get("prompt", 0) or 0) * in_rate + billed_output * out_rate) / 1_000_000


def _is_permanent_error(exc: Exception) -> bool:
    """True for client errors that won't be fixed by retrying (bad key, no
    balance, bad request, ...): a 4xx HTTP status other than 429 rate-limit.

    Reads the status off whichever attribute the SDK exposes (`status_code` on
    the OpenAI SDK, `code` on google-genai errors)."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429


def _retry(call, *, attempts: int = 4, label: str = "LLM"):
    """Call `call()` with exponential backoff on transient failures.

    A dropped connection or 5xx (e.g. httpx.RemoteProtocolError) shouldn't kill a
    long run, so we retry a few times before giving up. Permanent client errors
    (auth, insufficient balance, bad request) fail fast — retrying can't help.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if _is_permanent_error(exc) or attempt == attempts:
                raise
            wait = 2 ** attempt  # 2, 4, 8 seconds
            logger.warning(
                "%s call failed (attempt %d/%d): %s — retrying in %ds",
                label, attempt, attempts, type(exc).__name__, wait,
            )
            time.sleep(wait)


class Session:
    """A live, multi-turn forecasting conversation with a backend.

    `add_user` appends a user turn; `step` advances one model turn and returns the
    parsed belief (the update_belief_and_act arguments, or None if the model
    returned no structured output) plus that turn's normalized token usage,
    recording the model's turn internally; `add_tool_result` feeds back a search
    or read result. Implementations own the provider-native message format.
    """

    def add_user(self, text: str) -> None:
        raise NotImplementedError

    def step(self) -> tuple[dict | None, dict]:
        raise NotImplementedError

    def add_tool_result(self, text: str) -> None:
        raise NotImplementedError


class Provider:
    """A pluggable LLM backend. Subclasses set the model/key attributes and
    implement `new_session` (the agent loop) and `complete` (one-shot text)."""

    name: str
    model: str             # main reasoning model (the agent loop)
    summarizer_model: str  # cheaper model for file summaries + the final briefing
    thinking_level: str    # recorded on saved runs; meaning is provider-specific
    env_var: str
    max_steps: int = 14          # agent steps per run before a forced submission
    max_output_tokens: int = 8192  # output cap per agent call (incl. thinking room)

    def require_key(self) -> None:
        if not os.environ.get(self.env_var):
            raise RuntimeError(f"{self.env_var} is not set (expected in {ENV_PATH}).")

    def new_session(
        self, *, system_instruction: str, temperature: float, max_output_tokens: int
    ) -> Session:
        raise NotImplementedError

    def complete(
        self, *, model: str, system: str | None, prompt: str,
        temperature: float, max_output_tokens: int, thinking_level: str = "low",
    ) -> tuple[str, dict]:
        raise NotImplementedError


# --- Gemini backend (google-genai SDK, forced function calling) --------------


def _gemini_usage(response) -> dict:
    """Normalize a google-genai response's usage_metadata (best-effort)."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return _new_usage()
    return {
        "prompt": getattr(um, "prompt_token_count", 0) or 0,
        "output": getattr(um, "candidates_token_count", 0) or 0,
        "thinking": getattr(um, "thoughts_token_count", 0) or 0,
        "total": getattr(um, "total_token_count", 0) or 0,
    }


class GeminiSession(Session):
    def __init__(self, provider: "GeminiProvider", config: "types.GenerateContentConfig"):
        self._provider = provider
        self._config = config
        self._contents: list[types.Content] = []

    def add_user(self, text: str) -> None:
        self._contents.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )

    def step(self) -> tuple[dict | None, dict]:
        response = _retry(
            lambda: self._provider.client().models.generate_content(
                model=self._provider.model, contents=self._contents, config=self._config
            ),
            label="Gemini",
        )
        usage = _gemini_usage(response)
        calls = response.function_calls or []
        if not calls:
            return None, usage
        self._contents.append(response.candidates[0].content)  # preserve model turn
        return dict(calls[0].args or {}), usage

    def add_tool_result(self, text: str) -> None:
        # The function_response must echo the (only) tool name so Gemini pairs it
        # with the preceding forced call.
        self._contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=TOOL_NAME, response={"result": text}
                    )
                ],
            )
        )


class GeminiProvider(Provider):
    name = "gemini"
    model = "gemini-3.5-flash"           # latest GA Gemini generation
    summarizer_model = "gemini-3.5-flash"
    thinking_level = "high"              # "high" maximizes reasoning depth; "low" is cheaper
    env_var = "GEMINI_API_KEY"
    max_steps = 14            # kept modest: per-step context grows, so cost climbs fast
    max_output_tokens = 8192

    _client: "genai.Client | None" = None

    def client(self) -> "genai.Client":
        if GeminiProvider._client is None:
            self.require_key()
            GeminiProvider._client = genai.Client(api_key=os.environ[self.env_var])
        return GeminiProvider._client

    def new_session(self, *, system_instruction, temperature, max_output_tokens) -> Session:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            tools=[UPDATE_BELIEF_TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[TOOL_NAME]
                )
            ),
        )
        return GeminiSession(self, config)

    def complete(self, *, model, system, prompt, temperature, max_output_tokens,
                 thinking_level="low") -> tuple[str, dict]:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            **({"system_instruction": system} if system else {}),
        )
        response = _retry(
            lambda: self.client().models.generate_content(
                model=model, contents=prompt, config=config
            ),
            label="Gemini",
        )
        return (response.text or "").strip(), _gemini_usage(response)


# --- DeepSeek backend (OpenAI-compatible API, JSON structured output) --------
#
# DeepSeek V4 runs in thinking mode by default and rejects forced tool_choice
# (HTTP 400 "Thinking mode does not support this tool_choice"), so instead of a
# forced function call we constrain the reply to a JSON object (response_format)
# and parse the belief out of it. Same structured fields, different mechanism.

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _openai_usage(response) -> dict:
    """Normalize an OpenAI-style usage object. `completion_tokens` already
    includes reasoning tokens, so split them so output excludes thinking
    (matching the Gemini convention; both bill as output)."""
    u = getattr(response, "usage", None)
    if u is None:
        return _new_usage()
    completion = getattr(u, "completion_tokens", 0) or 0
    details = getattr(u, "completion_tokens_details", None)
    reasoning = (getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return {
        "prompt": getattr(u, "prompt_tokens", 0) or 0,
        "output": max(0, completion - reasoning),
        "thinking": reasoning,
        "total": getattr(u, "total_tokens", 0) or 0,
    }


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of a JSON object from a model reply (tolerates fences)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, re.DOTALL)  # last resort: outermost {...}
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None


class DeepSeekSession(Session):
    def __init__(self, provider: "DeepSeekProvider", system_instruction: str,
                 temperature: float, max_output_tokens: int):
        self._provider = provider
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._messages: list[dict] = [
            {"role": "system", "content": system_instruction + JSON_OUTPUT_INSTRUCTION}
        ]

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    _REPARSE_NUDGE = (
        "Your previous reply was not a single valid JSON object. Respond now with "
        "ONLY the JSON object described — no prose, no markdown fences."
    )

    def _generate(self, usage: dict, *, nudge: bool = False) -> str:
        """One create() call; merge its usage into `usage` and return the content.

        With `nudge`, a transient corrective instruction is appended for this call
        only (not persisted) — used to re-ask after an unparseable reply."""
        messages = self._messages
        if nudge:
            messages = messages + [{"role": "user", "content": self._REPARSE_NUDGE}]
        response = _retry(
            lambda: self._provider.client().chat.completions.create(
                model=self._provider.model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
                response_format={"type": "json_object"},
            ),
            label="DeepSeek",
        )
        _merge_usage(usage, _openai_usage(response))
        return response.choices[0].message.content or ""

    def step(self) -> tuple[dict | None, dict]:
        usage = _new_usage()
        content = self._generate(usage)
        belief = _parse_json_object(content)
        if belief is None:
            # The reply wasn't a parseable JSON belief (e.g. truncated after heavy
            # reasoning, or stray prose). Re-ask once before giving up.
            logger.warning("DeepSeek reply was not valid JSON; re-asking once")
            content = self._generate(usage, nudge=True)
            belief = _parse_json_object(content)
        if belief is None:
            return None, usage
        self._messages.append({"role": "assistant", "content": content})  # preserve model turn
        return belief, usage

    def add_tool_result(self, text: str) -> None:
        # No real tool call was made (JSON mode), so the result comes back as a
        # plain user turn rather than an OpenAI tool message.
        self._messages.append({"role": "user", "content": text})


class DeepSeekProvider(Provider):
    name = "deepseek"
    model = "deepseek-v4-pro"            # main reasoning model
    summarizer_model = "deepseek-v4-flash"  # cheap tier for summaries + briefing
    thinking_level = "default"           # V4 thinks by default; no high/low knob
    env_var = "DEEPSEEK_API_KEY"
    max_steps = 24            # cheap enough to hunt deep (flat per-step context)
    max_output_tokens = 16384  # extra headroom: V4 reasons heavily before the JSON belief

    _client = None  # openai.OpenAI

    def client(self):
        if DeepSeekProvider._client is None:
            self.require_key()
            from openai import OpenAI  # lazy import: only when this provider is used
            DeepSeekProvider._client = OpenAI(
                api_key=os.environ[self.env_var], base_url=DEEPSEEK_BASE_URL
            )
        return DeepSeekProvider._client

    def new_session(self, *, system_instruction, temperature, max_output_tokens) -> Session:
        return DeepSeekSession(self, system_instruction, temperature, max_output_tokens)

    def complete(self, *, model, system, prompt, temperature, max_output_tokens,
                 thinking_level="low") -> tuple[str, dict]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        response = _retry(
            lambda: self.client().chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_output_tokens,
            ),
            label="DeepSeek",
        )
        return (response.choices[0].message.content or "").strip(), _openai_usage(response)


# --- Provider registry + active selection ------------------------------------

PROVIDERS: dict[str, type[Provider]] = {
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
}
DEFAULT_PROVIDER = "gemini"

_provider: Provider | None = None
_brave_api_key: str | None = None


def set_provider(name: str) -> Provider:
    """Select the active LLM backend by name (see PROVIDERS)."""
    global _provider
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    _provider = PROVIDERS[name]()
    return _provider


def get_provider() -> Provider:
    """Return the active provider, defaulting to DEFAULT_PROVIDER on first use."""
    if _provider is None:
        return set_provider(DEFAULT_PROVIDER)
    return _provider


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
    sources: dict | None = None,
    query: str | None = None,
) -> str:
    """Write each result's full content to a file, register its id, and return
    the short snippet list (ids + title + url + one-line snippet) for context.

    This is the progressive-disclosure split: only the short snippets enter the
    agent's context; the full text lives in `<scratch>/<id>.md` until the agent
    asks to read it. When `sources` is given, each id is recorded as
    {title, url, query} so the run's evidence citations stay resolvable after the
    scratch files are deleted.
    """
    if not results:
        return "No results found."
    lines = []
    for j, result in enumerate(results, start=1):
        file_id = f"search_{search_index}_result_{j}"
        path = scratch / f"{file_id}.md"
        path.write_text(result["full_content"], encoding="utf-8")
        registry[file_id] = path
        if sources is not None:
            sources[file_id] = {
                "title": result["title"],
                "url": result["url"],
                "query": query,
            }
        snippet = result["short_snippet"] or "(no snippet)"
        lines.append(f"[{file_id}] {result['title']}\n{result['url']}\n{snippet}")
    header = (
        f"{len(results)} result(s). To read the full content of the most relevant "
        "ones, call action='read_files' with their ids in read_file_ids and a "
        "focused extraction instruction in action_input.\n"
    )
    return header + "\n\n".join(lines)


def summarize(
    question: str,
    instruction: str,
    docs: list[str],
    resolution_criteria: str | None = None,
    usage: dict | None = None,
) -> str:
    """Summarize selected search-result files with the provider's sub-LLM.

    Acts as an assistant to the superforecaster: it extracts only the facts and
    data from the documents relevant to the question's resolution, without adding
    its own analysis. `instruction` is the main agent's specific extraction
    request (what to pull this time). Returns an error string (never raises) so a
    failure can't kill a run.
    """
    if not docs:
        return "No documents to summarize."
    instruction = instruction.strip() or (
        f"Extract any facts relevant to the forecasting question: {question}"
    )
    criteria = (resolution_criteria or "").strip() or "(not separately specified)"
    combined = "\n\n---\n\n".join(docs)[:SUMMARIZER_MAX_DOC_CHARS]
    provider = get_provider()
    try:
        text, delta = provider.complete(
            model=provider.summarizer_model,
            system=(
                "You are an assistant to a superforecaster. Extract the facts "
                "and data from the search results that are most relevant to "
                "predicting the outcome of this question.\n\n"
                f"Question: {question}\n"
                f"Resolution criteria: {criteria}\n\n"
                "Instructions:\n"
                "- Extract concrete facts, statistics, dates, and named-source "
                "expert opinions.\n"
                "- Note any quantitative data (prices, percentages, counts, "
                "trends).\n"
                "- Distinguish hard facts from speculation or editorial opinion.\n"
                "- Omit information not relevant to the resolution criteria.\n"
                "- Do NOT add your own analysis or forecast — only extract what "
                f"the sources say.\n\n{NO_MARKETS_RULE}"
            ),
            prompt=f"Specifically: {instruction}\n\nSearch results:\n\n{combined}",
            temperature=0.3,
            max_output_tokens=SUMMARIZER_MAX_TOKENS,
            thinking_level="low",
        )
        if usage is not None:
            _merge_usage(usage, delta)
        return text or "Summary was empty."
    except Exception as exc:
        logger.warning("summarization failed: %s", exc)
        return f"Summarization failed: {exc}"


# --- Agent loop --------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def run_read_files(
    question: str,
    belief: dict,
    registry: dict[str, Path],
    resolution_criteria: str | None = None,
    usage: dict | None = None,
) -> str:
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
    summary = summarize(
        question, belief.get("action_input") or "", docs, resolution_criteria, usage=usage
    )
    logger.info("  read_files: %s%s", ", ".join(valid), notes)
    return f"Summary of {', '.join(valid)}{notes}:\n\n{summary}"


def run_agent(
    question: str,
    prior: float | None = None,
    max_steps: int | None = None,
    background: str | None = None,
    resolution_criteria: str | None = None,
) -> dict:
    """Run a single forecasting agent loop and return its final belief.

    At every step the model emits the `update_belief_and_act` fields (a forced
    function call on Gemini, constrained JSON on DeepSeek). On a 'web_search'
    action we run Brave Search, write each result's full text to a per-run scratch
    file, and feed back only short snippets (each tagged with a file id). On a
    'read_files' action we summarize the chosen files via the sub-LLM and feed
    back the summary (progressive disclosure — the heavy text never floods the
    main context). On 'submit' (or when the step budget is exhausted) we return
    the last belief — the recorded fields (probability, evidence_for/against,
    reasoning, ...) with `probability` clamped to [0.05, 0.95].

    If `prior` (a probability in [0, 1]) is given, the agent starts anchored on
    it instead of forming its own base rate (prior injection).
    """
    if prior is not None and not 0.0 <= prior <= 1.0:
        raise ValueError(f"prior must be a probability in [0, 1], got {prior!r}")

    provider = get_provider()
    if max_steps is None:
        max_steps = provider.max_steps

    today = datetime.now().strftime("%Y-%m-%d (%A)")
    session = provider.new_session(
        system_instruction=(
            f"{SYSTEM_PROMPT}\n\n"
            f"Your step budget is {max_steps} steps; you MUST submit before it "
            "runs out.\n"
            f"Today's date is {today}; do not search for the current date."
        ),
        temperature=TEMPERATURE,
        max_output_tokens=provider.max_output_tokens,
    )
    session.add_user(
        build_initial_message(question, prior, background, resolution_criteria)
    )
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

    # Decision-process record: per-step trajectory, source provenance, counters,
    # and token usage — attached to the returned belief for the saved run.
    start = time.perf_counter()
    trajectory: list[dict] = []
    sources: dict = {}
    usage = _new_usage()
    n_reads = 0
    terminated_by = "no_call"

    try:
        for step in range(1, max_steps + 1):
            step_belief, delta = session.step()
            _merge_usage(usage, delta)

            if step_belief is None:
                # No structured output this turn (e.g. the model declined to emit
                # the JSON / tool call); bail with the current estimate.
                logger.warning("step %d: no structured belief returned; stopping early", step)
                break

            belief = step_belief
            probability = _clamp(float(belief.get("probability", probability)), 0.05, 0.95)
            belief["probability"] = probability  # store the clamped value
            action = belief.get("action", "submit")

            try:
                base_str = f"{float(belief['base_rate']):.2f}"
            except (KeyError, TypeError, ValueError):
                base_str = "?"
            logger.info(
                "step %2d/%d | base=%s p=%.3f | confidence=%s | action=%s",
                step,
                max_steps,
                base_str,
                probability,
                belief.get("confidence", "?"),
                action,
            )
            reasoning = belief.get("update_reasoning")
            if reasoning:
                logger.info("  reasoning: %s", reasoning)
            logger.debug("  evidence_for: %s", belief.get("evidence_for", []))
            logger.debug("  evidence_against: %s", belief.get("evidence_against", []))

            trajectory.append(
                {
                    "step": step,
                    "probability": probability,
                    "base_rate": belief.get("base_rate"),
                    "comparison_class": belief.get("comparison_class"),
                    "confidence": belief.get("confidence"),
                    "action": action,
                    "action_input": belief.get("action_input"),
                    "update_reasoning": belief.get("update_reasoning"),
                    "read_file_ids": (
                        list(belief.get("read_file_ids") or [])
                        if action == "read_files"
                        else []
                    ),
                    "n_evidence_for": len(belief.get("evidence_for") or []),
                    "n_evidence_against": len(belief.get("evidence_against") or []),
                }
            )

            # (the model's turn was recorded inside session.step())

            if action == "submit" or step == max_steps:
                terminated_by = "submit" if action == "submit" else "budget"
                if action == "submit":
                    logger.info("  submit: %s", belief.get("action_input", ""))
                else:
                    logger.info(
                        "  step budget (%d) reached; submitting current estimate", max_steps
                    )
                break

            if action == "read_files":
                n_reads += 1
                result = run_read_files(
                    question, belief, registry, resolution_criteria, usage=usage
                )
            else:  # 'web_search' (and any unexpected action): run a new search
                search_index += 1
                query = belief.get("action_input") or question
                results, error = brave_search(query)
                result = error or register_search_results(
                    results, search_index, scratch, registry, sources, query
                )

            session.add_tool_result(result)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # If the model never produced a single parseable belief, there is no forecast
    # to report — only the 0.5 fallback. Treat that as a failed trial (raise) so
    # aggregate_forecasts discards it instead of polluting the mean with a fake
    # 0.5. (A 'no_call' AFTER some good steps keeps the last real belief.)
    if not trajectory:
        raise RuntimeError(
            "agent produced no structured belief (model returned no valid output "
            "on the first step)"
        )

    # Attach the decision-process record (#1 trajectory, #2 provenance, #3 stats).
    belief["steps"] = trajectory
    belief["sources"] = sources
    belief["stats"] = {
        "steps_used": len(trajectory),
        "n_searches": search_index,
        "n_reads": n_reads,
        "terminated_by": terminated_by,
        "seconds": round(time.perf_counter() - start, 1),
    }
    belief["usage"] = usage
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
    usage: dict | None = None  # run-level token totals (all trials + briefing)

    @property
    def trial_probabilities(self) -> list[float]:
        return [float(b["probability"]) for b in self.trials]


def summarize_forecast(
    question: str,
    beliefs: list[dict],
    aggregated: float,
    prior: float | None = None,
    usage: dict | None = None,
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
    provider = get_provider()
    try:
        text, delta = provider.complete(
            model=provider.summarizer_model,
            system=None,
            prompt=prompt,
            temperature=0.3,
            max_output_tokens=2048,
            thinking_level="low",
        )
    except Exception as exc:  # a briefing failure must not lose the forecast
        logger.warning("briefing synthesis failed: %s", exc)
        return f"(briefing unavailable: {exc})"
    if usage is not None:
        _merge_usage(usage, delta)
    return text


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
    background: str | None = None,
    resolution_criteria: str | None = None,
) -> ForecastResult:
    """Run the agent `num_trials` times and combine results via a logit-space mean.

    Averaging in log-odds (logit) space is more principled than averaging raw
    probabilities: it treats evidence additively and is symmetric around 0.5.
    An optional `prior` anchor (a probability in [0, 1]) plus optional `background`
    and `resolution_criteria` context are forwarded to every run. `category`
    (e.g. a Polymarket tag like "politics") is the source key for calibration: if
    a fit has been saved, the aggregate is also calibrated for it. Returns a
    `ForecastResult` with the raw aggregate, the calibrated aggregate (when
    available), the per-trial spread, and a synthesized briefing.
    """
    logger.info(
        "Forecasting in %d trials | provider=%s model=%s | prior=%s | question: %s",
        num_trials,
        get_provider().name,
        get_provider().model,
        f"{prior:.0%}" if prior is not None else "none",
        question,
    )
    overall_start = time.perf_counter()

    total_usage = _new_usage()
    beliefs = []
    for trial in range(1, num_trials + 1):
        logger.info("=== Trial %d/%d ===", trial, num_trials)
        trial_start = time.perf_counter()
        try:
            belief = run_agent(
                question,
                prior=prior,
                background=background,
                resolution_criteria=resolution_criteria,
            )
        except Exception as exc:  # one bad trial shouldn't lose the others
            logger.warning("Trial %d/%d failed, skipping: %s", trial, num_trials, exc)
            continue
        beliefs.append(belief)
        for key in total_usage:
            total_usage[key] += belief.get("usage", {}).get(key, 0)
        logger.info(
            "Trial %d/%d result: p=%.3f (%.1fs, %d tokens)",
            trial,
            num_trials,
            belief["probability"],
            time.perf_counter() - trial_start,
            belief.get("usage", {}).get("total", 0),
        )

    if not beliefs:
        raise RuntimeError(f"all {num_trials} trials failed for {question!r}")
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
    summary = summarize_forecast(question, beliefs, aggregated, prior, usage=total_usage)
    logger.info(
        "Total tokens: %d (prompt %d, output %d, thinking %d)",
        total_usage["total"],
        total_usage["prompt"],
        total_usage["output"],
        total_usage["thinking"],
    )
    return ForecastResult(
        probability=aggregated,
        trials=beliefs,
        summary=summary,
        calibrated_probability=calibrated,
        category=category,
        usage=total_usage,
    )


# --- Persistence -------------------------------------------------------------


def save_run(
    question: str,
    result: ForecastResult,
    prior: float | None = None,
    num_trials: int | None = None,
    background: str | None = None,
    resolution_criteria: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a forecast run to RUNS_DIR as JSON so the decision is not lost.

    `extra` is merged into the payload — used e.g. to attach a `market` reference
    (platform, slug, condition_id) so the outcome can be resolved automatically.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60] or "forecast"
    path = RUNS_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    payload = {
        "timestamp": now.isoformat(),
        "question": question,
        "background": background,
        "resolution_criteria": resolution_criteria,
        "category": result.category,  # calibration source key
        "outcome": None,  # fill in via --resolve once the question resolves
        "prior": prior,
        "provider": get_provider().name,
        "model": get_provider().model,
        "summarizer_model": get_provider().summarizer_model,
        "thinking_level": get_provider().thinking_level,
        "num_trials": num_trials if num_trials is not None else len(result.trials),
        "probability": result.probability,  # raw aggregate (feeds calibration fits)
        "calibrated_probability": result.calibrated_probability,
        "trial_probabilities": result.trial_probabilities,
        "usage": result.usage,  # run-level token totals
        "est_cost_usd": cost_usd(result.usage, get_provider().model),
        "summary": result.summary,
        "trials": result.trials,
        **(extra or {}),
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
        "--provider",
        choices=sorted(PROVIDERS),
        default=DEFAULT_PROVIDER,
        help=f"LLM backend (default: {DEFAULT_PROVIDER}). 'deepseek' needs "
        "DEEPSEEK_API_KEY; 'gemini' needs GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Question category (e.g. a Polymarket tag) — the calibration source "
        "key. Used to calibrate the aggregate and recorded on the saved run.",
    )
    parser.add_argument(
        "--background",
        default=None,
        help="Optional background context shown to the forecaster under the "
        "'Background and resolution criteria' heading.",
    )
    parser.add_argument(
        "--resolution-criteria",
        default=None,
        help="Optional resolution criteria — how the question settles YES/NO. "
        "Shown to the forecaster and the search summarizer.",
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

    provider = set_provider(args.provider)
    missing = [
        key for key in (provider.env_var, "BRAVE_API_KEY") if not os.environ.get(key)
    ]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)} "
            f"(expected in {ENV_PATH})"
        )

    result = aggregate_forecasts(
        args.question,
        prior=args.prior,
        num_trials=args.trials,
        category=args.category,
        background=args.background,
        resolution_criteria=args.resolution_criteria,
    )
    saved_path = save_run(
        args.question,
        result,
        prior=args.prior,
        num_trials=args.trials,
        background=args.background,
        resolution_criteria=args.resolution_criteria,
    )

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
    u = result.usage or {}
    est = cost_usd(u, provider.model)
    cost_str = f"≈ ${est:.4f}" if est is not None else "n/a"
    print(
        f"Provider: {provider.name} ({provider.model})   "
        f"tokens: {u.get('total', 0):,} (prompt {u.get('prompt', 0):,}, "
        f"output {u.get('output', 0):,}, thinking {u.get('thinking', 0):,})   "
        f"est. cost: {cost_str}"
    )
    print(f"Saved to: {saved_path.relative_to(Path.cwd()) if saved_path.is_relative_to(Path.cwd()) else saved_path}")


if __name__ == "__main__":
    main()
