"""Build a calibration set from World Cup match-winner markets.

Match-winner markets resolve within hours, so they're an ideal fast-feedback
dataset for calibrating the forecaster. This driver:

  1. Fetches upcoming FIFA World Cup matches from Polymarket (Gamma API) and
     pulls each match's binary "Will {team} win?" markets.
  2. Forecasts each one INDEPENDENTLY (no market prior — the agent forms its own
     view; markets are ignored per the no-markets rule), saving a run tagged with
     a calibration category and a reference to its Polymarket market.
  3. Auto-resolves: once a match settles, re-fetches its market from Gamma and
     records the actual 0/1 outcome on the saved run.

Then `forecaster.py --calibrate` fits hierarchical Platt scaling over the runs.

Run with:
    uv run train_worldcup.py --days 3 --dry-run        # preview, no LLM calls
    uv run train_worldcup.py --days 3 --limit 4         # forecast next 4 matches
    uv run train_worldcup.py --resolve                  # record settled outcomes
    uv run forecaster.py --calibrate                    # fit once enough resolved
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import forecaster
import polymarket

logger = logging.getLogger("train_worldcup")

WC_TAG = "fifa-world-cup"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_base_match(event: dict) -> bool:
    """A per-match moneyline event ("A vs. B"), not a "- More Markets" spin-off."""
    title = event.get("title") or ""
    return " vs. " in title and " - " not in title and bool(event.get("markets"))


def _market_kind(market: dict) -> str | None:
    """Classify a match market by its question: 'win', 'draw', or None."""
    q = (market.get("question") or "").lower()
    if "draw" in q:
        return "draw"
    if "win" in q:
        return "win"
    return None


def _yes_price(market: dict) -> float | None:
    """The market-implied probability of the 'Yes' outcome (None if unparseable)."""
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = json.loads(market.get("outcomePrices") or "[]")
    except (json.JSONDecodeError, TypeError):
        return None
    for name, price in zip(outcomes, prices):
        if str(name).lower() == "yes":
            try:
                return float(price)
            except (TypeError, ValueError):
                return None
    return None


def _resolved_outcome(market: dict) -> int | None:
    """The settled 0/1 outcome of a binary market, or None if not yet resolved."""
    if not market.get("closed"):
        return None
    p = _yes_price(market)
    if p is None or abs(p - round(p)) >= 0.05:  # not a clean 0/1 settlement
        return None
    return int(round(p))


def fetch_match_markets(
    within_days: int,
    min_liquidity: float,
    include_draws: bool = False,
    limit_matches: int | None = None,
) -> list[dict]:
    """Return forecasting tasks for upcoming World Cup match-winner markets.

    Each task: {question, background, resolution_criteria, market_price, and a
    `market` reference (platform, event_slug, condition_id, question)}.
    """
    now = datetime.now(timezone.utc)
    filters = {
        "active": "true",
        "closed": "false",
        "tag_slug": WC_TAG,
        "end_date_min": _iso(now),
        "end_date_max": _iso(now + timedelta(days=within_days)),
        "liquidity_min": min_liquidity,
        "order": "endDate",
        "ascending": "true",
    }
    matches = [e for e in polymarket.fetch_events_raw(filters) if _is_base_match(e)]
    if limit_matches:
        matches = matches[:limit_matches]

    tasks = []
    for event in matches:
        for market in event.get("markets") or []:
            kind = _market_kind(market)
            if kind == "win" or (include_draws and kind == "draw"):
                tasks.append(
                    {
                        "question": market.get("question", ""),
                        "background": (
                            f"{event.get('title')} — a 2026 FIFA World Cup match "
                            f"(scheduled {event.get('endDate')})."
                        ),
                        "resolution_criteria": market.get("description"),
                        "market_price": _yes_price(market),
                        "market": {
                            "platform": "polymarket",
                            "event_slug": event.get("slug"),
                            "condition_id": market.get("conditionId"),
                            "question": market.get("question"),
                        },
                    }
                )
    return tasks


def forecast_tasks(tasks: list[dict], trials: int, category: str) -> list:
    """Forecast each task independently and save a run per market."""
    saved = []
    for i, task in enumerate(tasks, start=1):
        mp = task["market_price"]
        logger.info(
            "[%d/%d] %s (market %s)",
            i, len(tasks), task["question"],
            f"{mp:.0%}" if mp is not None else "n/a",
        )
        result = forecaster.aggregate_forecasts(
            task["question"],
            prior=None,  # independent: the agent forms its own view
            num_trials=trials,
            category=category,
            background=task["background"],
            resolution_criteria=task["resolution_criteria"],
        )
        path = forecaster.save_run(
            task["question"],
            result,
            prior=None,
            num_trials=trials,
            background=task["background"],
            resolution_criteria=task["resolution_criteria"],
            extra={"market": task["market"], "market_price": mp},
        )
        saved.append(path)
        logger.info(
            "  -> agent p=%.3f vs market %s | saved %s",
            result.probability,
            f"{mp:.3f}" if mp is not None else "n/a",
            path.name,
        )
    return saved


def resolve_pending() -> None:
    """Record actual outcomes on pending Polymarket runs whose markets have settled."""
    pending = []
    for path in sorted(forecaster.RUNS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        market = data.get("market") or {}
        if data.get("outcome") is None and market.get("platform") == "polymarket":
            pending.append((path, market))

    if not pending:
        logger.info("No pending Polymarket runs to resolve.")
        return

    by_slug: dict[str, list] = {}
    for path, market in pending:
        by_slug.setdefault(market.get("event_slug"), []).append((path, market))

    resolved = still_pending = 0
    for slug, items in by_slug.items():
        events = polymarket.fetch_events_raw({"slug": slug}) if slug else []
        markets = {
            m.get("conditionId"): m
            for e in events
            for m in (e.get("markets") or [])
        }
        for path, ref in items:
            gamma_market = markets.get(ref.get("condition_id"))
            outcome = _resolved_outcome(gamma_market) if gamma_market else None
            if outcome is None:
                still_pending += 1
                continue
            forecaster.resolve_run(path, outcome)
            resolved += 1
            logger.info("  resolved %s -> outcome=%d", path.name, outcome)
    logger.info("Resolved %d run(s); %d still pending.", resolved, still_pending)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast upcoming World Cup match-winner markets to build a "
        "calibration set, and auto-resolve them from Polymarket."
    )
    parser.add_argument("--days", type=int, default=3,
                        help="matches resolving within this many days (default: 3)")
    parser.add_argument("--min-liquidity", type=float, default=50000,
                        help="minimum event liquidity in USD (default: 50000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of matches forecast")
    parser.add_argument("--trials", type=int, default=3,
                        help="independent runs aggregated per market (default: 3)")
    parser.add_argument("--category", default="world-cup",
                        help="calibration category recorded on each run (default: world-cup)")
    parser.add_argument("--include-draws", action="store_true",
                        help="also forecast each match's draw market")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the markets that would be forecast; make no LLM calls")
    parser.add_argument("--resolve", action="store_true",
                        help="auto-resolve pending Polymarket runs from Gamma, then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.resolve:
        resolve_pending()
        return

    tasks = fetch_match_markets(
        args.days, args.min_liquidity, args.include_draws, args.limit
    )
    logger.info("Found %d match-winner market(s) within %d day(s).", len(tasks), args.days)

    if args.dry_run or not tasks:
        for task in tasks:
            mp = task["market_price"]
            print(f"- {task['question']}  (market {mp:.0%})  [{task['market']['event_slug']}]"
                  if mp is not None else f"- {task['question']}  [{task['market']['event_slug']}]")
        if args.dry_run:
            print(f"\n{len(tasks)} market(s) × {args.trials} trial(s) would run. "
                  "(dry run — nothing forecast)")
        return

    missing = [k for k in ("GEMINI_API_KEY", "BRAVE_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")

    forecast_tasks(tasks, args.trials, args.category)
    logger.info("Done. Resolve later with: uv run train_worldcup.py --resolve")


if __name__ == "__main__":
    main()
