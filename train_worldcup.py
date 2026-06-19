"""Build a calibration set from World Cup match-winner markets (sports preset).

A curated preset of `train_markets.py`: instead of every competitive binary
market under a tag, it forecasts each upcoming FIFA World Cup match's
"Will {team} win?" markets (and draws with --include-draws), tagged --category
world-cup. The forecast/save and auto-resolve machinery is shared with
`train_markets.py`.

Run with:
    uv run train_worldcup.py --days 3 --dry-run        # preview, no LLM calls
    uv run train_worldcup.py --days 3 --limit 4         # forecast next 4 matches
    uv run train_worldcup.py --resolve                  # record settled outcomes
    uv run forecaster.py --calibrate                    # fit once enough resolved
"""

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone

import polymarket
from train_markets import (
    _iso,
    _yes_price,
    cap_to_events,
    dedupe_tasks,
    forecast_tasks,
    print_tasks,
    resolve_pending,
)

logger = logging.getLogger("train_worldcup")

WC_TAG = "fifa-world-cup"


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


def fetch_match_markets(
    within_days: int,
    min_liquidity: float,
    include_draws: bool = False,
) -> list[dict]:
    """Return forecasting tasks for upcoming World Cup match-winner markets."""
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

    tasks = fetch_match_markets(args.days, args.min_liquidity, args.include_draws)
    found = len(tasks)
    tasks = dedupe_tasks(tasks)
    new_count = len(tasks)
    if args.limit:
        tasks = cap_to_events(tasks, args.limit)  # --limit counts matches
    logger.info(
        "Found %d match-winner market(s) within %d day(s) | %d new "
        "(%d already in runs/), forecasting %d.",
        found, args.days, new_count, found - new_count, len(tasks),
    )

    if args.dry_run or not tasks:
        print_tasks(tasks, args.trials, args.category)
        return

    missing = [k for k in ("GEMINI_API_KEY", "BRAVE_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")

    forecast_tasks(tasks, args.trials, args.category)
    logger.info("Done. Resolve later with: uv run train_worldcup.py --resolve")


if __name__ == "__main__":
    main()
