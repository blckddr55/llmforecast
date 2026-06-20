"""Build a calibration set from Polymarket markets, selected by tag.

Generalizes the World Cup trainer to any topic. Fetch events carrying given
Polymarket tag(s), forecast each competitive binary (Yes/No) market
INDEPENDENTLY (no market prior — markets are ignored as evidence), and save a run
per market tagged with a single calibration category. Auto-resolve once the
markets settle, then `forecaster.py --calibrate`.

Pool several tags into ONE calibration category — e.g. politics + geopolitics:

    uv run train_markets.py --tags politics,geopolitics --category politics --dry-run
    uv run train_markets.py --tags politics,geopolitics --category politics --limit 10
    uv run train_markets.py --resolve
    uv run forecaster.py --calibrate

This module also holds the shared forecast/resolve machinery reused by the World
Cup preset (`train_worldcup.py`).
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import forecaster
import polymarket

logger = logging.getLogger("train_markets")


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _yes_price(market: dict) -> float | None:
    """Market-implied probability of the 'Yes' outcome (None if unparseable)."""
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


def _is_binary(market: dict) -> bool:
    """True if the market is a Yes/No binary."""
    try:
        outcomes = [str(o).lower() for o in json.loads(market.get("outcomes") or "[]")]
    except (json.JSONDecodeError, TypeError):
        return False
    return sorted(outcomes) == ["no", "yes"]


def _resolved_outcome(market: dict) -> int | None:
    """Settled 0/1 outcome of a binary market, or None if not yet resolved."""
    if not market.get("closed"):
        return None
    p = _yes_price(market)
    if p is None or abs(p - round(p)) >= 0.05:  # not a clean 0/1 settlement
        return None
    return int(round(p))


def make_task(event: dict, market: dict) -> dict:
    """A forecasting task: question + context + a market reference for resolving."""
    return {
        "question": market.get("question", ""),
        "background": f"{event.get('title')} (resolves by {event.get('endDate')}).",
        "resolution_criteria": market.get("description"),
        "market_price": _yes_price(market),
        "market": {
            "platform": "polymarket",
            "event_slug": event.get("slug"),
            "condition_id": market.get("conditionId"),
            "question": market.get("question"),
        },
    }


def _event_tag_slugs(event: dict) -> set:
    """The set of tag slugs an event carries."""
    return {t.get("slug") for t in event.get("tags") or []}


def fetch_tag_markets(
    tags: list[str],
    within_days: int,
    min_liquidity: float,
    min_price: float = 0.05,
    max_price: float = 0.95,
    max_per_event: int = 5,
    exclude_tags: list[str] | None = None,
    exclude_pattern: str | None = None,
    max_liquidity: float | None = None,
) -> list[dict]:
    """Fetch competitive binary markets from events carrying any of `tags`.

    Events are fetched per tag (server-side) and de-duplicated. An event carrying
    any slug in `exclude_tags` is dropped (e.g. "tweets-markets" novelty markets).
    A market whose question (or its event title) matches the `exclude_pattern`
    regex is dropped too (e.g. "publicly insult" novelty markets). Within each
    kept event we keep open binary Yes/No markets whose price is in
    [min_price, max_price] (skipping near-decided ones), most-liquid first up to
    `max_per_event`. Returns forecasting tasks (see `make_task`); the caller
    de-duplicates against runs/ and applies any `--limit`.
    """
    now = datetime.now(timezone.utc)
    base = {
        "active": "true",
        "closed": "false",
        "end_date_min": _iso(now),
        "end_date_max": _iso(now + timedelta(days=within_days)),
        "liquidity_min": min_liquidity,
        "order": "liquidity",
        "ascending": "false",
    }
    if max_liquidity is not None:
        base["liquidity_max"] = max_liquidity
    excluded = set(exclude_tags or [])
    pattern = re.compile(exclude_pattern, re.IGNORECASE) if exclude_pattern else None
    events: dict = {}
    for tag in tags:
        for e in polymarket.fetch_events_raw({**base, "tag_slug": tag}):
            events.setdefault(e.get("id"), e)

    tasks = []
    for event in events.values():
        if excluded & _event_tag_slugs(event):
            continue
        title = event.get("title") or ""
        eligible = []
        for market in event.get("markets") or []:
            if market.get("closed") or not _is_binary(market):
                continue
            if pattern and (pattern.search(market.get("question") or "") or pattern.search(title)):
                continue
            price = _yes_price(market)
            if price is None or not (min_price <= price <= max_price):
                continue
            eligible.append(market)
        eligible.sort(key=lambda m: _to_float(m.get("liquidityNum")), reverse=True)
        tasks.extend(make_task(event, m) for m in eligible[:max_per_event])
    return tasks


def list_tags(
    tags: list[str],
    within_days: int,
    min_liquidity: float,
    max_liquidity: float | None = None,
) -> None:
    """Print the tags present on matching events, with event counts.

    With no `tags`, lists every tag across active, liquid events in the window —
    the universe of slugs you can pass to --tags / --exclude-tags. With `tags`
    given, lists the tags that co-occur with those (useful for finding slugs to
    exclude). Discovery only: price/exclude filters are not applied.
    """
    now = datetime.now(timezone.utc)
    base = {
        "active": "true",
        "closed": "false",
        "end_date_min": _iso(now),
        "end_date_max": _iso(now + timedelta(days=within_days)),
        "liquidity_min": min_liquidity,
        "order": "liquidity",
        "ascending": "false",
    }
    if max_liquidity is not None:
        base["liquidity_max"] = max_liquidity

    events: dict = {}
    queries = [{**base, "tag_slug": t} for t in tags] if tags else [base]
    for params in queries:
        for e in polymarket.fetch_events_raw(params):
            events.setdefault(e.get("id"), e)

    counts: Counter = Counter()
    labels: dict = {}
    for e in events.values():
        for t in e.get("tags") or []:
            slug = t.get("slug")
            if slug:
                counts[slug] += 1
                labels[slug] = t.get("label") or ""
    scope = f"co-occurring with {tags}" if tags else "across active, liquid events"
    print(f"{len(counts)} tag(s) {scope} ({len(events)} event(s)):")
    print(f"  {'events':>6}  {'slug':<34} label")
    for slug, n in counts.most_common():
        print(f"  {n:>6}  {slug:<34} {labels.get(slug, '')}")


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
            task["question"], result,
            prior=None, num_trials=trials,
            background=task["background"], resolution_criteria=task["resolution_criteria"],
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
    """Record actual outcomes on pending Polymarket runs whose markets have settled.

    Covers every pending Polymarket run regardless of category, so one call
    resolves sports and politics runs alike.
    """
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

    by_slug: dict = {}
    for path, market in pending:
        by_slug.setdefault(market.get("event_slug"), []).append((path, market))

    resolved = still_pending = 0
    for slug, items in by_slug.items():
        events = polymarket.fetch_events_raw({"slug": slug}) if slug else []
        markets = {
            m.get("conditionId"): m for e in events for m in (e.get("markets") or [])
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


def forecasted_condition_ids() -> set:
    """Condition ids of markets already saved to runs/ (any status).

    Used to avoid re-forecasting a market we've already recorded for calibration.
    Resolved runs are included — once forecast, a market is never re-run (it is
    still picked up by `resolve_pending`, which scans runs/ independently).
    """
    seen = set()
    for path in forecaster.RUNS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cid = (data.get("market") or {}).get("condition_id")
        if cid:
            seen.add(cid)
    return seen


def dedupe_tasks(tasks: list[dict]) -> list[dict]:
    """Drop tasks whose market already has a saved run, plus any in-batch repeats."""
    seen = forecasted_condition_ids()
    out = []
    for task in tasks:
        cid = task["market"].get("condition_id")
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        out.append(task)
    return out


def cap_to_events(tasks: list[dict], n: int) -> list[dict]:
    """Keep all tasks belonging to the first `n` distinct events (e.g. matches)."""
    out, events = [], []
    for task in tasks:
        slug = task["market"].get("event_slug")
        if slug not in events:
            if len(events) >= n:
                break
            events.append(slug)
        out.append(task)
    return out


def print_tasks(tasks: list[dict], trials: int, category: str) -> None:
    """Print the markets that would be forecast (for --dry-run / empty results)."""
    for task in tasks:
        mp = task["market_price"]
        slug = task["market"]["event_slug"]
        print(f"- {task['question']}  (market {mp:.0%})  [{slug}]"
              if mp is not None else f"- {task['question']}  [{slug}]")
    print(f"\n{len(tasks)} market(s) × {trials} trial(s), category={category}. "
          "(dry run — nothing forecast)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast Polymarket markets by tag to build a calibration "
        "set, and auto-resolve them from Polymarket."
    )
    parser.add_argument("--tags", default="",
                        help="comma-separated Polymarket tag slugs, e.g. politics,geopolitics")
    parser.add_argument("--category", default=None,
                        help="calibration category for every run (default: the first tag)")
    parser.add_argument("--days", type=int, default=30,
                        help="markets resolving within this many days (default: 30)")
    parser.add_argument("--min-liquidity", type=float, default=50000,
                        help="minimum event liquidity in USD (default: 50000)")
    parser.add_argument("--max-liquidity", type=float, default=None,
                        help="maximum event liquidity in USD (default: no cap)")
    parser.add_argument("--min-price", type=float, default=0.05,
                        help="skip markets priced below this — near-decided (default: 0.05)")
    parser.add_argument("--max-price", type=float, default=0.95,
                        help="skip markets priced above this — near-decided (default: 0.95)")
    parser.add_argument("--max-per-event", type=int, default=5,
                        help="most-liquid markets to take per event (default: 5)")
    parser.add_argument("--exclude-tags", default="tweets-markets",
                        help="comma-separated tag slugs whose events to drop "
                        "(default: tweets-markets; pass '' to keep everything)")
    parser.add_argument("--exclude-pattern", default="publicly insult",
                        help="drop markets whose question/title matches this "
                        "case-insensitive regex (default: 'publicly insult' novelty "
                        "markets; pass '' to keep everything)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of NEW markets forecast this run "
                        "(applied after de-duplicating against runs/)")
    parser.add_argument("--trials", type=int, default=5,
                        help="independent runs aggregated per market (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the markets that would be forecast; make no LLM calls")
    parser.add_argument("--list-tags", action="store_true",
                        help="list the tags on matching events (with counts), then exit; "
                        "works with or without --tags")
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

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.list_tags:
        list_tags(tags, args.days, args.min_liquidity, args.max_liquidity)
        return
    if not tags:
        parser.error("--tags is required (e.g. --tags politics,geopolitics)")
    category = args.category or tags[0]

    exclude_tags = [t.strip() for t in args.exclude_tags.split(",") if t.strip()]
    tasks = fetch_tag_markets(
        tags, args.days, args.min_liquidity,
        args.min_price, args.max_price, args.max_per_event,
        exclude_tags=exclude_tags,
        exclude_pattern=args.exclude_pattern or None,
        max_liquidity=args.max_liquidity,
    )
    found = len(tasks)
    tasks = dedupe_tasks(tasks)
    new_count = len(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info(
        "Found %d market(s) for tags=%s -> category=%s | %d new "
        "(%d already in runs/), forecasting %d.",
        found, tags, category, new_count, found - new_count, len(tasks),
    )

    if args.dry_run or not tasks:
        print_tasks(tasks, args.trials, category)
        return

    missing = [k for k in ("GEMINI_API_KEY", "BRAVE_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")

    forecast_tasks(tasks, args.trials, category)
    logger.info("Done. Resolve later with: uv run train_markets.py --resolve")


if __name__ == "__main__":
    main()
