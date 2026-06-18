"""Polymarket market discovery (Gamma API).

Fetch open Polymarket *events* with at least a given liquidity that resolve
within a given number of days — a candidate list to screen for forecasting.

Read-only and unauthenticated: the Gamma `/events` endpoint is public. It sits
behind Cloudflare, which rejects the default `Python-urllib` User-Agent, so we
send a browser one. No API key is required.

Run with:
    uv run polymarket.py --min-liquidity 50000 --days 7
    uv run polymarket.py --min-liquidity 50000 --days 7 --limit 20 --json
"""

import argparse
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("polymarket")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
# Gamma is behind Cloudflare, which 403s the default Python-urllib User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_PAGE_LIMIT = 100  # Gamma's max page size


def _iso(dt: datetime) -> str:
    """Format a UTC datetime the way the Gamma date filters expect."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_prices(market: dict) -> list[tuple[str, float]]:
    """Parse a market's JSON-encoded `outcomes`/`outcomePrices` into (name, price).

    Each pair is the market-implied probability of that outcome (Gamma encodes
    both arrays as JSON strings, e.g. '["Yes", "No"]' / '["0.17", "0.83"]').
    """
    try:
        outcomes = json.loads(market.get("outcomes") or "[]")
        prices = json.loads(market.get("outcomePrices") or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [(name, _to_float(price)) for name, price in zip(outcomes, prices)]


def fetch_markets(
    min_liquidity: float,
    within_days: int,
    max_results: int | None = None,
) -> list[dict]:
    """Fetch open events with liquidity >= `min_liquidity` ending within `within_days`.

    Hits the public Gamma `/events` endpoint, filtering server-side by liquidity
    and an end-date window [now, now + within_days] and ordering by liquidity
    (descending), paginating until exhausted (or `max_results` is reached).

    Returns one dict per event: `id`, `slug`, `title`, `endDate`, `liquidity`,
    `volume`, and `markets` — a list of {question, prices: [(outcome, prob), ...]}.
    """
    now = datetime.now(timezone.utc)
    base = {
        "active": "true",
        "closed": "false",
        "liquidity_min": min_liquidity,
        "end_date_min": _iso(now),
        "end_date_max": _iso(now + timedelta(days=within_days)),
        "order": "liquidity",
        "ascending": "false",
        "limit": _PAGE_LIMIT,
    }

    raw: list[dict] = []
    offset = 0
    while True:
        url = f"{GAMMA_EVENTS_URL}?{urllib.parse.urlencode({**base, 'offset': offset})}"
        request = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(request, timeout=30) as resp:
            page = json.load(resp)
        if not page:
            break
        raw.extend(page)
        if len(page) < _PAGE_LIMIT or (max_results and len(raw) >= max_results):
            break
        offset += _PAGE_LIMIT

    if max_results:
        raw = raw[:max_results]

    events = [
        {
            "id": e.get("id"),
            "slug": e.get("slug"),
            "title": e.get("title"),
            "endDate": e.get("endDate"),
            "liquidity": _to_float(e.get("liquidity")),
            "volume": _to_float(e.get("volume")),
            "markets": [
                {"question": m.get("question", ""), "prices": _parse_prices(m)}
                for m in (e.get("markets") or [])
            ],
        }
        for e in raw
    ]
    logger.info(
        "fetched %d event(s) | liquidity >= $%s | ending within %d day(s)",
        len(events),
        f"{min_liquidity:,.0f}",
        within_days,
    )
    return events


def _format_event(event: dict, max_markets: int = 6) -> str:
    """Render one event as a compact, readable block for the CLI listing."""
    lines = [
        f"{event['title']}  [{event['slug']}]",
        f"  ends {event['endDate']} | liquidity ${event['liquidity']:,.0f} "
        f"| volume ${event['volume']:,.0f}",
    ]
    markets = event["markets"]
    for m in markets[:max_markets]:
        priced = ", ".join(f"{name} {prob:.0%}" for name, prob in m["prices"])
        lines.append(f"    - {m['question']}: {priced}" if priced else f"    - {m['question']}")
    if len(markets) > max_markets:
        lines.append(f"    … and {len(markets) - max_markets} more market(s)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List open Polymarket events by liquidity and time-to-resolution."
    )
    parser.add_argument(
        "--min-liquidity", type=float, default=10000,
        help="minimum event liquidity in USD (default: 10000)",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="only events ending within this many days (default: 7)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of events returned (default: no cap)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit raw JSON instead of the readable listing",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    events = fetch_markets(args.min_liquidity, args.days, max_results=args.limit)
    if args.json:
        print(json.dumps(events, indent=2))
        return
    if not events:
        print("No matching events.")
        return
    for event in events:
        print(_format_event(event))
        print()


if __name__ == "__main__":
    main()
