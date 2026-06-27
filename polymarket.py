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
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("polymarket")

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_COMMENTS_URL = "https://gamma-api.polymarket.com/comments"
# Gamma is behind Cloudflare, which 403s the default Python-urllib User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
_PAGE_LIMIT = 100  # Gamma's max page size

# Polymarket comment threads are heavy with telegram-pump / marketing spam; a
# profile bio (or body) matching this is dropped wholesale.
_SPAM_RE = re.compile(r"tele\W?gram|t\.me/|whats\W?app|\bpump\b|join my|\bspam\b", re.I)


def _parse_dt(value) -> datetime | None:
    """Parse a Gamma ISO timestamp (e.g. '2026-07-20T00:00:00Z') to aware UTC.

    Returns None on a missing or unparseable value so callers can skip it.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def market_in_window(
    market: dict, event_end: str | None, now: datetime, horizon: datetime
) -> bool:
    """True if `market` is open and resolves within [now, horizon].

    The date check is authoritative PER MARKET: a "by <date>?" ladder bundles
    staggered sub-markets under one event whose `endDate` is unreliable (it can be
    earlier OR later than the open rungs), so it must not gate the rungs. Uses the
    market's own `endDate`, falling back to the event's only if absent, and drops
    a market that is individually `closed` even when its parent event is open.
    """
    if market.get("closed"):
        return False
    end = _parse_dt(market.get("endDate") or event_end)
    return end is not None and now <= end <= horizon


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


def fetch_events_raw(params: dict, max_results: int | None = None) -> list[dict]:
    """Page through the Gamma `/events` endpoint and return the raw event dicts.

    `params` are the query filters (active, closed, tag_slug, liquidity_min,
    end_date_min/max, slug, order, ...); pagination (limit/offset) is handled
    here. Returns the unmodified Gamma event objects — full market fields and all.
    """
    base = {**params, "limit": _PAGE_LIMIT}
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
    return raw[:max_results] if max_results else raw


def fetch_markets(
    min_liquidity: float,
    within_days: int,
    max_results: int | None = None,
    tag_slug: str | None = None,
) -> list[dict]:
    """Fetch open events whose markets resolve within `within_days` (liquidity >= `min_liquidity`).

    Hits the public Gamma `/events` endpoint, filtering server-side by liquidity
    and ordering by liquidity (descending), paginating until exhausted. If
    `tag_slug` is given (e.g. "politics", "sports", "crypto"), only events carrying
    that tag are returned (also filtered server-side).

    The date filter is applied entirely PER MARKET (see `market_in_window`): within
    each event, only open markets whose own `endDate` falls in [now, now +
    within_days] are kept, and an event left with no qualifying market is dropped.
    There is deliberately NO server-side event-date window — an event's `endDate`
    is unreliable for "by <date>?" ladders (it can be earlier OR later than the
    open rungs), so any event-level window hides rungs that actually resolve in
    range. `max_results` caps the returned events AFTER this filter (so it counts
    in-window events, not raw ones).

    Returns one dict per event: `id`, `slug`, `title`, `tags` (label list),
    `endDate`, `liquidity`, `volume`, and `markets` — a list of
    {question, prices: [(outcome, prob), ...]}.
    """
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=within_days)
    filters = {
        "active": "true",
        "closed": "false",
        "liquidity_min": min_liquidity,
        "order": "liquidity",
        "ascending": "false",
    }
    if tag_slug:
        filters["tag_slug"] = tag_slug

    events = []
    for e in fetch_events_raw(filters):
        markets = [
            {"question": m.get("question", ""), "prices": _parse_prices(m)}
            for m in (e.get("markets") or [])
            if market_in_window(m, e.get("endDate"), now, horizon)
        ]
        if not markets:
            continue  # nothing in this event resolves within the window
        events.append(
            {
                "id": e.get("id"),
                "slug": e.get("slug"),
                "title": e.get("title"),
                "tags": [t.get("label") for t in (e.get("tags") or []) if t.get("label")],
                "endDate": e.get("endDate"),
                "liquidity": _to_float(e.get("liquidity")),
                "volume": _to_float(e.get("volume")),
                "markets": markets,
            }
        )
        if max_results and len(events) >= max_results:
            break  # cap on IN-WINDOW events, after per-market filtering
    logger.info(
        "fetched %d event(s) | liquidity >= $%s | ending within %d day(s)%s",
        len(events),
        f"{min_liquidity:,.0f}",
        within_days,
        f" | tag={tag_slug}" if tag_slug else "",
    )
    return events


def fetch_event_comments(
    event_id: str | int,
    top_n: int = 20,
    min_len: int = 25,
    max_scan: int = 200,
) -> list[dict]:
    """Fetch an event's top crowd comments (by reaction count), spam-filtered.

    Polymarket comments are EVENT-level, public, and noisy (jokes, abuse,
    telegram-pump spam). This pulls the highest-reaction comments server-side,
    drops obvious spam (marketing bios/bodies, reported comments) and trivially
    short ones, de-duplicates, and returns a compact pool — intended as crowd
    'leads' to VERIFY downstream, NOT as evidence. Returns up to `top_n` dicts:
    {body, reactions, created, author}. Best-effort: returns [] on any error.
    """
    base = {
        "parent_entity_type": "Event",
        "parent_entity_id": event_id,
        "order": "reactionCount",
        "ascending": "false",
        "limit": min(_PAGE_LIMIT, max_scan),
    }
    raw: list[dict] = []
    offset = 0
    try:
        while len(raw) < max_scan:
            url = f"{GAMMA_COMMENTS_URL}?{urllib.parse.urlencode({**base, 'offset': offset})}"
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=30) as resp:
                page = json.load(resp)
            if not page:
                break
            raw.extend(page)
            if len(page) < base["limit"]:
                break
            offset += base["limit"]
    except Exception as exc:  # network/Cloudflare/JSON — never break a forecast
        logger.warning("comment fetch failed for event %s: %s", event_id, exc)
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for c in raw:
        body = (c.get("body") or "").strip()
        key = body.lower()
        if len(body) < min_len or key in seen:
            continue
        if (c.get("reportCount") or 0) > 0:
            continue
        bio = (c.get("profile") or {}).get("bio") or ""
        if _SPAM_RE.search(bio) or _SPAM_RE.search(body):
            continue
        seen.add(key)
        out.append(
            {
                "body": body,
                "reactions": c.get("reactionCount") or 0,
                "created": c.get("createdAt"),
                "author": (c.get("profile") or {}).get("name") or "anon",
            }
        )
        if len(out) >= top_n:
            break
    return out


def _format_event(event: dict, max_markets: int = 6) -> str:
    """Render one event as a compact, readable block for the CLI listing."""
    lines = [
        f"{event['title']}  [{event['slug']}]",
        f"  ends {event['endDate']} | liquidity ${event['liquidity']:,.0f} "
        f"| volume ${event['volume']:,.0f}",
    ]
    if event["tags"]:
        lines.append(f"  tags: {', '.join(event['tags'])}")
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
        "--tag", default=None, metavar="SLUG",
        help="only events carrying this tag slug, e.g. politics, sports, crypto "
        "(filtered server-side)",
    )
    parser.add_argument(
        "--comments", action="store_true",
        help="also fetch each event's top crowd comments (spam-filtered) — opt-in, "
        "off by default; an extra request per event",
    )
    parser.add_argument(
        "--comments-top", type=int, default=10, metavar="N",
        help="max comments to fetch per event with --comments (default: 10)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit raw JSON instead of the readable listing",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    events = fetch_markets(
        args.min_liquidity, args.days, max_results=args.limit, tag_slug=args.tag
    )
    if args.comments:  # opt-in: attach top comments to each event
        for event in events:
            event["comments"] = fetch_event_comments(event["id"], top_n=args.comments_top)
    if args.json:
        print(json.dumps(events, indent=2))
        return
    if not events:
        print("No matching events.")
        return
    for event in events:
        print(_format_event(event))
        for c in event.get("comments") or []:
            print(f"    💬 [{c['reactions']}] {c['body'][:150]}")
        print()


if __name__ == "__main__":
    main()
