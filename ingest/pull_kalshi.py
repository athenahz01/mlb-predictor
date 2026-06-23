"""
ingest/pull_kalshi.py
---------------------
Pull MLB market lines from Kalshi's public REST API. Market-DATA endpoints need
NO authentication, so this is read-only and key-free.

Series confirmed (verify live, Kalshi rotates tickers):
  KXMLBGAME  - per-game winner (moneyline)
  KXMLBTOTAL - per-game run total (over/under)

Usage:
  python -m ingest.pull_kalshi --series winner --status open
  python -m ingest.pull_kalshi --series total  --date 2026-04-19

Stores a frozen JSON snapshot under data/snapshots/ keyed by date+series so you
never silently overwrite a line you already logged against (snapshot discipline).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import requests

import config

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json",
                        "User-Agent": "mlb-predictor/0.1"})


def get_markets(series_ticker: str, status: str = "open",
                limit: int = 1000) -> list[dict]:
    """Page through /markets for a series. Returns a list of market dicts."""
    url = f"{config.KALSHI_BASE}/markets"
    out, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "limit": limit,
                  "status": status}
        if cursor:
            params["cursor"] = cursor
        r = SESSION.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)            # be polite; token-bucket friendly
    return out


def _to_cents(m: dict, base: str):
    """Normalize a price field to cents (0..100).
    Kalshi's 2026 schema uses <base>_dollars (string/float, 0..1); older schema
    used integer cents under <base>. Prefer dollars, fall back to int."""
    d = m.get(f"{base}_dollars")
    if d is not None:
        try:
            return round(float(d) * 100, 1)
        except (TypeError, ValueError):
            pass
    return m.get(base)


def simplify(m: dict) -> dict:
    """Pull just the fields we benchmark against (normalized to cents)."""
    return {
        "ticker": m.get("ticker"),
        "title": m.get("title") or m.get("yes_sub_title"),
        "yes_sub_title": m.get("yes_sub_title"),
        "status": m.get("status"),
        "yes_bid": _to_cents(m, "yes_bid"),
        "yes_ask": _to_cents(m, "yes_ask"),
        "no_bid": _to_cents(m, "no_bid"),
        "no_ask": _to_cents(m, "no_ask"),
        "last_price": _to_cents(m, "last_price"),
        "volume": m.get("volume") or m.get("volume_fp"),
        "close_time": m.get("close_time"),
    }


def snapshot(series_key: str = "winner", status: str = "open") -> Path:
    series = config.KALSHI_SERIES[series_key]
    markets = [simplify(m) for m in get_markets(series, status=status)]
    today = dt.date.today().isoformat()
    path = config.SNAPSHOTS / f"kalshi_{series_key}_{today}.json"
    path.write_text(json.dumps(
        {"pulled": dt.datetime.now(dt.timezone.utc).isoformat(), "series": series,
         "n": len(markets), "markets": markets}, indent=2))
    print(f"[kalshi] {series} -> {len(markets)} markets -> {path.name}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="winner",
                    choices=list(config.KALSHI_SERIES.keys()))
    ap.add_argument("--status", default="open")
    args = ap.parse_args()
    snapshot(args.series, args.status)
