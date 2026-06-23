"""
market/kalshi_match.py
----------------------
Match a scheduled game to its Kalshi markets (from a snapshot) and compute the
de-vigged fair probability + your model's edge.

Kalshi MLB ticker shapes (codes match statsapi abbreviations exactly):
  winner : KXMLBGAME-<YYMMMDD><HHMM><AWAY><HOME>-<TEAM>   (2 markets/game, YES per side)
  total  : KXMLBTOTAL-<YYMMMDD><HHMM><AWAY><HOME>-<N>     ("Over (N-0.5) runs", YES=over)

Prices are cents (1..99 = implied prob). We prefer the bid/ask midpoint, then
the ask, then last trade. Missing prices (no liquidity yet) -> edge skipped.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import config
from market.devig import fair_two_way, edge


def load_snapshot(date: str, series_key: str) -> list[dict]:
    path = config.SNAPSHOTS / f"kalshi_{series_key}_{date}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("markets", [])


def _price_cents(m: dict, side: str = "yes") -> Optional[float]:
    """Best available price in cents for the given side: mid -> ask -> last."""
    bid, ask = m.get(f"{side}_bid"), m.get(f"{side}_ask")
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    if ask is not None:
        return ask
    last = m.get("last_price")
    if last is not None:
        # last is a YES trade price; flip for the NO side
        return last if side == "yes" else 100 - last
    return None


def _spread(m: dict, side: str = "yes") -> Optional[float]:
    """Bid-ask spread in cents, if both sides are quoted."""
    bid, ask = m.get(f"{side}_bid"), m.get(f"{side}_ask")
    if bid is not None and ask is not None:
        return ask - bid
    return None


def _middle(ticker: str) -> str:
    return ticker.split("-")[1] if "-" in ticker else ""


def _suffix(ticker: str) -> str:
    return ticker.rsplit("-", 1)[1] if "-" in ticker else ""


# --------------------------------------------------------------------------
# Moneyline
# --------------------------------------------------------------------------
def winner_edge(markets: list[dict], away: str, home: str,
                model_home_p: float) -> Optional[dict]:
    pair = away + home
    home_mkt = away_mkt = None
    for m in markets:
        t = m.get("ticker", "")
        if _middle(t).endswith(pair):
            if _suffix(t) == home:
                home_mkt = m
            elif _suffix(t) == away:
                away_mkt = m
    if not home_mkt or not away_mkt:
        return None

    home_yes = _price_cents(home_mkt, "yes")
    away_yes = _price_cents(away_mkt, "yes")
    if home_yes is None or away_yes is None:
        return {"market": "moneyline", "note": "no price yet",
                "home_ticker": home_mkt.get("ticker")}
    if home_yes >= 99.5 or home_yes <= 0.5 or away_yes >= 99.5 or away_yes <= 0.5:
        return {"market": "moneyline", "note": "market settled (game over)",
                "home_ticker": home_mkt.get("ticker")}

    fair = fair_two_way(home_yes, away_yes, method="power")   # yes=home, no=away
    sp = _spread(home_mkt, "yes")
    return {
        "market": "moneyline",
        "model_home_p": model_home_p,
        "market_home_p": fair["fair_yes"],
        "edge": edge(model_home_p, fair["fair_yes"]),
        "overround": fair["overround"],
        "spread": sp,
        "home_ticker": home_mkt.get("ticker"),
    }


# --------------------------------------------------------------------------
# Totals
# --------------------------------------------------------------------------
def _line_of(m: dict) -> Optional[float]:
    """Get the over-line. Prefer the title text, fall back to ticker math."""
    title = m.get("title") or ""
    mo = re.search(r"[Oo]ver\s+([\d.]+)", title)
    if mo:
        return float(mo.group(1))
    suf = _suffix(m.get("ticker", ""))
    if suf.isdigit():
        return int(suf) - 0.5
    return None


def total_edge(markets: list[dict], away: str, home: str, line: float,
               model_over_p: float) -> Optional[dict]:
    pair = away + home
    for m in markets:
        if not _middle(m.get("ticker", "")).endswith(pair):
            continue
        if _line_of(m) != line:
            continue
        over = _price_cents(m, "yes")
        under = _price_cents(m, "no")
        if over is None or under is None:
            return {"market": f"total_over_{line}", "note": "no price yet"}
        fair = fair_two_way(over, under, method="power")
        return {
            "market": f"total_over_{line}",
            "model_over_p": model_over_p,
            "market_over_p": fair["fair_yes"],
            "edge": edge(model_over_p, fair["fair_yes"]),
            "overround": fair["overround"],
            "spread": _spread(m, "yes"),
            "ticker": m.get("ticker"),
        }
    return None


# --------------------------------------------------------------------------
# Card helper
# --------------------------------------------------------------------------
def print_edges(date: str, away: str, home: str, res: dict):
    """Pull both snapshots and print model-vs-market edge lines for the card."""
    win = load_snapshot(date, "winner")
    tot = load_snapshot(date, "total")
    if not win and not tot:
        print("\nMarket: no Kalshi snapshot for today. Run "
              "`python -m ingest.pull_kalshi --series winner` (and --series total).")
        return

    print("\n--- model vs market (Kalshi, de-vigged) ---")
    w = winner_edge(win, away, home, res["p_home_win"]) if win else None
    if w is None:
        print(f"Moneyline   no {home} market found in snapshot")
    elif "note" in w:
        print(f"Moneyline   {w['note']} - market likely closed/not open "
              f"({w.get('home_ticker','')})")
    else:
        thin = f"  (thin: {w['spread']:.0f}c spread)" if w.get("spread") and w["spread"] > 12 else ""
        print(f"Moneyline   model {w['model_home_p']:.3f}  market {w['market_home_p']:.3f}  "
              f"edge {w['edge']:+.3f}  (home {home}){thin}")

    if tot:
        for line in (7.5, 8.5, 9.5):
            key = f"over_{line}"
            if key not in res["total_over"]:
                continue
            t = total_edge(tot, away, home, line, res["total_over"][key])
            if t is None:
                continue
            if "note" in t:
                print(f"Total {line}  {t['note']}")
            else:
                thin = f"  (thin: {t['spread']:.0f}c)" if t.get("spread") and t["spread"] > 12 else ""
                print(f"Total {line}  model {t['model_over_p']:.3f}  "
                      f"market {t['market_over_p']:.3f}  edge {t['edge']:+.3f}{thin}")


def live_games(date: str) -> list[dict]:
    """Scan the winner snapshot and return games that currently have a usable
    two-sided price AND are still open (not settled). A market pinned at 0 or
    100 cents has already resolved, so we drop it."""
    win = load_snapshot(date, "winner")
    games = {}
    for m in win:
        t = m.get("ticker", "")
        mid, suf = _middle(t), _suffix(t)
        price = _price_cents(m, "yes")
        status = (m.get("status") or "").lower()
        # settled if pinned to an extreme or status says so
        settled = (price is not None and (price <= 0.5 or price >= 99.5)) \
            or status in ("closed", "settled", "finalized")
        games.setdefault(mid, {})[suf] = None if settled else price
    out = []
    for key, sides in games.items():
        priced = {s: p for s, p in sides.items() if p is not None}
        if len(priced) >= 2:                       # both sides open & quoted
            out.append({"segment": key, "sides": priced})
    return out


if __name__ == "__main__":
    import datetime as dt
    d = dt.date.today().isoformat()
    lg = live_games(d)
    if not lg:
        print(f"No games with live two-sided prices in today's snapshot ({d}).")
        print("Either re-pull (markets fill closer to first pitch) or all of "
              "today's games have started/closed.")
    else:
        print(f"Games with LIVE Kalshi prices right now ({len(lg)}):")
        for g in lg:
            seg = g["segment"]
            sides = "  ".join(f"{s}:{p:.0f}c" for s, p in g["sides"].items())
            print(f"  {seg}   {sides}")
