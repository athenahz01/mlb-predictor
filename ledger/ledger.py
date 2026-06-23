"""
ledger/ledger.py
----------------
Append-only prediction ledger. Mirrors your NHL/NBA/WC discipline:

  1. log the MODEL number BEFORE checking the market price (honest out-of-sample)
  2. attach the de-vigged market price as a separate field
  3. once the game resolves, score it
  4. report log-loss / Brier / calibration for model AND market, plus CLV

Every row is one (game, market) prediction. Markets: moneyline, total_over_X,
nrfi, f5, starter_k_over_X, batter_hr, etc. Keep it flat and boring.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np

import config

LEDGER = config.LEDGER_PATH


def _load() -> list[dict]:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return []


def _save(rows: list[dict]):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=2))


def log_prediction(game_id: str, market: str, model_p: float,
                   *, market_p: Optional[float] = None,
                   market_close_p: Optional[float] = None,
                   meta: Optional[dict] = None) -> dict:
    """
    Log one prediction. model_p is logged first and is immutable. market_p is the
    de-vigged price at log time; market_close_p is filled later for CLV.
    """
    rows = _load()
    row = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "game_id": game_id,
        "market": market,
        "model_p": round(float(model_p), 4),
        "market_p": None if market_p is None else round(float(market_p), 4),
        "market_close_p": None if market_close_p is None else round(float(market_close_p), 4),
        "result": None,            # filled on resolution: 1 (yes) / 0 (no)
        "meta": meta or {},
    }
    rows.append(row)
    _save(rows)
    return row


def resolve(game_id: str, market: str, result: int,
            market_close_p: Optional[float] = None):
    """Mark the binary outcome (1/0) for a logged prediction."""
    rows = _load()
    for r in rows:
        if r["game_id"] == game_id and r["market"] == market and r["result"] is None:
            r["result"] = int(result)
            if market_close_p is not None:
                r["market_close_p"] = round(float(market_close_p), 4)
    _save(rows)


def _logloss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def report(market_prefix: str = "") -> dict:
    """Scoring summary for resolved rows (optionally filter by market prefix)."""
    rows = [r for r in _load() if r["result"] is not None
            and r["market"].startswith(market_prefix)]
    if not rows:
        return {"n": 0, "note": "no resolved predictions yet"}

    y = np.array([r["result"] for r in rows], float)
    mp = np.array([r["model_p"] for r in rows], float)
    out = {
        "n": len(rows),
        "model_logloss": float(_logloss(mp, y).mean()),
        "model_brier": float(np.mean((mp - y) ** 2)),
        "model_acc": float(np.mean((mp > 0.5) == (y == 1))),
    }
    have_mkt = [r for r in rows if r["market_p"] is not None]
    if have_mkt:
        ym = np.array([r["result"] for r in have_mkt], float)
        kp = np.array([r["market_p"] for r in have_mkt], float)
        mm = np.array([r["model_p"] for r in have_mkt], float)
        out["market_logloss"] = float(_logloss(kp, ym).mean())
        out["market_brier"] = float(np.mean((kp - ym) ** 2))
        out["model_vs_market_logloss_delta"] = out["model_logloss"] - out["market_logloss"]
    # CLV: did our logged market_p beat the close?
    clv_rows = [r for r in rows if r["market_p"] is not None
                and r["market_close_p"] is not None]
    if clv_rows:
        beat = [1 if (r["model_p"] > 0.5 and r["market_close_p"] > r["market_p"])
                or (r["model_p"] <= 0.5 and r["market_close_p"] < r["market_p"]) else 0
                for r in clv_rows]
        out["clv_beat_rate"] = float(np.mean(beat))
        out["clv_n"] = len(clv_rows)
    return out


def reliability(market_prefix: str = "", bins: int = 10) -> list[dict]:
    """Reliability curve points for a calibration plot."""
    rows = [r for r in _load() if r["result"] is not None
            and r["market"].startswith(market_prefix)]
    if not rows:
        return []
    p = np.array([r["model_p"] for r in rows])
    y = np.array([r["result"] for r in rows], float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                        "pred": float(p[m].mean()),
                        "obs": float(y[m].mean()),
                        "n": int(m.sum())})
    return out


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
