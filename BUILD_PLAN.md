# MLB Predictor — Build Plan

Model-vs-market, calibration in public. Same house rules as the WC / NHL / NBA
pipelines: build → backtest → paired-bootstrap significance test → ship or hold.
No model version ships without clearing **p < 0.05**. Snapshot data per day,
never auto-download mid-run. Log the model number before checking the market.

The architectural bet for baseball: a **base-out Monte Carlo simulation is the
generative core**, because one sim yields game lines *and* every prop coherently.
The RF+XGBoost ensemble you already trust becomes a **blending/correction layer
on the moneyline**, not the spine. Calibration + significance-gating discipline
is unchanged.

---

## Stage 0 — Scaffold  ✅ DONE
Project skeleton, config, league baselines, park factors.
- `config.py` — paths, `LEAGUE_PA_RATES` (8-outcome, sums to 1), `PARK_FACTORS`,
  sim knobs, Kalshi/StatsAPI bases.
- Directory layout: `ingest/ features/ sim/ market/ ledger/ backtest/ reports/`.

## Stage 1 — Data plumbing  ◑ PARTIAL (scripts written, live pulls to run on your box)
Pin every pull as a frozen snapshot under `data/snapshots/`.
- `ingest/pull_mlb_statsapi.py` ✅ — schedule, **probable pitchers**, **confirmed
  lineups**. Your listed-pitcher confirmation backbone. Poll pre-game; re-run on change.
- `ingest/pull_statcast.py` ✅ — pybaseball Statcast pull (Windows `__main__`
  guard included) → derives per-PA outcome rates for batters and pitchers.
- `ingest/pull_fangraphs.py` ✅ — Steamer/ZiPS + season rates as priors; `shrink_rate()`
  blends projection ↔ observed by PA count.
- `ingest/pull_kalshi.py` ✅ — `KXMLBGAME` (winner) + `KXMLBTOTAL` (total),
  unauthenticated REST, cursor pagination, snapshot to JSON.
- **TODO**: a `pull_retrosheet.py` to build empirical base-out advancement rates
  (replaces the hand-set advancement constants in the sim). Wire weather + umpire
  K-tendency feeds (statsapi venue + a public ump source).

## Stage 2 — Baselines  ✅ DONE
The floor every future model must beat at p<0.05.
- `models/elo.py` ✅ — pitcher-adjusted Elo (538 params: HFA 24, K=4 (6 postseason),
  travel `-miles^(1/3)*0.31`, rest `days*2.3`, MOV multiplier, pitcher delta).
- `models/pythag.py` ✅ — Pythagenpat (exponent `(R/G)^0.287`) + log5 game win prob.
- `models/baselines.py` ✅ + `ingest/pull_results.py` ✅ — season results -> ratings -> P(home).
- Wired into the card as `--baseline` floor lines (Elo / Pythag / Sim side by side).

## Stage 3 — Simulation core  ✅ DONE (v1)
- `features/pa_probabilities.py` ✅ — multinomial log5 (odds-ratio) with park,
  platoon, umpire, TTO adjustments + Morey-Cohen HR shrink for extreme matchups.
- `sim/markov_game.py` ✅ — 24-state base-out event-driven Monte Carlo. Emits the
  joint distribution: moneyline, total (over grid), run line, NRFI/YRFI, F5,
  first-to-score, per-pitcher K distribution, per-batter hits/TB/HR/P(HR)/P(2+H).
- Validated: league-avg teams → ~8.6 total runs, ~53% home, SP K ~5, anytime-HR ~14%.
- **TODO v2**: bullpen as multiple arms (not one aggregate), pinch-hitting,
  pitcher-fatigue curve, Retrosheet-fit advancement, base-stealing (2023 SB spike).

## Stage 4 — Ensemble blend + calibration  ◑ STARTED ("Stage 2" in the content series)
- `models/features.py` ✅ — leakage-free walk-forward training table (Elo diff,
  Pythag diff, rest) + stored Elo baseline prob per game.
- `models/ensemble.py` ✅ — calibrated RF+XGBoost game-winner; `train_and_gate()`
  significance-tests it vs the Elo baseline (paired bootstrap) and only saves an
  artifact when it clears **p<0.05** (otherwise HOLD).
- `backtest/sim_backtest.py` ✅ — puts the MAIN predictor (the simulation) on
  trial: runs the sim on completed games (real lineups+starters from boxscores),
  compares to walk-forward Elo + Pythagenpat, and paired-bootstraps the sim's
  log-loss vs each baseline -> PASS/HOLD verdict. Caches sim predictions to resume.
- `backtest/props_backtest.py` ✅ — puts the NON-WINNER markets on trial:
  totals (over/under), NRFI, and starter strikeouts. Checks calibration
  (bias + reliability/ECE) of the sim's probabilities against real outcomes,
  so each market is only published once it proves trustworthy.
- **TODO**: stack the ensemble with the sim's win-prob once enough games are
  backtested/logged; validate per-batter props (hits/HR) once more data exists;
  add beta calibration for thin prop markets.

## Stage 5 — Ledger + Kalshi benchmark + reports  ◑ PARTIAL
- `ledger/ledger.py` ✅ — log model-first, attach de-vigged market, resolve, report
  (log-loss/Brier/acc, model-vs-market delta, CLV beat-rate, reliability curve).
- `market/devig.py` ✅ — power + multiplicative de-vig, edge.
- `backtest/walk_forward.py` ✅ — walk-forward harness + `should_ship()` gate.
- **TODO**: `reports/` reliability-diagram + rolling log-loss plots for the public
  "calibration in public" artifact. Auto-pull Kalshi close for CLV.

---

## Run order (today)
```
python run_predict.py --sims 20000        # demo matchup, full card
python run_predict.py --log               # also write to the ledger
python -m ingest.pull_kalshi --series winner
python -m ingest.pull_mlb_statsapi --date today
python -m ingest.pull_statcast --season 2025     # heavy, cached after first run
```

## Ship gates (non-negotiable)
- New model artifact ships only if paired-bootstrap **p<0.05** on held-out log-loss.
- Version number bumps only when the trained artifact changes; tooling = same version.
- Props don't publish below ~300 graded outcomes per market (variance too high).
- Any single-season accuracy >65% → treat as leakage, not skill.

## Honest framing
MLB favorites win far less than NBA/soccer. Best public game models ~60–62%.
The deliverable is **calibration + CLV vs Kalshi**, never "winners." Edge is
"model vs market," never "beating the market."
