# mlb-predictor

A baseball prediction pipeline built on the same philosophy as the WC / NHL / NBA
pipelines: **model vs. market, calibration in public, not betting advice.**

The core difference from the other sports: baseball's prediction targets (game
lines *and* player props) all fall out of one **base-out Monte Carlo simulation**.
You don't train a separate model per market — you simulate the game thousands of
times and read every outcome off the same joint distribution. The RF+XGBoost
ensemble you already use becomes a blending layer on the moneyline.

## Why simulation is the spine here
A 24-state base-out engine (3 out-counts × 8 base configurations) simulates each
plate appearance using log5 matchup probabilities, then aggregates:

- **Game**: moneyline, run total (over grid), run line, NRFI/YRFI, F5, first-to-score
- **Pitcher**: strikeout distribution → P(over k.5)
- **Batter**: hits, total bases, HR, P(≥1 HR), P(≥2 hits)

All internally consistent, from one model. That coherence is the whole point.

## Quickstart
```bash
pip install -r requirements.txt          # numpy/scipy/pandas are enough for the sim
python run_predict.py --sims 20000       # demo matchup → full prediction card
python run_predict.py --log              # also writes to ledger/ledger.json
```

Pull live data (run as scripts so the Windows multiprocessing guard fires):
```bash
python -m ingest.pull_mlb_statsapi --date today     # probables + lineups
python -m ingest.pull_kalshi --series winner        # KXMLBGAME lines
python -m ingest.pull_kalshi --series total         # KXMLBTOTAL lines
python -m ingest.pull_statcast --season 2025        # heavy; cached after first run
python -m ingest.pull_fangraphs --season 2025       # Steamer/ZiPS priors
```

## Module map
| Path | What it does |
|---|---|
| `config.py` | League PA rates, park factors, sim knobs, API bases |
| `features/pa_probabilities.py` | Multinomial log5 + park/platoon/umpire/TTO → per-PA probs |
| `sim/markov_game.py` | Base-out Monte Carlo engine → joint outcome distribution |
| `market/devig.py` | Power + multiplicative de-vig, model edge |
| `ingest/pull_mlb_statsapi.py` | Schedule, **probable pitchers**, confirmed lineups |
| `ingest/pull_statcast.py` | Statcast → per-PA rates (Windows-safe) |
| `ingest/pull_fangraphs.py` | Steamer/ZiPS priors + `shrink_rate()` |
| `ingest/pull_kalshi.py` | Kalshi MLB winner/total snapshots |
| `ledger/ledger.py` | Log model-first, resolve, score, CLV, reliability |
| `backtest/walk_forward.py` | Walk-forward harness + paired-bootstrap ship gate |

## House rules (carried over)
- Log the model number **before** checking the market price.
- Snapshot data per matchday; never auto-download mid-run.
- **Listed-pitcher confirmation** is the MLB analog of lineup confirmation, and
  it matters more — one starter is 30–40% of the game. Re-run on any scratch.
- Ship a new model only if it clears **paired-bootstrap p<0.05** on held-out log-loss.
- Frame everything as "model vs. market edge," never "beating the market."

## Reality check
MLB favorites win far less often than NBA/soccer favorites. The best public game
models land around 60–62% straight-up. The honest public deliverable is
calibration quality (reliability, log-loss/Brier) and CLV vs. Kalshi — not win rate.

See `BUILD_PLAN.md` for the staged roadmap and what's next.
