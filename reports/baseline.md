# Athena Baseball baseline

Generated: 2026-07-23T18:20:49.600726+00:00

All validation labels are **provisional**. The historical cache lacks complete model,
feature, snapshot, and simulation metadata, so it cannot satisfy the final shipping
standard even where descriptive results look favorable.

## Game winner

Chronological frozen-cache evaluation over 150 games from
2026-03-26 through
2026-04-07. Bootstrap resampling is clustered
by game date with 10,000 resamples.

| Model | Log loss | Brier | ECE | Calibration slope | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy simulation | 0.6882 | 0.2474 | 0.0614 | 0.5780 | 0.560 |
| Elo baseline | 0.6922 | 0.2495 | 0.0200 | -2.7338 | 0.553 |

Simulation minus Elo improvement: +0.0039 log loss,
95% CI [-0.0201, +0.0272],
date-clustered one-sided p=0.3705,
Holm-adjusted p=0.3705. **Decision: HOLD.**

The current game-winner champion remains the provisional Elo baseline. The simulation
does not clear the practical effect, confidence-interval, and multiple-testing gate.

## Public dashboard comparison

Across 265 completed games with a market value, model Brier is
0.242 and market Brier is 0.240.
Athena must not claim market outperformance.

## Other categories

| Category | Current status | Notes |
| --- | --- | --- |
| Totals | Experimental | 100-game replay: bias -0.26 runs, ECE 0.121; needs work. |
| NRFI/YRFI | Experimental | 100-game replay: Brier 0.247, ECE above readiness threshold. |
| Starter strikeouts | Provisional | Descriptive calibration passes the old tolerance, but no full frozen report yet. |
| Batter hits | Unavailable | No complete reproducible evaluation artifact. |
| Batter total bases | Experimental | Prior x-rate experiment lacks the full new gate metadata. |
| Batter home runs | Experimental | Prior x-rate experiment lacks the full new gate metadata. |

Machine-readable results, exact game IDs, per-game losses inputs, versions, checksums,
dates, bootstrap output, and seeds are stored in `reports/baseline.json`.
