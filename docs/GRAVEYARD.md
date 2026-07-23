# Experiment graveyard

Failed or held experiments remain visible so the same idea is not repeatedly shipped on
weaker evidence.

## Legacy simulation as game-winner champion — HOLD

- Change: replace the walk-forward Elo champion with the base-out simulation.
- Data/evaluation window: 150 chronological 2026 cached games using prior-season 2025
  player rates; exact IDs and dates are in `reports/baseline.json`.
- Primary metric: binary log loss.
- Effect: see `comparisons.simulation_vs_elo.effect` in the machine-readable report.
- Confidence interval: see the date-clustered 95% interval in the same artifact.
- Bootstrap: 10,000 date-clustered paired resamples with seed 20260723; Holm adjusted with
  the Pythagenpat comparison.
- Decision: hold. The challenger did not clear the practical-effect, confidence-interval,
  and adjusted significance gate. The legacy cache also lacks complete version metadata.

## Totals validation after legacy bullpen changes — REJECT FOR SHIPPING

- Change: publish the simulation's 8.5 total probability as validated.
- Data/evaluation window: first 100 completed games in the current 2026 replay cache.
- Primary metric: calibration/ECE for the binary over, with run bias descriptive.
- Result: projected mean 8.46 vs actual 8.72; bias -0.26; ECE 0.121.
- Confidence interval/bootstrap: not stored by the old experiment, which itself fails the
  new artifact standard.
- Decision: experimental only. Re-run after role-aware bullpen work on a frozen snapshot.

## NRFI validation claim — REJECT FOR SHIPPING

- Change: label NRFI as validated.
- Data/evaluation window: first 100 completed games in the current 2026 replay cache.
- Primary metric: Brier/calibration.
- Result: Brier 0.247 with reliability error above the old readiness threshold.
- Confidence interval/bootstrap: missing from the old artifact.
- Decision: experimental only; the previous validation claim is not reproducible under the
  new standard.

## Phase 2 Tier 1 challengers — ALL REJECTED

Frozen prior: 2025 Statcast. Chronological test: first 150 usable 2026 games. Each arm ran
250 simulations per game with a game-based seed. Decisions use 10,000 date-clustered paired
bootstrap resamples, practical-effect thresholds, time-half stability, collateral-output
checks, and Holm adjustment across the five Tier 1 hypotheses. Positive effect = lower
challenger loss. Full per-game/per-player artifacts are in `reports/phase2/`.

| Challenger | Version | Primary metric | Champion | Challenger | Effect | 95% CI | Holm p | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| projection | hierarchical-marcel-v1 | winner_log_loss | 0.69016 | 0.68681 | +0.00335 | [-0.01179, +0.01864] | 1.0000 | REJECT |
| platoon | player-platoon-partial-pooling-v1 | winner_log_loss | 0.69016 | 0.69493 | -0.00477 | [-0.01581, +0.00540] | 1.0000 | REJECT |
| workload | starter-workload-v1 | starter_k_crps | 1.27477 | 1.29510 | -0.02033 | [-0.08942, +0.05827] | 1.0000 | REJECT |
| bullpen | three-leverage-tier-v1 | totals_crps | 2.70832 | 2.74726 | -0.03894 | [-0.07687, +0.01107] | 1.0000 | REJECT |
| playing_time | starter-pa-survival-v1 | batter_hr_brier | 0.09237 | 0.09226 | +0.00011 | [-0.00006, +0.00027] | 0.7867 | REJECT |
| transitions | empirical-base-out-transitions-v1 | totals_crps | 2.70832 | 2.73566 | -0.02734 | [-0.07810, +0.03355] | 1.0000 | REJECT |
| pitch_types | pitch-type-matchup-v1 | winner_log_loss | 0.69016 | 0.69160 | -0.00144 | [-0.01230, +0.00664] | 1.0000 | REJECT |

- Decision: none cleared the gate. `projection` was the only challenger with a positive
  primary point estimate, but its confidence interval crosses zero (Holm p = 1.0). All
  implementations remain available as opt-in research code; production defaults stay on the
  existing champions.
- Caveat for future work: the test is likely underpowered — 150 games (~2 weeks), a single
  prior season (which disables the projection model's multi-season weighting), and 250
  simulations per arm. Re-run with more games, 2–3 seasons of priors, and more simulations
  before treating these as settled negatives.

## Phase 2 Tier 2 research — REJECTED FOR DATA AVAILABILITY

- **Umpire model:** umpire identity coverage is exactly 0% in the frozen snapshot
  (called-pitch location 97.9%, catcher ID 100%). An umpire effect cannot be trained or
  joined without identity. Rejected.
- **Defense model (OAA / framing / arm):** fielder IDs and alignment exist, but Outs Above
  Average, catcher framing/blocking outcomes, and outfield arm strength are absent. The
  frozen snapshot cannot reconstruct the requested defensive quality features. Rejected.
- **Expanded environment model:** no frozen pregame forecast, actual roof-open state,
  park-renovation history, or handed park-factor history. A replay would substitute hindsight
  or invented context. Rejected; operational default stays neutral unless the existing live
  environment option is explicitly enabled.
- **Pitch-type matchup:** data audit found ~98–100% coverage (pitch type, velocity, movement,
  extension, handedness), so it was feasible and was evaluated as a Tier 1 challenger above —
  and rejected on its metrics, not on data.
