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
