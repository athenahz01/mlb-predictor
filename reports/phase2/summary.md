# Phase 2 challenger results

Frozen prior: 2025 Statcast. Chronological test: first 150 usable 2026 games.
Each arm used 250 simulations per game and the same game-based seed. Statistical
decisions use 10,000 date-clustered paired bootstrap resamples, practical-effect
thresholds, time-half stability, collateral-output checks, and Holm adjustment.

| Challenger | Primary metric | Champion | Challenger | Effect | 95% CI | Holm p | Decision |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| projection | winner_log_loss | 0.69016 | 0.68681 | +0.00335 | [-0.01179, +0.01864] | 1.0000 | REJECT |
| platoon | winner_log_loss | 0.69016 | 0.69493 | -0.00477 | [-0.01581, +0.00540] | 1.0000 | REJECT |
| workload | starter_k_crps | 1.27477 | 1.29510 | -0.02033 | [-0.08942, +0.05827] | 1.0000 | REJECT |
| bullpen | totals_crps | 2.70832 | 2.74726 | -0.03894 | [-0.07687, +0.01107] | 1.0000 | REJECT |
| playing_time | batter_hr_brier | 0.09237 | 0.09226 | +0.00011 | [-0.00006, +0.00027] | 0.7867 | REJECT |
| transitions | totals_crps | 2.70832 | 2.73566 | -0.02734 | [-0.07810, +0.03355] | 1.0000 | REJECT |
| pitch_types | winner_log_loss | 0.69016 | 0.69160 | -0.00144 | [-0.01230, +0.00664] | 1.0000 | REJECT |

Positive effect means lower challenger loss. A positive point estimate alone is
insufficient: the confidence interval, practical threshold, stability, adjusted
p-value, and collateral checks must all pass.

No challenger cleared the gate unless explicitly marked SHIP above. Rejected
challengers remain available only as research code and are not category champions.
