# Historical ledger reconciliation

Generated: 2026-07-23T19:04:36.513535+00:00

| Check | Count |
| --- | ---: |
| Existing JSON ledger rows | 1497 |
| Dashboard game rows | 276 |
| Newly imported rows | 0 |
| Already-imported idempotent rows | 1497 |
| Canonical database rows | 1497 |
| Exact duplicate source rows | 0 |
| Games missing from one source | 0 |
| Outcomes recovered from dashboard | 542 |
| Rows without recoverable outcomes | 955 |
| Rows missing player IDs | 669 |
| Rows missing model versions | 1497 |

Legacy rows are preserved with explicit quality flags. Only game-winner and
8.5-total outcomes can be reconstructed from the legacy dashboard; NRFI,
starter-strikeout, and batter-HR outcomes require box-score-level migration.
