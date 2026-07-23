# Athena Baseball build log

Last updated: 2026-07-23

## Current phase

**Phase 2 (model improvement) is complete as a research pass.** All seven Tier 1 challengers
were implemented behind compatibility-safe interfaces, evaluated on frozen data through the
Phase 2 gate, and **rejected** — champions are unchanged. Phase 1 integrity infrastructure and
Phase 4/5 product surfaces remain implemented and verified locally.

The Phase 1 production gate is still **not passed** (no Supabase project, legacy prop outcomes
partially unrecoverable, incremental Statcast promotion not yet run on a frozen production
snapshot). The existing static dashboard remains the production surface. No production cutover,
infrastructure provisioning, DNS change, paid service, or live trading action has occurred.

## Phase 2 results (model improvement)

Seven Tier 1 challengers were built and evaluated one model-family-at-a-time against the
simulation champion, using 2025 Statcast as a frozen prior and the first 150 chronological
2026 games as the test window (250 sims/arm, 10,000 date-clustered bootstrap resamples,
practical-effect threshold, time-half stability, collateral-output checks, Holm adjustment).

**Outcome: all seven rejected; no champion changed.** `projection` (hierarchical Marcel) was
the only arm with a positive primary point estimate (+0.00335 winner log-loss) but its CI
crosses zero (Holm p = 1.0). Full table and rationale are in `docs/GRAVEYARD.md`; machine
artifacts (per-game/per-player losses, seeds, snapshot hashes, versions) are in
`reports/phase2/`.

Tier 2: umpire and defense models rejected for data availability (0% umpire-ID coverage; no
OAA/framing fields); pitch-type matchup was feasible, evaluated, and rejected on its metrics.

Known limitation: the Phase 2 test is likely underpowered (150 games, single prior season,
250 sims/arm). Re-run with more games, multi-season priors, and more simulations before
treating the negatives as settled.

## Champion models (unchanged after Phase 2)

| Category | Champion | Status | Rationale |
| --- | --- | --- | --- |
| Game winner | Legacy Elo baseline | Provisional | Simulation and all Phase 2 challengers failed the date-clustered ship gate. |
| Totals | None | Experimental | Bullpen/transition challengers did not clear the totals CRPS gate. |
| NRFI/YRFI | None | Experimental | Descriptive calibration did not clear readiness threshold. |
| Starter strikeouts | Simulation v2 | Provisional | Workload challenger did not improve starter-K CRPS. |
| Batter hits | None | Unavailable | No complete reproducible evaluation artifact. |
| Batter total bases | None | Experimental | Existing x-rate result lacks full new gate metadata. |
| Batter home runs | None | Experimental | Playing-time challenger did not improve HR Brier. |

No market-outperformance claim is permitted. Across 265 legacy dashboard games with a
market value, model Brier is 0.242 versus market Brier 0.240.

## Completed milestones

- Created and switched to the required `athena-baseball` branch while preserving the
  pre-existing `run_predict.py` working-tree change.
- Audited ingestion, feature generation, simulation, evaluation, JSON ledger, dashboard
  export, market ingestion, and scheduled workflow.
- Added a PostgreSQL/Supabase-compatible canonical ledger with immutable prediction
  revisions, headline selection, initial/latest-pregame tracks, resolution guards,
  structured evidence, validation status, and data-quality flags.
- Added reversible Alembic migration `20260723_0001`.
- Imported all 1,497 legacy ledger rows into the local canonical database.
- Reconciled all 276 dashboard games with zero game-level mismatches and zero exact source
  duplicates.
- Recovered 542 game-winner/total outcomes from the dashboard; retained 955 rows with
  explicit unresolved/legacy flags.
- Added content-addressed snapshot manifests, checksum verification, atomic promotion, and
  rollback; tests demonstrate promotion refusal after tampering and successful rollback.
- Replaced the old individual-game significance gate for the baseline report with a
  10,000-resample date-clustered paired bootstrap, practical-effect threshold, confidence
  interval gate, and Holm multiple-testing adjustment.
- Generated `reports/baseline.md` and machine-readable `reports/baseline.json` over 150
  chronological cached games.
- Fixed weather/environment multipliers so they apply to bullpen matchups as well as
  starters.
- Exposed missing batter, pitcher, handedness, and bullpen inputs as data-quality flags.
- Expanded the simulation contract with team run distributions, team totals, shutout/five-
  plus probabilities, extra-innings probability, player IDs, hit probability, and starter
  pitch/batters-faced expectations.
- Added FastAPI v1 endpoints for predictions, revisions, resolution, game timelines, daily
  slate, model status, profiles, following, and Ask Athena.
- Added Supabase JWT verification and email magic-link frontend integration behind
  configuration.
- Added provider-neutral OpenAI/Anthropic adapters, deterministic fallback, timeouts,
  output-token guard, response caching, tool-call audit logging, and PII-safe question
  hashes.
- Added explicit disabled Premium service boundaries; live execution raises a hard error.
- Built the responsive Next.js App Router product with Today, Games, Pitchers, Batters,
  Ask Athena, Following, Profile, and game timeline routes.
- Added loading, empty, missing-data, error, keyboard-focus, touch-target, and reduced-
  motion states.
- Added a product-specific Open Graph card and verified exact generated text.
- Added 30 grounded-agent evaluation cases, including 14 refusal cases.
- **Phase 2:** implemented hierarchical player projection, player-level platoon splits,
  stochastic starter workload, role/leverage bullpen tiers, batter playing-time survival,
  empirical base-out transitions, and pitch-type matchup models — all opt-in behind the
  existing simulation contract.
- **Phase 2:** added the frozen-data challenger evaluation harness (`evaluation/run_phase2.py`,
  `evaluation/challenger.py`, `evaluation/finalize_phase2.py`) with discrete CRPS, time-half
  stability, collateral-output checks, and Holm adjustment; wrote all machine-readable
  artifacts to `reports/phase2/`.
- **Phase 2:** removed a simulator inefficiency (matchup probabilities were recomputed per
  Monte Carlo game instead of once per batch).

## Tests and gates completed

- Existing `self_test.py`: pass.
- Python tests: **28 pass** (15 Phase 1 + 13 Phase 2 model/challenger tests).
- Targeted Ruff checks: pass (including Phase 2 modules).
- Mypy type check: pass across the new API/evaluation/pipeline/model modules.
- Frontend strict TypeScript, ESLint, production build, npm audit: pass.
- API smoke, snapshot promotion/rollback/tamper, seed reproducibility, distribution
  invariants, agent grounding/refusal: pass.
- Phase 2 challenger gate: all seven challengers evaluated and rejected with saved artifacts.

## Open issues

- Re-run the Phase 2 gate with more statistical power (more test games, multi-season priors,
  more simulations per arm) before treating the challenger rejections as final.
- Build the fully incremental Statcast partition pull and measured daily rebuild; existing
  `pull_statcast.py` still downloads a full date range.
- Run a production-sized snapshot candidate through promotion/rollback and record runtime.
- Resolve legacy NRFI, starter-K, and batter-HR rows from MLB box scores where IDs can be
  recovered; 955 outcomes are still unavailable.
- Add MLB gamePk and stable player IDs to old rows where deterministic matching is possible.
- Expand real stored outputs for all Phase 3 categories.
- Add end-to-end browser tests after the local browser connection is available.
- Run preview auth against an actual Supabase project.

## Blockers

- Supabase URL, publishable key, JWT secret, and Postgres connection are not configured.
- No OpenAI or Anthropic credential is present; Ask Athena uses the safe deterministic
  provider locally.
- Render/Vercel previews cannot be provisioned without explicit approval and credentials.
- Production cutover requires explicit user approval.

## Next steps

1. Re-run the Phase 2 evaluation with more power; promote any challenger that then clears.
2. Complete incremental Statcast partitioning and frozen daily rebuild/promotion.
3. Add canonical ledger writes to the scheduled slate job and retire display-side resolution.
4. Recover prop outcomes/player IDs and rerun baselines from a complete snapshot.
5. Connect a user-approved Supabase preview and verify magic-link auth/preferences.
6. Run responsive browser QA and preview deployment.
7. Prepare cutover evidence; do not switch production without explicit approval.
