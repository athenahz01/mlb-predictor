# Athena Baseball build log

Last updated: 2026-07-23

## Current phase

Phase 1 integrity infrastructure is implemented and verified locally. Phase 4/5 product
surfaces are implemented against the canonical API and pass local build/tests. The Phase 1
production gate is **not yet passed** because a Supabase project is not connected, legacy
prop outcomes remain partially unrecoverable, and the full incremental Statcast promotion
job has not yet been run on a frozen production snapshot.

The existing static dashboard remains the production surface. No production cutover,
infrastructure provisioning, DNS change, paid service, or live trading action has occurred.

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

## Current champion models

| Category | Champion | Status | Rationale |
| --- | --- | --- | --- |
| Game winner | Legacy Elo baseline | Provisional | Simulation did not clear the new date-clustered ship gate against Elo. |
| Totals | None | Experimental | 100-game replay ECE 0.121; needs work after bullpen changes. |
| NRFI/YRFI | None | Experimental | Descriptive calibration did not clear readiness threshold. |
| Starter strikeouts | Simulation v2 | Provisional | Old descriptive gate passed; full frozen report still required. |
| Batter hits | None | Unavailable | No complete reproducible evaluation artifact. |
| Batter total bases | None | Experimental | Existing x-rate result lacks full new gate metadata. |
| Batter home runs | None | Experimental | Existing x-rate result lacks full new gate metadata. |

No market-outperformance claim is permitted. Across 265 legacy dashboard games with a
market value, model Brier is 0.242 versus market Brier 0.240.

## Audit findings and important decisions

- The dashboard independently resolves finals while every source JSON ledger row remains
  unresolved. Decision: database is canonical; frontend/API no longer infer canonical
  resolution from display data.
- Source rows have no model versions, player IDs, snapshots, feature versions, seeds, or
  revision lineage. Decision: preserve them as `legacy-unknown` with explicit quality flags.
- `run_slate.py` skips a whole game once any row exists, so scratches/lineup changes cannot
  revise it. Decision: all new writes go through idempotent immutable revision creation.
- Missing player data previously fell back silently to league average. Decision: keep
  conservative priors only with user-visible quality flags and reduced confidence.
- Environment effects were omitted from bullpen matchup probabilities. Decision: apply
  the same validated context to starter and bullpen matchup calculation.
- Existing backtests bootstrap by game and treat p-value as the ship rule. Decision: use
  chronological frozen data, date clusters, effect threshold, confidence interval, and
  multiple-testing policy.
- Render is the selected backend preview target because it supports a health-checked web
  service from a repository blueprint with no required infrastructure code. Vercel remains
  the frontend target. Neither has been provisioned.
- Alembic is the single migration system; Supabase migrations are not mixed in.
- Market comparison remains stored for research but secondary in the product.

## Tests and gates completed

- Existing `self_test.py`: pass.
- Python tests: 15 pass.
- Targeted Ruff checks: pass.
- Mypy type check: pass across 21 new API/evaluation/pipeline/script modules.
- Frontend strict TypeScript: pass.
- Frontend ESLint: pass.
- Frontend production build: pass using an isolated verification output directory.
- NPM dependency audit: zero known vulnerabilities.
- API smoke: health, Today with real imported rows, and model-performance endpoints pass.
- Historical migration: idempotent design; first import report complete.
- Snapshot promotion/rollback/tamper tests: pass.
- Seed reproducibility and probability distribution invariants: pass.
- Agent grounding/refusal catalog: pass.

## Open issues

- Build the fully incremental Statcast partition pull and measured daily rebuild; existing
  `pull_statcast.py` still downloads a full date range.
- Run a production-sized snapshot candidate through promotion/rollback and record runtime.
- Resolve legacy NRFI, starter-K, and batter-HR rows from MLB box scores where IDs can be
  recovered; 955 outcomes are still unavailable.
- Add MLB gamePk and stable player IDs to old rows where deterministic matching is possible.
- Replace fixed prior weights with the full event-specific hierarchical Marcel system.
- Implement role-sequenced multi-reliever bullpen simulation and stochastic starter hook.
- Fit play transitions from play-by-play data; current advancement logic remains simplified.
- Complete Tier 1 challenger evaluations and document each ship/reject decision.
- Expand real stored outputs for all Phase 3 categories; the simulation can calculate more
  than the legacy ledger currently stores.
- Add end-to-end browser tests after the local browser connection is available.
- Run preview auth against an actual Supabase project.

## Blockers

- Supabase URL, publishable key, JWT secret, and Postgres connection are not configured.
- No OpenAI or Anthropic credential is present; Ask Athena uses the safe deterministic
  provider locally.
- Render/Vercel previews cannot be provisioned without explicit approval and credentials.
- Production cutover requires explicit user approval.

## Next steps

1. Complete incremental Statcast partitioning and frozen daily rebuild/promotion.
2. Add canonical ledger writes to the scheduled slate job and retire display-side
   resolution after parity verification.
3. Recover prop outcomes/player IDs and rerun baselines from a complete snapshot.
4. Execute Tier 1 model challengers one at a time and fill `docs/GRAVEYARD.md`.
5. Connect a user-approved Supabase preview and verify magic-link auth/preferences.
6. Run responsive browser QA and preview deployment.
7. Prepare cutover evidence; do not switch production without explicit approval.
