# Operations runbook

## Local development

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.migrate_legacy_ledger
.\.venv\Scripts\python.exe -m uvicorn athena_api.main:app --reload
```

In another terminal:

```powershell
Set-Location web
Copy-Item .env.example .env.local
npm install
npm run dev
```

The API is at `http://localhost:8000`, OpenAPI at `/docs`, and the product at
`http://localhost:3000`.

## Daily checks

- `/health/live` confirms the process.
- `/health/ready` confirms database connectivity.
- Check that the promoted snapshot points through the previous completed MLB day.
- Check that every active game/statistic has one headline and at least one initial track.
- Review data-quality flags before publishing high-support lists.
- Treat off-days as a successful empty slate, not a job failure.

## Snapshot recovery

```powershell
.\.venv\Scripts\python.exe -m pipeline.cli rollback
```

Rollback verifies the previous snapshot checksum before changing the atomic pointer.

## Incident behavior

- Missing lineup/player/weather input: retain the prior or conservative output only with
  a visible flag and reduced confidence.
- Pitcher scratch: create a new revision; never edit the overnight row.
- Postponement/suspension: invalidate the current headline with a reason and do not resolve
  it as a completed game.
- AI provider timeout: deterministic grounded explanation fallback; no free-form retry
  loop.
- Database unavailable: API readiness fails; the frontend shows an actionable error and
  must not fall back to fabricated prediction data.
