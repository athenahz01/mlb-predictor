# Preview and production deployment

No infrastructure has been provisioned by this build.

## Selected targets

- Frontend: Vercel, using the `web/` Next.js project.
- Backend: Render, using `render.yaml` and `Dockerfile.api`.
- Database/auth: a user-owned Supabase project.

Render was selected over Railway/Fly.io for the first preview because the repository
blueprint can declare a health-checked service without adding cluster administration.

## Preview setup

1. Create a Supabase project and obtain its pooled PostgreSQL connection string, project
   URL, publishable key, and JWT secret.
2. Configure the backend environment from `.env.example`; set `AUTH_REQUIRED=true`.
3. Run `alembic upgrade head` against the preview database.
4. Run `python -m scripts.migrate_legacy_ledger` once, then inspect
   `reports/reconciliation.md`.
5. Configure the Vercel project root as `web` and set the variables in
   `web/.env.example`.
6. Point `NEXT_PUBLIC_API_BASE_URL` at the Render preview URL and add the Vercel URL to
   backend `CORS_ORIGINS`.
7. Configure the Supabase magic-link redirect allowlist with
   `https://<preview-host>/auth/callback`.
8. Verify health, auth, profile/following writes, daily slate, timelines, and Ask Athena.

## Cutover gate

Before replacing the static production dashboard:

- complete responsive browser QA;
- prove scheduled writes are idempotent;
- compare old/new headline values for a full slate;
- demonstrate snapshot rollback;
- confirm every displayed validation label from frozen reports;
- create a database backup and rollback procedure;
- obtain explicit user approval.

Do not change DNS, production accounts, or the current root `vercel.json` before approval.
