# MODEL vs MARKET — MLB Calibration Ledger (dashboard)

A single-file, fully responsive web dashboard that renders the model-vs-market
calibration ledger exactly to the design system (`mlb predict design system desktop.pdf`).

## Files
- `index.html` — the entire app (React + inline styles via CDN, IBM Plex fonts). No build step.
- `ledger.json` — the data the dashboard reads. Schema = the §06 data contract.

## How "live update" works
On load and **every 30 seconds**, the page fetches `./ledger.json` (also a manual
**Refresh** button in the header). Whenever your pipeline overwrites `ledger.json`,
the dashboard reflects it on the next poll. The header shows a **LIVE** dot + last
sync time when it's reading the served file, or **SNAPSHOT** when it's falling back
to the data embedded in the HTML (e.g. opened directly from disk via `file://`,
where browsers block `fetch`).

## Run / share
- **Locally:** double-click `index.html` (uses the embedded snapshot), or serve the
  folder for true polling: `python -m http.server` then open `http://localhost:8000`.
- **Publicly:** deploy this `dashboard/` folder to any static host (GitHub Pages,
  Netlify, Vercel, S3). `index.html` + `ledger.json` are all you need.

## Wiring to the real pipeline
The repo's internal `ledger/ledger.json` uses a per-(game,market) row schema. The
dashboard expects the §06 game-record schema, so write the dashboard feed separately,
e.g. a small step at the end of `run_predict.py` that emits one record per game:

```json
{ "date":"2026-06-21","away":"NYY","away_name":"Yankees","home":"BOS","home_name":"Red Sox",
  "model_home_win_pct":47,"market_home_win_pct":53,"edge":6,"logged_at":"2026-06-21T16:05:00",
  "status":"final","away_score":6,"home_score":3,"winner":"NYY","pick_correct":true }
```
Upcoming games omit `away_score`/`home_score`/`winner`/`pick_correct` and use `"status":"upcoming"`.

All scoreboard figures (Brier per side, edge, contrarian flag, record, accuracy,
calibration trend) are derived **client-side** from these raw inputs, per §07/§08 —
the file only stores raw inputs.

> Not betting advice — calibration tracking only.
