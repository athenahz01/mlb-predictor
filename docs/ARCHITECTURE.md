# Athena Baseball architecture

## System boundaries

```text
MLB Stats API / Statcast / Open-Meteo / Kalshi research feed
                         |
                  versioned snapshots
                         |
             feature + simulation services
                         |
             canonical prediction ledger
                (Supabase PostgreSQL)
                  /        |         \
            FastAPI    resolution    evaluation
               |
       Next.js product + grounded Ask Athena
```

The prediction service is authoritative for numbers. Ask Athena receives structured
evidence and may explain or select predictions, but never computes a probability.

## Canonical revision identity

`game_id + category + statistic + player_id + team_id` forms a forecast key. A material
input or output change creates a new immutable revision. Exact retries are idempotent.
The previous revision becomes `superseded`; it remains available in the timeline and
initial-forecast evaluation track.

## Authentication and ownership

Supabase sends email magic links and issues JWTs. FastAPI validates the JWT server-side
before profile/following writes. Development can use an explicit local identity only when
`AUTH_REQUIRED=false`.

## Premium isolation

Prediction, opportunity ranking, risk, execution, position monitoring, and audit are
separate protocols. The shipped execution implementation is `TradingDisabled` and raises
on every submission. Prediction generation never imports an execution client.
