"use client";

import { AlertCircle, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { EmptyState, Percent, StatusPill } from "@/components/prediction-ui";
import { getPredictions } from "@/lib/api";
import type { Prediction } from "@/lib/types";

const LABELS: Record<string, string> = {
  home_win_probability: "Home win",
  total_over_8_5: "Over 8.5 runs",
  nrfi: "No run, first inning",
  home_starter_strikeouts_over_5_5: "Home starter over 5.5 K",
  away_starter_strikeouts_over_5_5: "Away starter over 5.5 K",
  home_run_probability: "Home-run outlook",
};

export function PredictionExplorer({
  category,
  emptyTitle,
}: {
  category?: Prediction["category"];
  emptyTitle: string;
}) {
  const [rows, setRows] = useState<Prediction[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  useEffect(() => {
    getPredictions(category ? `category=${category}&limit=500` : "limit=500")
      .then(setRows)
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [category]);
  const filtered = useMemo(
    () =>
      rows.filter((row) =>
        `${row.game_id} ${LABELS[row.statistic] ?? row.statistic}`
          .toLowerCase()
          .includes(query.toLowerCase()),
      ),
    [query, rows],
  );
  if (error) {
    return (
      <div className="error-state">
        <AlertCircle size={20} />
        <div><strong>Predictions could not be loaded.</strong><p>Check the API connection.</p></div>
      </div>
    );
  }
  return (
    <>
      <label className="search-control">
        <Search size={17} />
        <span className="sr-only">Filter predictions</span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by team, player, or prediction"
        />
      </label>
      {loading ? (
        <div className="table-skeleton" aria-label="Loading predictions" />
      ) : filtered.length ? (
        <div className="prediction-table">
          <div className="table-row table-header" aria-hidden="true">
            <span>Matchup</span><span>Forecast</span><span>Value</span><span>Status</span>
          </div>
          {filtered.map((row) => (
            <div className="table-row" key={row.id}>
              <div>
                <strong>{row.game_id.split("-20")[0].replace("@", " at ")}</strong>
                <small>{new Date(row.created_at).toLocaleDateString()}</small>
              </div>
              <span>{LABELS[row.statistic] ?? row.statistic.replaceAll("_", " ")}</span>
              <strong className="table-value">
                {row.probability !== null ? (
                  <Percent value={row.probability} />
                ) : (
                  <span className="tabular">{row.projected_value?.toFixed(2) ?? "—"}</span>
                )}
              </strong>
              <StatusPill status={row.validation_status} />
            </div>
          ))}
        </div>
      ) : (
        <EmptyState
          title={emptyTitle}
          body="This output stays unavailable until a versioned prediction is stored. Athena never fills gaps with invented numbers."
        />
      )}
    </>
  );
}
