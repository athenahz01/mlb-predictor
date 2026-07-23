"use client";

import { AlertCircle, CalendarDays, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { GameCard, EmptyState, QualityBanner } from "@/components/prediction-ui";
import { PageHeading } from "@/components/page-heading";
import { getToday } from "@/lib/api";
import type { TodayResponse } from "@/lib/types";

function LoadingSlate() {
  return (
    <div className="game-grid" aria-label="Loading today’s slate">
      {[0, 1, 2].map((index) => (
        <div className="game-card skeleton-card" key={index}>
          <span />
          <span />
          <span />
        </div>
      ))}
    </div>
  );
}

export default function TodayPage() {
  const [data, setData] = useState<TodayResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getToday());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The slate could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    let active = true;
    getToday()
      .then((response) => {
        if (active) setData(response);
      })
      .catch((caught) => {
        if (active) {
          setError(caught instanceof Error ? caught.message : "The slate could not be loaded.");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="page">
      <PageHeading
        eyebrow={new Intl.DateTimeFormat("en-US", {
          weekday: "long",
          month: "long",
          day: "numeric",
        }).format(new Date())}
        title="Today’s clearest signals"
        description="A short list of the forecasts with the best combination of model support, calibration, and complete inputs."
        aside={
          <button className="quiet-button" onClick={load} disabled={loading}>
            <RefreshCw size={16} className={loading ? "spin" : ""} />
            Refresh slate
          </button>
        }
      />
      {error && (
        <div className="error-state" role="alert">
          <AlertCircle size={20} />
          <div>
            <strong>The prediction service is unavailable.</strong>
            <p>Start the local API or check the configured API address, then try again.</p>
          </div>
          <button onClick={load}>Try again</button>
        </div>
      )}
      {loading && <LoadingSlate />}
      {!loading && data && (
        <>
          <QualityBanner count={data.games.length} />
          <section className="section-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Best supported</p>
                <h2>Start here</h2>
              </div>
              <span>{data.strongest.length} forecasts</span>
            </div>
            {data.strongest.length ? (
              <div className="game-grid">
                {data.strongest.map((game) => (
                  <GameCard game={game} key={game.game_id} />
                ))}
              </div>
            ) : (
              <EmptyState
                title="No high-support forecasts yet"
                body="Athena is waiting for stronger inputs. Games will appear here as starters and lineups become reliable."
              />
            )}
          </section>
          <section className="section-block avoid-block">
            <div className="section-heading">
              <div>
                <p className="eyebrow">High uncertainty</p>
                <h2>Forecasts to treat carefully</h2>
              </div>
              <span>{data.uncertain.length} flagged</span>
            </div>
            <div className="uncertain-list">
              {data.uncertain.map((game) => (
                <div key={game.game_id}>
                  <CalendarDays size={17} />
                  <strong>{game.away} at {game.home}</strong>
                  <span>
                    {game.data_quality_flags.length
                      ? game.data_quality_flags[0].replaceAll("_", " ")
                      : "limited historical support"}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
