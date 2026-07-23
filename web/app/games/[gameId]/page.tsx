"use client";

import { AlertTriangle, ArrowRight, Clock3 } from "lucide-react";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { PageHeading } from "@/components/page-heading";
import { Percent, StatusPill } from "@/components/prediction-ui";
import { getGame } from "@/lib/api";
import type { GameDetail } from "@/lib/types";

export default function GamePage() {
  const params = useParams<{ gameId: string }>();
  const [detail, setDetail] = useState<GameDetail | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    getGame(decodeURIComponent(params.gameId)).then(setDetail).catch(() => setError(true));
  }, [params.gameId]);
  if (error) {
    return <div className="page"><div className="error-state">This game could not be loaded.</div></div>;
  }
  if (!detail) {
    return <div className="page"><div className="table-skeleton" /></div>;
  }
  const game = detail.game;
  const winner = game.predictions.home_win_probability;
  return (
    <div className="page">
      <PageHeading
        eyebrow={game.lineup_status + " inputs"}
        title={`${game.away_name} at ${game.home_name}`}
        description="The latest valid pregame forecast leads. Every earlier revision remains visible below."
        aside={<StatusPill status={winner?.validation_status ?? "provisional"} />}
      />
      <section className="matchup-hero">
        <div>
          <span>{game.away}</span>
          <strong><Percent value={winner?.probability == null ? null : 1 - winner.probability} /></strong>
          <small>Away win</small>
        </div>
        <div className="matchup-seam" aria-hidden="true">
          <i />
          <span>forecast</span>
          <i />
        </div>
        <div>
          <span>{game.home}</span>
          <strong><Percent value={winner?.probability} /></strong>
          <small>Home win</small>
        </div>
      </section>
      <section className="section-block">
        <div className="section-heading">
          <div><p className="eyebrow">Prediction seam</p><h2>What changed</h2></div>
          <span>{detail.timeline.length} stored revisions</span>
        </div>
        <div className="prediction-seam">
          {detail.timeline.map((row, index) => (
            <article key={row.id}>
              <span className="seam-node">{index + 1}</span>
              <div>
                <header>
                  <strong>{row.statistic.replaceAll("_", " ")}</strong>
                  <time><Clock3 size={14} />{new Date(row.created_at).toLocaleString()}</time>
                </header>
                <p>
                  {row.revision_reason?.replaceAll("_", " ") ?? "Initial forecast"}
                  <ArrowRight size={14} />
                  {row.probability !== null ? <Percent value={row.probability} /> : row.projected_value}
                </p>
                {row.data_quality_flags.length > 0 && (
                  <small><AlertTriangle size={13} />{row.data_quality_flags.join(", ").replaceAll("_", " ")}</small>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
