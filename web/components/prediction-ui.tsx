import { AlertTriangle, Check, ChevronRight, Clock3, Sparkles } from "lucide-react";
import Link from "next/link";
import type { GameSummary, Prediction } from "@/lib/types";

export function Percent({ value }: { value: number | null | undefined }) {
  return (
    <span className="tabular">
      {typeof value === "number" ? `${Math.round(value * 100)}%` : "—"}
    </span>
  );
}

export function StatusPill({
  status,
}: {
  status: Prediction["validation_status"] | "confirmed" | "projected" | "unknown";
}) {
  return <span className={`status status-${status}`}>{status.replace("_", " ")}</span>;
}

export function GameCard({ game }: { game: GameSummary }) {
  const winner = game.predictions.home_win_probability;
  const total = game.predictions.total_over_8_5;
  const home = winner?.probability ?? null;
  const mainReason =
    winner?.evidence.main_reason ??
    "Legacy forecast imported without structured driver contributions.";
  return (
    <article className="game-card">
      <header className="game-card-top">
        <div>
          <p className="eyebrow">{game.lineup_status} lineups</p>
          <h3>
            {game.away} <span>at</span> {game.home}
          </h3>
        </div>
        <StatusPill status={winner?.validation_status ?? "provisional"} />
      </header>
      <div className="win-line">
        <div>
          <span>{game.away}</span>
          <strong className="tabular">
            {home === null ? "—" : `${Math.round((1 - home) * 100)}%`}
          </strong>
        </div>
        <div className="probability-track" aria-label="Win probability split">
          <span style={{ width: `${home === null ? 50 : (1 - home) * 100}%` }} />
        </div>
        <div>
          <span>{game.home}</span>
          <strong>
            <Percent value={home} />
          </strong>
        </div>
      </div>
      <dl className="game-facts">
        <div>
          <dt>Expected total</dt>
          <dd>{total ? <Percent value={total.probability} /> : "Unavailable"}</dd>
        </div>
        <div>
          <dt>Support</dt>
          <dd>{game.support_score >= 0.6 ? "Moderate" : "Limited"}</dd>
        </div>
      </dl>
      <p className="main-reason">{mainReason}</p>
      {game.data_quality_flags.length > 0 && (
        <div className="quality-note">
          <AlertTriangle size={15} />
          {game.data_quality_flags.length} data warning
          {game.data_quality_flags.length === 1 ? "" : "s"}
        </div>
      )}
      <footer>
        <span>
          <Clock3 size={14} />
          Updated {new Date(game.last_updated).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        </span>
        <Link href={`/games/${encodeURIComponent(game.game_id)}`}>
          Game outlook <ChevronRight size={16} />
        </Link>
      </footer>
    </article>
  );
}

export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state">
      <span><Sparkles size={20} /></span>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}

export function QualityBanner({ count }: { count: number }) {
  return (
    <div className="quality-banner">
      <Check size={17} />
      <span>
        Every number below is linked to a stored model revision. {count} active
        forecast{count === 1 ? "" : "s"} on this slate.
      </span>
    </div>
  );
}
