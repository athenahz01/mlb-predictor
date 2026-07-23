export type Prediction = {
  id: string;
  game_id: string;
  category: "game" | "team" | "pitcher" | "batter";
  statistic: string;
  player_id: number | null;
  team_id: number | null;
  probability: number | null;
  projected_value: number | null;
  interval: [number, number] | null;
  model_version: string;
  lineup_status: string;
  confidence: "low" | "medium" | "high";
  validation_status: "validated" | "provisional" | "experimental" | "unavailable";
  data_quality_flags: string[];
  evidence: {
    main_reason?: string;
    main_uncertainty?: string;
    legacy_meta?: Record<string, unknown>;
    [key: string]: unknown;
  };
  revision_number: number;
  revision_reason: string | null;
  created_at: string;
};

export type GameSummary = {
  game_id: string;
  mlb_game_pk: number | null;
  away: string;
  home: string;
  away_name: string;
  home_name: string;
  start_time: string | null;
  lineup_status: string;
  last_updated: string;
  support_score: number;
  data_quality_flags: string[];
  predictions: Record<string, Prediction>;
};

export type TodayResponse = {
  date: string;
  generated_at: string;
  games: GameSummary[];
  strongest: GameSummary[];
  uncertain: GameSummary[];
  waiting_for_lineups: GameSummary[];
};

export type GameDetail = {
  game: GameSummary;
  timeline: Prediction[];
  evaluation_tracks: {
    initial: Prediction[];
    latest_pregame: Prediction[];
  };
};
