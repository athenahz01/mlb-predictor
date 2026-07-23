import type { GameDetail, Prediction, TodayResponse } from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}/api/v1${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function getToday(date?: string): Promise<TodayResponse> {
  return request(`/today${date ? `?date=${encodeURIComponent(date)}` : ""}`);
}

export function getPredictions(params = ""): Promise<Prediction[]> {
  return request(`/predictions${params ? `?${params}` : ""}`);
}

export function getGame(gameId: string): Promise<GameDetail> {
  return request(`/games/${encodeURIComponent(gameId)}`);
}

export function askAthena(question: string, gameId?: string) {
  return request<{
    answer: string;
    grounded: boolean;
    request_id: string;
    tool_calls: { name: string; arguments: Record<string, unknown> }[];
  }>("/agent/ask", {
    method: "POST",
    body: JSON.stringify({ question, game_id: gameId, detail_level: "balanced" }),
  });
}
