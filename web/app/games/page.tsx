import type { Metadata } from "next";
import { PageHeading } from "@/components/page-heading";
import { PredictionExplorer } from "@/components/prediction-explorer";

export const metadata: Metadata = { title: "Games" };

export default function GamesPage() {
  return (
    <div className="page">
      <PageHeading
        eyebrow="Slate & history"
        title="Game forecasts"
        description="Winner, total, first-five, and first-inning outlooks—ordered by what the model can support, not by drama."
      />
      <PredictionExplorer category="game" emptyTitle="No game forecasts match" />
    </div>
  );
}
