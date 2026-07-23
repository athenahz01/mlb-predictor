import type { Metadata } from "next";
import { PageHeading } from "@/components/page-heading";
import { PredictionExplorer } from "@/components/prediction-explorer";

export const metadata: Metadata = { title: "Batters" };

export default function BattersPage() {
  return (
    <div className="page">
      <PageHeading
        eyebrow="Plate appearance outlooks"
        title="Batters"
        description="Hits, total bases, home runs, runs, and RBI—shown only when the player and lineup inputs are identified."
      />
      <PredictionExplorer category="batter" emptyTitle="No batter forecasts match" />
    </div>
  );
}
