import type { Metadata } from "next";
import { PageHeading } from "@/components/page-heading";
import { PredictionExplorer } from "@/components/prediction-explorer";

export const metadata: Metadata = { title: "Pitchers" };

export default function PitchersPage() {
  return (
    <div className="page">
      <PageHeading
        eyebrow="Workload meets matchup"
        title="Starting pitchers"
        description="Strikeout and workload outlooks with confirmation state, uncertainty, and the exact model revision attached."
      />
      <PredictionExplorer category="pitcher" emptyTitle="No pitcher forecasts match" />
    </div>
  );
}
