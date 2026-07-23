import type { Metadata } from "next";
import { Heart } from "lucide-react";
import Link from "next/link";
import { PageHeading } from "@/components/page-heading";

export const metadata: Metadata = { title: "Following" };

export default function FollowingPage() {
  return (
    <div className="page">
      <PageHeading
        eyebrow="Your watchlist"
        title="Following"
        description="Teams, hitters, pitchers, and prediction categories you care about will shape your daily briefing."
      />
      <div className="empty-state">
        <span><Heart size={20} /></span>
        <h3>Your watchlist is clear</h3>
        <p>Sign in, then follow a team or player from its forecast page. Athena will surface changes here.</p>
        <Link className="primary-button" href="/profile">Set up your profile</Link>
      </div>
    </div>
  );
}
