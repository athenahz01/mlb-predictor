import type { Metadata } from "next";
import { AccountPanel } from "@/components/account-panel";
import { PageHeading } from "@/components/page-heading";

export const metadata: Metadata = { title: "Profile" };

export default function ProfilePage() {
  return (
    <div className="page narrow-page">
      <PageHeading
        eyebrow="Your account"
        title="Profile & preferences"
        description="Control how much detail Athena shows, what appears first, and where game times are displayed."
      />
      <AccountPanel />
    </div>
  );
}
