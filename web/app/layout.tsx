import type { Metadata } from "next";
import { Instrument_Sans, Newsreader } from "next/font/google";
import { AppShell } from "@/components/app-shell";
import "./globals.css";

const instrument = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-dugout",
  display: "swap",
});
const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-scorecard",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: { default: "Athena Baseball", template: "%s · Athena Baseball" },
  description:
    "Evidence-grounded MLB forecasts for games, pitchers, and batters.",
  openGraph: {
    title: "Athena Baseball",
    description: "Every forecast has a reason.",
    type: "website",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "Athena Baseball" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Athena Baseball",
    description: "Every forecast has a reason.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${instrument.variable} ${newsreader.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
