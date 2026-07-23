"use client";

import {
  CalendarDays,
  CircleDot,
  CircleUserRound,
  Heart,
  House,
  MessageCircleQuestion,
  ScanSearch,
  Shield,
  UsersRound,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navigation = [
  { href: "/", label: "Today", icon: House },
  { href: "/games", label: "Games", icon: CalendarDays },
  { href: "/pitchers", label: "Pitchers", icon: ScanSearch },
  { href: "/batters", label: "Batters", icon: UsersRound },
  { href: "/ask-athena", label: "Ask Athena", icon: MessageCircleQuestion },
  { href: "/following", label: "Following", icon: Heart },
  { href: "/profile", label: "Profile", icon: CircleUserRound },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="app-frame">
      <aside className="rail" aria-label="Primary navigation">
        <Link className="brand" href="/" aria-label="Athena Baseball home">
          <span className="brand-mark" aria-hidden="true">
            <CircleDot size={20} strokeWidth={1.7} />
          </span>
          <span>
            <strong>Athena</strong>
            <small>Baseball</small>
          </span>
        </Link>
        <nav className="nav-list">
          {navigation.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                className="nav-item"
                data-active={active}
                href={href}
                key={href}
                aria-current={active ? "page" : undefined}
              >
                <Icon size={18} strokeWidth={1.75} />
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="rail-note">
          <Shield size={16} />
          <p>Evidence first. No guaranteed outcomes.</p>
        </div>
      </aside>
      <main className="main-canvas">{children}</main>
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {navigation.slice(0, 6).map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              href={href}
              key={href}
              data-active={active}
              aria-label={label}
              aria-current={active ? "page" : undefined}
            >
              <Icon size={20} />
              <span>{label === "Ask Athena" ? "Ask" : label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
