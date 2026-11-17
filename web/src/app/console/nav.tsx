"use client";

import clsx from "clsx";
import { FlaskConical, Inbox, LayoutDashboard } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/console", label: "Overview", icon: LayoutDashboard, exact: true },
  { href: "/console/cases", label: "Review queue", icon: Inbox, exact: false },
  { href: "/console/playground", label: "Score playground", icon: FlaskConical, exact: false },
];

export function ConsoleNav() {
  const path = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col" aria-label="Console">
      {ITEMS.map(({ href, label, icon: Icon, exact }) => {
        const active = exact ? path === href : path.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            className={clsx(
              "flex shrink-0 items-center gap-2 rounded-lg px-2 py-1.5 text-sm",
              active ? "bg-accent-soft font-medium text-accent" : "text-ink-2 hover:bg-surface-2 hover:text-ink",
            )}
          >
            <Icon aria-hidden className="size-4" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
