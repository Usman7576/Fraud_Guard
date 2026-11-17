"use client";

import clsx from "clsx";
import { Activity, Boxes, FlaskConical, Inbox, LayoutDashboard } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/console", label: "Overview", icon: LayoutDashboard, exact: true, admin: false },
  { href: "/console/cases", label: "Review queue", icon: Inbox, exact: false, admin: false },
  { href: "/console/playground", label: "Score playground", icon: FlaskConical, exact: false, admin: false },
  { href: "/console/monitoring", label: "Monitoring", icon: Activity, exact: false, admin: false },
  { href: "/console/models", label: "Models", icon: Boxes, exact: false, admin: true },
];

export function ConsoleNav({ isAdmin = false }: { isAdmin?: boolean }) {
  const path = usePathname();
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col" aria-label="Console">
      {ITEMS.filter((i) => isAdmin || !i.admin).map(({ href, label, icon: Icon, exact }) => {
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
