"use client";

import clsx from "clsx";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { ProbabilityBar, StatusBadge } from "@/components/ui";
import { due, money } from "@/lib/format";
import type { CaseSummary } from "@/lib/types";

export function QueueTable({ items }: { items: CaseSummary[] }) {
  const router = useRouter();
  const [active, setActive] = useState(0);
  const rows = useRef<(HTMLTableRowElement | null)[]>([]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (target.closest("input, select, textarea, [contenteditable]")) return;
      if (e.key === "j" || e.key === "ArrowDown") setActive((i) => Math.min(i + 1, items.length - 1));
      else if (e.key === "k" || e.key === "ArrowUp") setActive((i) => Math.max(i - 1, 0));
      else if (e.key === "Enter" && items[active]) router.push(`/console/cases/${items[active].txn_id}`);
      else return;
      e.preventDefault();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, items, router]);

  useEffect(() => {
    // Braces matter: newer browsers return a Promise from scrollIntoView, and an effect must not return one.
    rows.current[active]?.scrollIntoView({ block: "nearest" });
  }, [active]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted">
          <tr className="border-b border-line">
            <th className="px-4 py-2.5 font-medium">Transaction</th>
            <th className="px-4 py-2.5 font-medium">Risk</th>
            <th className="px-4 py-2.5 text-right font-medium">Amount</th>
            <th className="px-4 py-2.5 font-medium">Top reasons</th>
            <th className="px-4 py-2.5 font-medium">Status</th>
            <th className="px-4 py-2.5 font-medium">Due</th>
          </tr>
        </thead>
        <tbody>
          {items.map((c, i) => (
            <tr
              key={c.txn_id}
              ref={(el) => {
                rows.current[i] = el;
              }}
              onClick={() => router.push(`/console/cases/${c.txn_id}`)}
              onMouseEnter={() => setActive(i)}
              aria-selected={i === active}
              className={clsx(
                "cursor-pointer border-b border-line last:border-0",
                i === active ? "bg-accent-soft" : "hover:bg-surface-2",
              )}
            >
              <td className="px-4 py-3">
                <a
                  href={`/console/cases/${c.txn_id}`}
                  className="tabular font-mono text-xs text-ink hover:underline"
                  onClick={(e) => e.stopPropagation()}
                >
                  {c.txn_id}
                </a>
                {c.assignee ? <div className="mt-0.5 text-xs text-muted">{c.assignee}</div> : null}
              </td>
              <td className="px-4 py-3">
                <ProbabilityBar p={c.probability} />
              </td>
              <td className="tabular px-4 py-3 text-right text-ink">{money(c.amount)}</td>
              <td className="max-w-md px-4 py-3">
                <div className="flex flex-wrap gap-1">
                  {c.reasons.slice(0, 2).map((r) => (
                    <span key={r.feature} className="truncate rounded-md bg-surface-2 px-1.5 py-0.5 text-xs text-ink-2" title={r.text}>
                      {r.text}
                    </span>
                  ))}
                </div>
              </td>
              <td className="px-4 py-3">
                <StatusBadge status={c.status} />
              </td>
              <td className={clsx("px-4 py-3 text-xs", due(c.sla_due).includes("overdue") ? "text-critical-ink" : "text-muted")}>
                {c.status === "closed" ? "—" : due(c.sla_due)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
