"use client";

import { useState } from "react";

import { LineChart } from "@/components/charts";
import { Card, CardHeader } from "@/components/ui";
import { count, moneyCompact, ms, percent, shortDate } from "@/lib/format";
import type { KpiWeek } from "@/lib/types";

// One measure per chart (never two y-scales on one chart), with a table view of the same numbers.
const CHARTS: { key: keyof KpiWeek; title: string; subtitle: string; format: (v: number) => string }[] = [
  { key: "volume", title: "Volume", subtitle: "Transactions scored per week", format: count },
  { key: "decline_rate", title: "Decline rate", subtitle: "Share of transactions declined", format: (v) => percent(v) },
  { key: "review_rate", title: "Review rate", subtitle: "Share sent to an analyst", format: (v) => percent(v) },
  { key: "recall", title: "Recall", subtitle: "Labelled fraud that was declined or reviewed", format: (v) => percent(v, 0) },
  { key: "cost_per_10k", title: "Cost per 10,000", subtitle: "Labelled fraud losses + false declines + reviews; recent weeks rise as chargebacks arrive", format: moneyCompact },
  { key: "p95_latency_ms", title: "p95 latency", subtitle: "Scoring time inside the API", format: ms },
];

export function KpiCharts({ weeks }: { weeks: KpiWeek[] }) {
  const [table, setTable] = useState(false);
  return (
    <section aria-label="Weekly trends" className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink">Weekly trends</h2>
        <button
          type="button"
          onClick={() => setTable((t) => !t)}
          className="rounded-lg px-2 py-1 text-sm text-accent hover:bg-accent-soft"
          aria-pressed={table}
        >
          {table ? "Show charts" : "Show as table"}
        </button>
      </div>
      {table ? (
        <Card className="overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr className="border-b border-line">
                <th className="px-4 py-2 font-medium">Week of</th>
                {CHARTS.map((c) => (
                  <th key={c.key} className="px-4 py-2 text-right font-medium">
                    {c.title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weeks.map((w) => (
                <tr key={w.week} className="border-b border-line last:border-0">
                  <td className="px-4 py-2 text-ink-2">{shortDate(w.week_start)}</td>
                  {CHARTS.map((c) => {
                    const v = w[c.key] as number | null;
                    return (
                      <td key={c.key} className="px-4 py-2 text-right text-ink">
                        {v === null ? "—" : c.format(v)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ) : (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {CHARTS.map((c) => (
            <Card key={c.key}>
              <CardHeader title={c.title} subtitle={c.subtitle} />
              <div className="px-2 pb-3">
                <LineChart
                  ariaLabel={`${c.title} by week`}
                  format={c.format}
                  data={weeks.map((w) => ({ label: shortDate(w.week_start), value: w[c.key] as number | null }))}
                />
              </div>
            </Card>
          ))}
        </div>
      )}
    </section>
  );
}
