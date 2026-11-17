import clsx from "clsx";
import { ArrowUpCircle, CircleSlash, ShieldCheck } from "lucide-react";
import type { Metadata } from "next";

import { MonitoringCharts } from "./monitoring-charts";
import { Card, CardHeader, EmptyState } from "@/components/ui";
import { ApiError, apiGet } from "@/lib/api";
import { count, percent, shortDate } from "@/lib/format";
import type { DriftWeek, KpiWeek, RetrainingEvent } from "@/lib/types";

export const metadata: Metadata = { title: "Monitoring" };

const OUTCOME = {
  promoted: { label: "Promoted", icon: ArrowUpCircle, cls: "text-good-ink bg-good/10" },
  kept_champion: { label: "Champion kept", icon: ShieldCheck, cls: "text-ink-2 bg-surface-2" },
  skipped: { label: "Skipped", icon: CircleSlash, cls: "text-muted bg-surface-2" },
} as const;

function ci(iv: { estimate: number; low: number; high: number }, fmt: (v: number) => string): string {
  return `${fmt(iv.estimate)} [${fmt(iv.low)}, ${fmt(iv.high)}]`;
}

export default async function Monitoring() {
  let drift: DriftWeek[];
  let events: RetrainingEvent[];
  let kpis: KpiWeek[];
  try {
    [drift, events, kpis] = await Promise.all([
      apiGet<DriftWeek[]>("/monitoring/drift"),
      apiGet<RetrainingEvent[]>("/monitoring/retraining"),
      apiGet<KpiWeek[]>("/kpi/weekly"),
    ]);
  } catch (e) {
    if (e instanceof ApiError) return <p className="text-sm text-critical-ink">{e.message}</p>;
    throw e;
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Monitoring</h1>
        <p className="mt-1 text-sm text-muted">
          Drift of each week against the training months, recall once chargebacks arrive, and every retraining decision.
        </p>
      </header>

      {!drift.length ? (
        <Card>
          <EmptyState title="No monitoring data yet">
            Run the replay worker (<code className="font-mono">fraudguard replay</code>) to play later months through the API.
          </EmptyState>
        </Card>
      ) : (
        <>
          <MonitoringCharts drift={drift} kpis={kpis} />

          <Card>
            <CardHeader
              title="Retraining decisions"
              subtitle="A challenger is promoted only if its PR-AUC gain is clearly positive (paired bootstrap) at no extra cost"
            />
            {events.length ? (
              <ol className="divide-y divide-line">
                {events.map((e) => {
                  const o = OUTCOME[e.outcome];
                  const Icon = o.icon;
                  return (
                    <li key={e.ts + e.week} className="grid gap-2 px-5 py-3 sm:grid-cols-[9rem_1fr]">
                      <div>
                        <span className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", o.cls)}>
                          <Icon aria-hidden className="size-3.5" /> {o.label}
                        </span>
                        <div className="mt-1 text-xs text-muted">
                          Week of {e.week_start ? shortDate(e.week_start) : e.week}
                        </div>
                      </div>
                      <div className="text-sm">
                        <p className="text-ink">{e.detail}</p>
                        <p className="mt-0.5 text-xs text-muted">
                          Trigger: {e.trigger}
                          {e.challenger_version ? ` · challenger v${e.challenger_version} vs champion v${e.champion_version}` : ""}
                        </p>
                        {e.comparison && (
                          <p className="tabular mt-1 text-xs text-ink-2">
                            PR-AUC gain {ci(e.comparison.pr_auc_gain, (v) => (v >= 0 ? "+" : "") + v.toFixed(3))} · cost saved per
                            10k {ci(e.comparison.cost_saved_per_10k, (v) => `${v < 0 ? "-" : ""}$${Math.abs(Math.round(v)).toLocaleString("en-US")}`)} ·{" "}
                            {count(e.comparison.n_rows)} rows, {e.comparison.n_fraud} fraud
                          </p>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ol>
            ) : (
              <p className="px-5 pb-5 text-sm text-muted">No retraining has been triggered.</p>
            )}
          </Card>

          <Card className="overflow-x-auto">
            <CardHeader title="Weekly drift" subtitle="Evidently tests on the champion's most important features" />
            <table className="tabular w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr className="border-y border-line">
                  <th className="px-5 py-2 font-medium">Week of</th>
                  <th className="px-5 py-2 text-right font-medium">Transactions</th>
                  <th className="px-5 py-2 text-right font-medium">Drifted</th>
                  <th className="px-5 py-2 font-medium">Most drifted features</th>
                  <th className="px-5 py-2 font-medium">Report</th>
                </tr>
              </thead>
              <tbody>
                {[...drift].reverse().map((d) => (
                  <tr key={d.week} className="border-b border-line last:border-0">
                    <td className="px-5 py-2 text-ink-2">{d.week_start ? shortDate(d.week_start) : d.week}</td>
                    <td className="px-5 py-2 text-right text-ink">{count(d.n_rows)}</td>
                    <td className="px-5 py-2 text-right text-ink">
                      {percent(d.drift_share, 0)} <span className="text-muted">({d.n_drifted}/{d.n_features})</span>
                    </td>
                    <td className="max-w-md px-5 py-2">
                      <span className="font-mono text-xs text-ink-2">{d.top_drifted.slice(0, 4).join(", ") || "—"}</span>
                    </td>
                    <td className="px-5 py-2">
                      {d.report_file ? (
                        <a
                          className="text-accent hover:underline"
                          href={`/api/fg/monitoring/drift/${d.week}/report`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Open
                        </a>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}
