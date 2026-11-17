import type { Metadata } from "next";
import { Activity, TriangleAlert } from "lucide-react";
import Link from "next/link";

import { KpiCharts } from "./kpi-charts";
import { Sparkline } from "@/components/charts";
import { Card, CardHeader, EmptyState } from "@/components/ui";
import { ApiError, apiGet } from "@/lib/api";
import { count, money, moneyCompact, ms, percent, shortDate } from "@/lib/format";
import type { CasePage, DriftWeek, Health, KpiWeek, Policy } from "@/lib/types";

export const metadata: Metadata = { title: "Overview" };

function Tile({ label, value, hint, trend }: { label: string; value: string; hint?: string; trend?: (number | null)[] }) {
  return (
    <Card className="px-4 py-3.5">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 flex items-end justify-between gap-2">
        <div className="text-2xl font-semibold tracking-tight text-ink">{value}</div>
        {trend ? <Sparkline values={trend} /> : null}
      </div>
      {hint ? <div className="mt-1 text-xs text-muted">{hint}</div> : null}
    </Card>
  );
}

/** Latest weekly drift check: icon, words and colour together, linking to the monitoring page. */
function DriftBadge({ drift }: { drift: DriftWeek | undefined }) {
  if (!drift) {
    return <span className="text-xs text-muted">No drift checks yet</span>;
  }
  const threshold = drift.threshold ?? 0.3;
  const high = drift.drift_share > threshold;
  const Icon = high ? TriangleAlert : Activity;
  return (
    <Link
      href="/console/monitoring"
      className={
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium hover:underline " +
        (high ? "bg-warning/15 text-warning-ink" : "bg-good/10 text-good-ink")
      }
    >
      <Icon aria-hidden className="size-3.5" />
      {high ? "Drift above the retraining line" : "Drift within limits"}: {percent(drift.drift_share, 0)} of features
      {drift.week_start ? `, week of ${shortDate(drift.week_start)}` : ""}
    </Link>
  );
}

export default async function Overview() {
  let weeks: KpiWeek[];
  let health: Health;
  let policy: Policy;
  let open: CasePage;
  let drift: DriftWeek[];
  try {
    [weeks, health, policy, open, drift] = await Promise.all([
      apiGet<KpiWeek[]>("/kpi/weekly"),
      apiGet<Health>("/health"),
      apiGet<Policy>("/policy"),
      apiGet<CasePage>("/cases?status=open&page_size=1"),
      apiGet<DriftWeek[]>("/monitoring/drift"),
    ]);
  } catch (e) {
    if (e instanceof ApiError) return <ApiDown message={e.message} />;
    throw e;
  }
  const latest = weeks.at(-1);

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink">Overview</h1>
          <p className="mt-1 text-sm text-muted">
            Model v{health.model_version} · {count(health.transactions_in_memory)} transactions in memory · false decline{" "}
            {money(policy.false_decline_cost, 0)}, review {money(policy.review_cost, 0)}, {policy.review_capacity_per_day} reviews a day
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <DriftBadge drift={drift.at(-1)} />
          <Link href="/console/cases" className="text-sm font-medium text-accent hover:underline">
            {count(open.total)} open {open.total === 1 ? "case" : "cases"} →
          </Link>
        </div>
      </header>

      {!latest ? (
        <Card>
          <EmptyState title="No transactions have been scored yet">
            Send transactions to <code className="font-mono">POST /api/v1/score</code> (or run the replay worker) and the
            weekly figures will appear here. Try the score playground in the meantime.
          </EmptyState>
        </Card>
      ) : (
        <>
          <section aria-labelledby="latest-week" className="space-y-3">
            <h2 id="latest-week" className="text-sm font-semibold text-ink">
              Week of {shortDate(latest.week_start)}{" "}
              <span className="font-normal text-muted">(latest week with transactions; trend lines show every week)</span>
            </h2>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <Tile label="Transactions" value={count(latest.volume)} trend={weeks.map((w) => w.volume)} />
              <Tile label="Declined" value={percent(latest.decline_rate)} trend={weeks.map((w) => w.decline_rate)} />
              <Tile label="Sent to review" value={percent(latest.review_rate)} trend={weeks.map((w) => w.review_rate)} />
              <Tile
                label="Recall on labelled fraud"
                value={percent(latest.recall, 0)}
                hint={latest.labelled_fraud ? `${count(latest.labelled_fraud)} labelled fraud` : "No labels yet this week"}
                trend={weeks.map((w) => w.recall)}
              />
              <Tile
                label="Cost per 10,000"
                value={moneyCompact(latest.cost_per_10k)}
                hint={`p95 latency ${ms(latest.p95_latency_ms)}`}
                trend={weeks.map((w) => w.cost_per_10k)}
              />
            </div>
          </section>
          <KpiCharts weeks={weeks} />
        </>
      )}
    </div>
  );
}

function ApiDown({ message }: { message: string }) {
  return (
    <Card className="mx-auto max-w-xl">
      <CardHeader title="Can't load the overview" />
      <p className="px-5 pb-5 text-sm text-ink-2">{message}</p>
    </Card>
  );
}
