"use client";

import { LineChart } from "@/components/charts";
import { Card, CardHeader } from "@/components/ui";
import { percent, shortDate } from "@/lib/format";
import type { DriftWeek, KpiWeek } from "@/lib/types";

export function MonitoringCharts({ drift, kpis }: { drift: DriftWeek[]; kpis: KpiWeek[] }) {
  // The retraining threshold the worker used (configs/monitoring.yaml), stored with each week.
  const threshold = drift.findLast((d) => d.threshold !== null)?.threshold ?? null;
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <Card>
        <CardHeader title="Drift share" subtitle="Share of monitored features that drifted from the training months" />
        <div className="px-2 pb-3">
          <LineChart
            ariaLabel="Drift share by week"
            format={(v) => percent(v, 0)}
            reference={threshold === null ? undefined : { value: threshold, label: "retrain" }}
            data={drift.map((d) => ({ label: d.week_start ? shortDate(d.week_start) : String(d.week), value: d.drift_share }))}
          />
        </div>
      </Card>
      <Card>
        <CardHeader title="Recall on labelled fraud" subtitle="Filled in as chargebacks arrive, about 30 days late" />
        <div className="px-2 pb-3">
          <LineChart
            ariaLabel="Recall by week"
            format={(v) => percent(v, 0)}
            data={kpis.map((k) => ({ label: shortDate(k.week_start), value: k.recall }))}
          />
        </div>
      </Card>
    </div>
  );
}
