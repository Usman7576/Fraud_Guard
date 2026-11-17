"use client";

import clsx from "clsx";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, useTransition } from "react";

import { LineChart } from "@/components/charts";
import { Button, Card, CardHeader } from "@/components/ui";
import { count, money, moneyCompact, percent } from "@/lib/format";
import type { Cost, Policy, Simulation } from "@/lib/types";

type Key = keyof Policy;

const SLIDERS: { key: Key; label: string; min: number; max: number; step: number; fmt: (v: number) => string; help: string }[] = [
  {
    key: "false_decline_cost",
    label: "False decline cost",
    min: 1,
    max: 100,
    step: 1,
    fmt: (v) => money(v, 0),
    help: "What declining a genuine customer costs: lost margin, support time, churn.",
  },
  {
    key: "review_cost",
    label: "Review cost",
    min: 0,
    max: 30,
    step: 0.5,
    fmt: (v) => money(v, v % 1 ? 2 : 0),
    help: "Analyst time for one case.",
  },
  {
    key: "review_capacity_per_day",
    label: "Reviews per day",
    min: 0,
    max: 1000,
    step: 10,
    fmt: (v) => count(v),
    help: "Above this, the next best of approve or decline is used.",
  },
  {
    key: "risk_multiplier",
    label: "Risk multiplier",
    min: 0.25,
    max: 4,
    step: 0.05,
    fmt: (v) => `${v.toFixed(2)}×`,
    help: "Scales the fraud odds before costs are compared. 1× trusts the calibrated score as it is.",
  },
];

function Delta({ proposed, current, lowerIsBetter, fmt }: { proposed: number; current: number; lowerIsBetter: boolean; fmt: (v: number) => string }) {
  const d = proposed - current;
  if (Math.abs(d) < 1e-9) return <span className="text-muted">same as now</span>;
  const better = lowerIsBetter ? d < 0 : d > 0;
  return (
    <span className={better ? "text-good-ink" : "text-critical-ink"}>
      {d > 0 ? "+" : "−"}
      {fmt(Math.abs(d))} vs now
    </span>
  );
}

function Tile({
  label,
  value,
  proposed,
  current,
  lowerIsBetter,
  fmt,
}: {
  label: string;
  value: string;
  proposed: number;
  current: number;
  lowerIsBetter: boolean;
  fmt: (v: number) => string;
}) {
  return (
    <Card className="p-4">
      <p className="text-xs text-muted">{label}</p>
      <p className="tabular mt-1 text-2xl font-semibold text-ink">{value}</p>
      <p className="tabular mt-1 text-xs">
        <Delta proposed={proposed} current={current} lowerIsBetter={lowerIsBetter} fmt={fmt} />
      </p>
    </Card>
  );
}

function Breakdown({ proposed, current }: { proposed: Cost; current: Cost }) {
  const rows: [string, keyof Cost][] = [
    ["Fraud let through", "fraud_loss"],
    ["False declines", "false_decline_cost"],
    ["Reviews", "review_cost"],
    ["Total", "total"],
  ];
  return (
    <table className="tabular w-full text-sm">
      <thead className="text-left text-xs text-muted">
        <tr className="border-y border-line">
          <th className="px-5 py-2 font-medium">Cost on the validation month</th>
          <th className="px-5 py-2 text-right font-medium">Now</th>
          <th className="px-5 py-2 text-right font-medium">Proposed</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([label, k]) => (
          <tr key={k} className={clsx("border-b border-line last:border-0", k === "total" && "font-medium")}>
            <td className="px-5 py-2 text-ink-2">{label}</td>
            <td className="px-5 py-2 text-right text-ink">{money(current[k], 0)}</td>
            <td className="px-5 py-2 text-right text-ink">{money(proposed[k], 0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function CostExplorer({ current }: { current: Policy }) {
  const router = useRouter();
  const [policy, setPolicy] = useState<Policy>(current);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [applying, startApply] = useTransition();
  const changed = SLIDERS.some(({ key }) => policy[key] !== current[key]);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch("/api/fg/policy/simulate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(policy),
          signal: ctrl.signal,
        });
        const body = (await res.json().catch(() => ({}))) as Simulation & { detail?: unknown };
        if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Simulation failed (${res.status})`);
        setSim(body);
        setError(null);
      } catch (e) {
        if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : "Simulation failed");
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 200);
    return () => {
      ctrl.abort();
      window.clearTimeout(t);
    };
  }, [policy]);

  const markerIndex = useMemo(() => {
    if (!sim) return -1;
    let best = 0;
    sim.curve.forEach((c, i) => {
      if (Math.abs(c.false_decline_cost - policy.false_decline_cost) < Math.abs(sim.curve[best]!.false_decline_cost - policy.false_decline_cost)) best = i;
    });
    return best;
  }, [sim, policy.false_decline_cost]);

  function apply() {
    if (!window.confirm("Apply this policy to every new transaction? This is recorded in the audit log.")) return;
    startApply(async () => {
      const res = await fetch("/api/fg/policy", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(policy),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { detail?: unknown };
        setError(typeof body.detail === "string" ? body.detail : `Could not apply the policy (${res.status})`);
        return;
      }
      router.refresh();
    });
  }

  const p = sim?.proposed;
  const c = sim?.current;
  return (
    <div className="grid gap-5 lg:grid-cols-[20rem_1fr]">
      <Card className="h-fit p-5">
        <div className="space-y-5">
          {SLIDERS.map((s) => (
            <div key={s.key}>
              <div className="flex items-baseline justify-between gap-2">
                <label htmlFor={s.key} className="text-sm font-medium text-ink">
                  {s.label}
                </label>
                <span className="tabular text-sm text-ink">{s.fmt(policy[s.key])}</span>
              </div>
              <input
                id={s.key}
                type="range"
                min={s.min}
                max={s.max}
                step={s.step}
                value={policy[s.key]}
                onChange={(e) => setPolicy((old) => ({ ...old, [s.key]: Number(e.target.value) }))}
                className="mt-2 w-full accent-[var(--accent)]"
              />
              <p className="mt-1 text-xs text-muted">
                {s.help} Now: <span className="tabular">{s.fmt(current[s.key])}</span>.
              </p>
            </div>
          ))}
        </div>
        <div className="mt-6 flex gap-2">
          <Button variant="secondary" disabled={!changed || applying} onClick={() => setPolicy(current)}>
            Reset
          </Button>
          <Button className="flex-1" disabled={!changed || applying || !!error} onClick={apply}>
            {applying ? "Applying…" : "Apply this policy"}
          </Button>
        </div>
      </Card>

      <div className={clsx("min-w-0 space-y-5 transition-opacity", loading && sim && "opacity-60")} aria-busy={loading}>
        {error && <p className="rounded-lg border border-critical/40 bg-critical/10 px-4 py-3 text-sm text-critical-ink">{error}</p>}
        {!sim && !error && <p className="text-sm text-muted">Scoring the validation month…</p>}
        {sim && p && c && (
          <>
            <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
              <Tile label="Cost per 10,000 payments" value={moneyCompact(p.per_10k)} proposed={p.per_10k} current={c.per_10k} lowerIsBetter fmt={(v) => moneyCompact(v)} />
              <Tile label="Fraud stopped" value={percent(p.fraud_caught)} proposed={p.fraud_caught} current={c.fraud_caught} lowerIsBetter={false} fmt={(v) => `${(v * 100).toFixed(1)} pts`} />
              <Tile label="Declined" value={percent(p.decline_rate, 2)} proposed={p.decline_rate} current={c.decline_rate} lowerIsBetter fmt={(v) => `${(v * 100).toFixed(2)} pts`} />
              <Tile label="Sent to review" value={percent(p.review_rate, 2)} proposed={p.review_rate} current={c.review_rate} lowerIsBetter fmt={(v) => `${(v * 100).toFixed(2)} pts`} />
            </div>

            <Card>
              <CardHeader
                title="Cost per 10,000 payments by false-decline cost"
                subtitle="Everything else as set on the left. The line marks your false-decline cost."
              />
              <div className="px-3 pb-4">
                <LineChart
                  data={sim.curve.map((pt) => ({ label: money(pt.false_decline_cost, 0), value: pt.per_10k }))}
                  format={(v) => moneyCompact(v)}
                  height={200}
                  ariaLabel="Cost per 10,000 payments as the false-decline cost goes from $5 to $100"
                  marker={{ index: markerIndex, label: "your setting" }}
                />
              </div>
            </Card>

            <div className="grid gap-5 xl:grid-cols-2">
              <Card className="overflow-x-auto">
                <CardHeader
                  title="Thresholds this policy implies"
                  subtitle="Lowest fraud probability that is reviewed or declined, by amount (before the review capacity)"
                />
                <table className="tabular w-full text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr className="border-y border-line">
                      <th className="px-5 py-2 font-medium">Amount</th>
                      <th className="px-5 py-2 text-right font-medium">Review or decline from</th>
                      <th className="px-5 py-2 text-right font-medium">Decline from</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sim.thresholds.map((t) => (
                      <tr key={t.amount} className="border-b border-line last:border-0">
                        <td className="px-5 py-2 text-ink-2">{money(t.amount, 0)}</td>
                        <td className="px-5 py-2 text-right text-ink">{t.act_from >= 1 ? "never" : percent(t.act_from, 2)}</td>
                        <td className="px-5 py-2 text-right text-ink">{t.decline_from >= 1 ? "never" : percent(t.decline_from, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
              <Card className="overflow-x-auto">
                <CardHeader title="Where the money goes" subtitle={`${count(sim.n_rows)} payments, ${count(sim.n_fraud)} fraud, model v${sim.model_version}`} />
                <Breakdown proposed={p} current={c} />
                <p className="px-5 py-3 text-xs text-muted">
                  The policy was tuned on this month, so these are in-sample figures. The untouched test month is in the evaluation
                  report.
                </p>
              </Card>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
