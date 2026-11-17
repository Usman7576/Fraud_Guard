import type { Metadata } from "next";

import { PromoteButton } from "./promote-button";
import { Card, CardHeader, EmptyState } from "@/components/ui";
import { ApiError, apiGet, currentUser } from "@/lib/api";
import { dateTime, moneyCompact } from "@/lib/format";
import type { ModelVersion } from "@/lib/types";

export const metadata: Metadata = { title: "Models" };

function metric(m: ModelVersion, key: string, fmt: (v: number) => string): string {
  const v = m.metrics[key];
  return v === undefined ? "—" : fmt(v);
}

export default async function Models() {
  const user = await currentUser();
  if (user?.role !== "admin") {
    return (
      <Card className="mx-auto max-w-xl">
        <EmptyState title="Admins only">Model versions and promotion are managed by an admin.</EmptyState>
      </Card>
    );
  }
  let versions: ModelVersion[];
  try {
    versions = await apiGet<ModelVersion[]>("/models");
  } catch (e) {
    if (e instanceof ApiError) return <p className="text-sm text-critical-ink">{e.message}</p>;
    throw e;
  }

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-ink">Models</h1>
        <p className="mt-1 text-sm text-muted">
          Every registered version of <code className="font-mono">fraudguard</code>. Promoting a version serves it immediately
          and is recorded in the audit log; promoting an older one is a rollback.
        </p>
      </header>
      <Card className="overflow-x-auto">
        <CardHeader title="Registry" subtitle="Champions are trained by `fraudguard train lightgbm`; challengers by the replay worker" />
        <table className="tabular w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr className="border-y border-line">
              <th className="px-5 py-2 font-medium">Version</th>
              <th className="px-5 py-2 font-medium">Role</th>
              <th className="px-5 py-2 font-medium">Registered</th>
              <th className="px-5 py-2 text-right font-medium">Validation PR-AUC</th>
              <th className="px-5 py-2 text-right font-medium">Validation cost / 10k</th>
              <th className="px-5 py-2 text-right font-medium">Challenger PR-AUC gain</th>
              <th className="px-5 py-2" />
            </tr>
          </thead>
          <tbody>
            {versions.map((m) => (
              <tr key={m.version} className="border-b border-line last:border-0">
                <td className="px-5 py-2.5 font-mono text-ink">v{m.version}</td>
                <td className="px-5 py-2.5">
                  <div className="flex flex-wrap gap-1">
                    {m.serving && (
                      <span className="rounded-full bg-good/10 px-2 py-0.5 text-xs font-medium text-good-ink">Serving</span>
                    )}
                    {m.aliases.map((a) => (
                      <span key={a} className="rounded-full border border-line px-2 py-0.5 text-xs text-ink-2">
                        {a}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-5 py-2.5 text-ink-2">{m.created_at ? dateTime(m.created_at) : "—"}</td>
                <td className="px-5 py-2.5 text-right text-ink">{metric(m, "valid_raw_pr_auc", (v) => v.toFixed(3))}</td>
                <td className="px-5 py-2.5 text-right text-ink">{metric(m, "valid_cost_per_10k", moneyCompact)}</td>
                <td className="px-5 py-2.5 text-right text-ink">
                  {m.metrics.eval_pr_auc_gain === undefined
                    ? "—"
                    : `${m.metrics.eval_pr_auc_gain >= 0 ? "+" : ""}${m.metrics.eval_pr_auc_gain.toFixed(3)}`}
                </td>
                <td className="px-5 py-2.5 text-right">{!m.serving && <PromoteButton version={m.version} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
