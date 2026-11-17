import type { Metadata } from "next";

import { CostExplorer } from "./explorer";
import { Card, EmptyState } from "@/components/ui";
import { ApiError, apiGet, currentUser } from "@/lib/api";
import type { Policy } from "@/lib/types";

export const metadata: Metadata = { title: "Cost explorer" };

export default async function Cost() {
  const user = await currentUser();
  if (user?.role !== "admin") {
    return (
      <Card className="mx-auto max-w-xl">
        <EmptyState title="Admins only">The cost policy is set by an admin.</EmptyState>
      </Card>
    );
  }
  let policy: Policy;
  try {
    policy = await apiGet<Policy>("/policy");
  } catch (e) {
    if (e instanceof ApiError) return <p className="text-sm text-critical-ink">{e.message}</p>;
    throw e;
  }
  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-ink">Cost explorer</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          Change the cost assumptions and see what they would have cost on the validation month, scored by the serving model.
          Nothing changes until you apply the policy; applying it is recorded in the audit log.
        </p>
      </header>
      <CostExplorer current={policy} />
    </div>
  );
}
