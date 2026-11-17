import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { CaseActions } from "./case-actions";
import { ContributionBars } from "@/components/contributions";
import { EntityGraph } from "@/components/entity-graph";
import { Card, CardHeader, DecisionBadge, ProbabilityBar, StatusBadge } from "@/components/ui";
import { ApiError, apiGet, currentUser } from "@/lib/api";
import { dateTime, due, featureValue, money } from "@/lib/format";
import type { CaseDetail } from "@/lib/types";

export const metadata: Metadata = { title: "Case" };

// Facts an analyst reads first, in plain words.
const FACTS: [string, string][] = [
  ["TransactionAmt", "Amount"],
  ["ProductCD", "Product"],
  ["card4", "Card network"],
  ["card6", "Card type"],
  ["P_emaildomain", "Purchaser email"],
  ["R_emaildomain", "Recipient email"],
  ["DeviceType", "Device type"],
  ["DeviceInfo", "Device"],
  ["addr1", "Billing region"],
  ["card_prior_txn", "Earlier transactions on card"],
  ["card_amt_mean", "Card's usual amount"],
  ["vel_card_24h_cnt", "Card transactions, last 24h"],
  ["graph_2hop_fraud_cards", "Fraud cards within two hops"],
];

export default async function CasePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^\d+$/.test(id)) notFound();
  let c: CaseDetail;
  try {
    c = await apiGet<CaseDetail>(`/cases/${id}`);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) notFound();
    throw e;
  }
  const user = await currentUser();

  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <Link href="/console/cases" className="text-sm text-accent hover:underline">
        ← Review queue
      </Link>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="tabular font-mono text-xl font-semibold text-ink">{c.txn_id}</h1>
            <StatusBadge status={c.status} />
            <DecisionBadge decision={c.decision} />
          </div>
          <p className="mt-1 text-sm text-muted">
            Opened {dateTime(c.created_at)} · {c.status === "closed" ? `verdict: ${c.verdict ?? "—"}` : `due ${due(c.sla_due)}`}
            {c.assignee ? ` · assigned to ${c.assignee}` : ""}
          </p>
        </div>
        <div className="text-right">
          <div className="tabular text-3xl font-semibold tracking-tight text-ink">{money(c.amount)}</div>
          <ProbabilityBar p={c.probability} className="mt-1 justify-end" />
        </div>
      </header>

      <div className="grid gap-5 lg:grid-cols-[1fr_22rem]">
        <div className="space-y-5">
          <Card>
            <CardHeader title="Why it was flagged" subtitle="The features that raised the fraud score most" />
            <ul className="space-y-2 px-5 pb-5">
              {c.reasons.map((r, i) => (
                <li key={r.feature} className="flex gap-3 text-sm">
                  <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-critical/10 text-xs font-semibold text-critical-ink">
                    {i + 1}
                  </span>
                  <span className="text-ink">{r.text}</span>
                </li>
              ))}
              {!c.reasons.length && <li className="text-sm text-muted">No single feature stood out.</li>}
            </ul>
          </Card>

          <Card>
            <CardHeader title="Feature contributions" subtitle="SHAP values in log-odds, largest first" />
            <ContributionBars contributions={c.contributions} values={c.features} />
          </Card>

          <Card>
            <CardHeader title="Transaction facts" />
            <dl className="grid gap-x-6 gap-y-2 px-5 pb-5 text-sm sm:grid-cols-2">
              {FACTS.filter(([k]) => k in c.features).map(([k, label]) => (
                <div key={k} className="flex justify-between gap-3 border-b border-line py-1">
                  <dt className="text-muted">{label}</dt>
                  <dd className="tabular text-right text-ink">
                    {k.endsWith("Amt") || k === "card_amt_mean"
                      ? typeof c.features[k] === "number"
                        ? money(c.features[k] as number)
                        : "—"
                      : featureValue(c.features[k])}
                  </dd>
                </div>
              ))}
            </dl>
            <p className="px-5 pb-4 text-xs text-muted">
              Card <span className="font-mono">{c.card_uid}</span> · model input has {Object.keys(c.features).length} features
            </p>
          </Card>
        </div>

        <div className="space-y-5">
          <Card>
            <CardHeader title="Decision" subtitle={c.status === "closed" ? "This case is closed" : "Your verdict writes a label for retraining"} />
            <CaseActions txnId={c.txn_id} status={c.status} assignee={c.assignee} me={user?.username ?? ""} />
          </Card>
          <Card>
            <CardHeader title="Entity neighbourhood" subtitle="Devices, addresses and email domains this card has used" />
            <EntityGraph card={c.card_uid} neighbourhood={c.neighbourhood} />
          </Card>
          <Card>
            <CardHeader title="Notes" />
            <ol className="space-y-3 px-5 pb-5">
              {c.notes.map((n) => (
                <li key={n.at + n.author} className="text-sm">
                  <div className="text-xs text-muted">
                    {n.author} · {dateTime(n.at)}
                  </div>
                  <p className="mt-0.5 whitespace-pre-wrap text-ink">{n.text}</p>
                </li>
              ))}
              {!c.notes.length && <li className="text-sm text-muted">No notes yet.</li>}
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}
