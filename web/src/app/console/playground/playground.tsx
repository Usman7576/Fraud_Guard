"use client";

import { useState, useTransition } from "react";

import { ContributionBars } from "@/components/contributions";
import { Button, Card, CardHeader, DecisionBadge, Input, Label, Select } from "@/components/ui";
import { money, ms } from "@/lib/format";
import type { PreviewResponse } from "@/lib/types";

interface FormState {
  TransactionAmt: string;
  ProductCD: string;
  card1: string;
  card4: string;
  card6: string;
  addr1: string;
  addr2: string;
  P_emaildomain: string;
  DeviceType: string;
  DeviceInfo: string;
  D1: string;
}

const START: FormState = {
  TransactionAmt: "249.00",
  ProductCD: "W",
  card1: "13926",
  card4: "visa",
  card6: "debit",
  addr1: "315",
  addr2: "87",
  P_emaildomain: "gmail.com",
  DeviceType: "",
  DeviceInfo: "",
  D1: "14",
};

const PRESETS: { label: string; values: Partial<FormState> }[] = [
  { label: "Everyday purchase", values: START },
  {
    label: "New card, big ticket",
    values: { TransactionAmt: "1899.00", ProductCD: "C", card1: "9500", D1: "0", P_emaildomain: "anonymous.com", DeviceType: "mobile", DeviceInfo: "" },
  },
  {
    label: "Small, odd cents",
    values: { TransactionAmt: "37.418", ProductCD: "C", card1: "4141", D1: "2", P_emaildomain: "outlook.com", DeviceType: "desktop", DeviceInfo: "Windows" },
  },
];

const PAST = { approve: "approved", review: "sent to review", decline: "declined" } as const;

/** Form strings -> API payload: empty fields are sent as null, numbers as numbers. */
export function toPayload(f: FormState): Record<string, string | number | null> {
  const num = (v: string) => (v.trim() === "" ? null : Number(v));
  const str = (v: string) => (v.trim() === "" ? null : v.trim());
  return {
    TransactionAmt: Number(f.TransactionAmt),
    ProductCD: str(f.ProductCD),
    card1: num(f.card1),
    card4: str(f.card4),
    card6: str(f.card6),
    addr1: num(f.addr1),
    addr2: num(f.addr2),
    P_emaildomain: str(f.P_emaildomain),
    DeviceType: str(f.DeviceType),
    DeviceInfo: str(f.DeviceInfo),
    D1: num(f.D1),
  };
}

function Gauge({ p }: { p: number }) {
  // Half-circle meter: the arc fill carries severity, the track is a lighter step of the same tone.
  const tone = p >= 0.5 ? "var(--critical)" : p >= 0.1 ? "var(--warning)" : "var(--accent)";
  const r = 70;
  const len = Math.PI * r;
  return (
    <svg viewBox="0 0 180 104" className="mx-auto w-full max-w-[220px]" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p * 100)} aria-label="Fraud probability">
      <path d="M20 92 A70 70 0 0 1 160 92" fill="none" stroke={tone} strokeOpacity={0.18} strokeWidth={12} strokeLinecap="round" />
      <path
        d="M20 92 A70 70 0 0 1 160 92"
        fill="none"
        stroke={tone}
        strokeWidth={12}
        strokeLinecap="round"
        strokeDasharray={`${Math.max(p, 0.01) * len} ${len}`}
        className="transition-[stroke-dasharray] duration-700 ease-out"
      />
      <text x="90" y="80" textAnchor="middle" className="tabular fill-[var(--ink)] text-[26px] font-semibold">
        {(p * 100).toFixed(p < 0.1 ? 1 : 0)}%
      </text>
      <text x="90" y="98" textAnchor="middle" className="fill-[var(--muted)] text-[10px]">
        fraud probability
      </text>
    </svg>
  );
}

export function Playground() {
  const [form, setForm] = useState<FormState>(START);
  const [result, setResult] = useState<PreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const set = (k: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    start(async () => {
      const res = await fetch("/api/fg/score/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(toPayload(form)),
      });
      const body = (await res.json().catch(() => ({}))) as PreviewResponse & { detail?: unknown };
      if (!res.ok) {
        setError(typeof body.detail === "string" ? body.detail : "Check the fields: some values are not valid.");
        return;
      }
      setResult(body);
    });
  }

  const field = (k: keyof FormState, label: string, props: React.InputHTMLAttributes<HTMLInputElement> = {}) => (
    <div>
      <Label htmlFor={k}>{label}</Label>
      <Input id={k} value={form[k]} onChange={set(k)} {...props} />
    </div>
  );

  return (
    <div className="grid gap-5 lg:grid-cols-[24rem_1fr]">
      <Card>
        <CardHeader title="Transaction" subtitle="IEEE-CIS fields; leave a field empty if unknown" />
        <form onSubmit={submit} className="space-y-3 px-5 pb-5">
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setForm((f) => ({ ...f, ...p.values }))}
                className="rounded-full border border-line px-2.5 py-1 text-xs text-ink-2 hover:bg-surface-2"
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-3">
            {field("TransactionAmt", "Amount (USD)", { inputMode: "decimal", required: true })}
            <div>
              <Label htmlFor="ProductCD">Product</Label>
              <Select id="ProductCD" value={form.ProductCD} onChange={set("ProductCD")} className="w-full">
                {["W", "C", "R", "H", "S"].map((v) => (
                  <option key={v}>{v}</option>
                ))}
              </Select>
            </div>
            {field("card1", "Card (card1)", { inputMode: "numeric" })}
            {field("D1", "Days since card first seen (D1)", { inputMode: "numeric" })}
            <div>
              <Label htmlFor="card4">Network</Label>
              <Select id="card4" value={form.card4} onChange={set("card4")} className="w-full">
                {["visa", "mastercard", "american express", "discover", ""].map((v) => (
                  <option key={v} value={v}>{v || "unknown"}</option>
                ))}
              </Select>
            </div>
            <div>
              <Label htmlFor="card6">Card type</Label>
              <Select id="card6" value={form.card6} onChange={set("card6")} className="w-full">
                {["debit", "credit", ""].map((v) => (
                  <option key={v} value={v}>{v || "unknown"}</option>
                ))}
              </Select>
            </div>
            {field("addr1", "Billing region (addr1)", { inputMode: "numeric" })}
            {field("addr2", "Country (addr2)", { inputMode: "numeric" })}
          </div>
          {field("P_emaildomain", "Purchaser email domain")}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <Label htmlFor="DeviceType">Device type</Label>
              <Select id="DeviceType" value={form.DeviceType} onChange={set("DeviceType")} className="w-full">
                {["", "desktop", "mobile"].map((v) => (
                  <option key={v} value={v}>{v || "unknown"}</option>
                ))}
              </Select>
            </div>
            {field("DeviceInfo", "Device")}
          </div>
          <Button type="submit" className="w-full" disabled={pending}>
            {pending ? "Scoring…" : "Score transaction"}
          </Button>
          {error ? (
            <p role="alert" className="rounded-lg bg-critical/10 px-3 py-2 text-sm text-critical-ink">
              {error}
            </p>
          ) : null}
        </form>
      </Card>

      <div className="space-y-5" aria-live="polite">
        {result ? (
          <>
            <Card>
              <div className="grid items-center gap-4 p-5 sm:grid-cols-[220px_1fr]">
                <Gauge p={result.probability} />
                <div>
                  <DecisionBadge decision={result.decision} />
                  <p className="mt-2 text-sm text-ink-2">
                    {money(Number(form.TransactionAmt) || 0)} would be{" "}
                    <strong className="text-ink">{PAST[result.decision]}</strong>{" "}
                    by model v{result.model_version}, scored in {ms(result.latency_ms)}.
                  </p>
                  <ol className="mt-3 space-y-1.5">
                    {result.reasons.map((r, i) => (
                      <li key={r.feature} className="flex gap-2 text-sm text-ink">
                        <span className="text-muted">{i + 1}.</span>
                        {r.text}
                      </li>
                    ))}
                    {!result.reasons.length && <li className="text-sm text-muted">No feature raised the risk noticeably.</li>}
                  </ol>
                </div>
              </div>
            </Card>
            <Card>
              <CardHeader title="Feature contributions" subtitle="SHAP values in log-odds" />
              <ContributionBars contributions={result.contributions} values={result.features} />
            </Card>
          </>
        ) : (
          <Card>
            <p className="px-5 py-14 text-center text-sm text-muted">Fill in a transaction and score it to see the decision.</p>
          </Card>
        )}
      </div>
    </div>
  );
}
