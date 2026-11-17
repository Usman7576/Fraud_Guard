"use client";

import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { CircleCheck, CircleX, Eye, LoaderCircle } from "lucide-react";
import dynamic from "next/dynamic";
import { useState } from "react";

import {
  CARD_TYPES,
  DEVICES,
  EMAILS,
  GAUGE_TICKS,
  NETWORKS,
  PRESETS,
  PRODUCTS,
  gaugePosition,
  tickLabel,
  type DemoInput,
  type DemoResult,
} from "@/lib/landing";

import { use3DMode, useOnScreen } from "./use-3d";

function SvgGauge({ value }: { value: number }) {
  const angle = Math.PI - value * Math.PI;
  const r = 100;
  const arc = (to: number) => {
    const a = Math.PI - to * Math.PI;
    return `M${-r},0 A${r},${r} 0 0 1 ${(Math.cos(a) * r).toFixed(2)},${(-Math.sin(a) * r).toFixed(2)}`;
  };
  return (
    <svg viewBox="-150 -135 300 160" className="size-full" aria-hidden>
      <path d={arc(1)} fill="none" stroke="var(--line)" strokeWidth={12} strokeLinecap="round" />
      {value > 0.002 && <path d={arc(value)} fill="none" stroke="var(--accent)" strokeWidth={12} strokeLinecap="round" />}
      {GAUGE_TICKS.map((t) => {
        const a = Math.PI - gaugePosition(t) * Math.PI;
        return (
          <text key={t} x={Math.cos(a) * 128} y={-Math.sin(a) * 128} dy="0.35em" textAnchor="middle" fontSize={10} fill="var(--muted)">
            {tickLabel(t)}
          </text>
        );
      })}
      <line x1={0} y1={0} x2={Math.cos(angle) * 82} y2={-Math.sin(angle) * 82} stroke="var(--ink)" strokeWidth={3} strokeLinecap="round" />
      <circle r={8} fill="var(--accent)" />
    </svg>
  );
}

const GaugeScene = dynamic(() => import("./gauge-scene"), { ssr: false });

function Gauge({ value }: { value: number }) {
  const mode = use3DMode();
  const [ref, onScreen] = useOnScreen<HTMLDivElement>();
  return (
    <div ref={ref} className="aspect-[15/8] w-full">
      {mode === "3d" ? <GaugeScene value={value} active={onScreen} /> : <SvgGauge value={value} />}
    </div>
  );
}

const DECISION = {
  approve: { label: "Approve", icon: CircleCheck, cls: "bg-good/15 text-good-ink" },
  review: { label: "Send to review", icon: Eye, cls: "bg-warning/15 text-warning-ink" },
  decline: { label: "Decline", icon: CircleX, cls: "bg-critical/15 text-critical-ink" },
} as const;

const field = "h-9 w-full rounded-lg border border-line bg-surface-2 px-2.5 text-sm text-ink";

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly (T | { value: T; label: string })[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-ink-2">{label}</span>
      <select className={clsx(field, "mt-1")} value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => {
          const v = typeof o === "string" ? o : o.value;
          return (
            <option key={v} value={v}>
              {typeof o === "string" ? o : o.label}
            </option>
          );
        })}
      </select>
    </label>
  );
}

export function ScoreDemo() {
  const [input, setInput] = useState<DemoInput>(PRESETS[1]!.input);
  const [result, setResult] = useState<DemoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const set = <K extends keyof DemoInput>(k: K) => (v: DemoInput[K]) => setInput((s) => ({ ...s, [k]: v }));

  async function score(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await fetch("/api/demo/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      const body = (await r.json().catch(() => ({}))) as DemoResult & { detail?: unknown };
      if (!r.ok) {
        setError(typeof body.detail === "string" ? body.detail : "Check the amount and try again.");
        return;
      }
      setResult(body);
    } catch {
      setError("The demo could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  const d = result ? DECISION[result.decision] : null;
  return (
    <div className="grid gap-6 rounded-2xl border border-line bg-surface p-5 sm:p-6 lg:grid-cols-[1fr_1.1fr]">
      <form onSubmit={score} className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => setInput(p.input)}
              className="rounded-full border border-line px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-ink"
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="text-xs font-medium text-ink-2">Amount (USD)</span>
            <input
              className={clsx(field, "tabular mt-1")}
              type="number"
              min={0.01}
              max={10000}
              step={0.01}
              required
              value={input.TransactionAmt}
              onChange={(e) => set("TransactionAmt")(Number(e.target.value))}
            />
          </label>
          <Select label="Product" value={input.ProductCD} options={PRODUCTS} onChange={set("ProductCD")} />
          <Select label="Card network" value={input.card4} options={NETWORKS} onChange={set("card4")} />
          <Select label="Card type" value={input.card6} options={CARD_TYPES} onChange={set("card6")} />
          <Select label="Email domain" value={input.P_emaildomain} options={EMAILS} onChange={set("P_emaildomain")} />
          <Select label="Device" value={input.DeviceType} options={DEVICES} onChange={set("DeviceType")} />
        </div>
        <label className="block">
          <span className="flex justify-between text-xs font-medium text-ink-2">
            <span>Card first used</span>
            <span className="tabular text-ink">{input.card_age_days === 0 ? "today" : `${input.card_age_days} days ago`}</span>
          </span>
          <input
            className="mt-2 w-full accent-[var(--accent)]"
            type="range"
            min={0}
            max={640}
            step={10}
            value={input.card_age_days}
            onChange={(e) => set("card_age_days")(Number(e.target.value))}
          />
        </label>
        <button
          type="submit"
          disabled={busy}
          className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-accent text-sm font-medium text-accent-ink hover:brightness-110 disabled:opacity-60"
        >
          {busy && <LoaderCircle aria-hidden className="size-4 animate-spin" />}
          Score this payment
        </button>
        <p className="text-xs text-muted">
          Scored live by the champion model as a new card&apos;s first payment. Nothing is stored.
        </p>
      </form>

      <div className="flex flex-col" aria-live="polite">
        <Gauge value={result ? gaugePosition(result.probability) : 0} />
        {error && <p className="mt-2 text-center text-sm text-critical-ink">{error}</p>}
        {result && d ? (
          <div className="mt-1 text-center">
            <p className="tabular text-3xl font-semibold text-ink">
              {(result.probability * 100).toLocaleString("en-US", { maximumFractionDigits: result.probability < 0.01 ? 2 : 1 })}%
              <span className="ml-2 text-sm font-normal text-muted">chance of fraud</span>
            </p>
            <span className={clsx("mt-2 inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium", d.cls)}>
              <d.icon aria-hidden className="size-4" /> {d.label}
            </span>
          </div>
        ) : (
          !error && <p className="mt-2 text-center text-sm text-muted">Pick a payment and score it.</p>
        )}
        <ul className="mt-4 space-y-2">
          <AnimatePresence mode="popLayout">
            {result?.reasons.map((r, i) => (
              <motion.li
                key={`${result.probability}-${r.feature}`}
                initial={{ opacity: 0, x: 24 }}
                animate={{ opacity: 1, x: 0, transition: { delay: 0.35 + i * 0.12 } }}
                exit={{ opacity: 0, x: -12, transition: { duration: 0.15 } }}
                className="flex items-center justify-between gap-3 rounded-lg bg-surface-2 px-3 py-2 text-sm"
              >
                <span className="text-ink">{r.text}</span>
                <span className="tabular shrink-0 text-xs text-muted">+{r.contribution.toFixed(2)}</span>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
        {result && (
          <p className="mt-3 text-center text-xs text-muted">
            Top reasons, in log-odds added to the score · model v{result.model_version} ·{" "}
            <span className="tabular">{result.latency_ms.toFixed(1)} ms</span>
          </p>
        )}
      </div>
    </div>
  );
}
