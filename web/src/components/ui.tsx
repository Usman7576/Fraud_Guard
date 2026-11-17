// Small UI primitives in the shadcn/ui spirit: plain Tailwind, no runtime dependencies.
import clsx from "clsx";
import { CheckCircle2, Eye, ShieldAlert } from "lucide-react";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

import type { CaseStatus, Decision } from "@/lib/types";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <section className={clsx("rounded-xl border border-line bg-surface shadow-[var(--shadow)]", className)}>
      {children}
    </section>
  );
}

export function CardHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <header className="flex items-start justify-between gap-4 px-5 pt-4 pb-3">
      <div>
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        {subtitle ? <p className="mt-0.5 text-xs text-muted">{subtitle}</p> : null}
      </div>
      {action}
    </header>
  );
}

type Variant = "primary" | "secondary" | "ghost" | "danger" | "good";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-110",
  secondary: "border border-line bg-surface text-ink hover:bg-surface-2",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink",
  danger: "bg-critical text-white hover:brightness-110",
  good: "bg-good text-white hover:brightness-110",
};

export function Button({
  variant = "primary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={clsx(
        "inline-flex h-9 items-center justify-center gap-2 rounded-lg px-3.5 text-sm font-medium transition",
        "disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        "h-9 w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink placeholder:text-muted",
        "focus:border-accent focus:outline-none",
        className,
      )}
      {...props}
    />
  );
}

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={clsx(
        "h-9 rounded-lg border border-line bg-surface px-2.5 text-sm text-ink focus:border-accent focus:outline-none",
        className,
      )}
      {...props}
    />
  );
}

export function Label({ htmlFor, children }: { htmlFor: string; children: ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="mb-1 block text-xs font-medium text-ink-2">
      {children}
    </label>
  );
}

/** Decision chip: colour, icon and word together, never colour alone. */
export function DecisionBadge({ decision }: { decision: Decision }) {
  const map = {
    approve: { icon: CheckCircle2, cls: "text-good-ink bg-good/10", label: "Approve" },
    review: { icon: Eye, cls: "text-warning-ink bg-warning/15", label: "Review" },
    decline: { icon: ShieldAlert, cls: "text-critical-ink bg-critical/10", label: "Decline" },
  }[decision];
  const Icon = map.icon;
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", map.cls)}>
      <Icon aria-hidden className="size-3.5" />
      {map.label}
    </span>
  );
}

export function StatusBadge({ status }: { status: CaseStatus }) {
  const label = { open: "Open", in_review: "In review", closed: "Closed" }[status];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium",
        status === "open" && "border-accent/40 text-accent",
        status === "in_review" && "border-warning/50 text-warning-ink",
        status === "closed" && "border-line text-muted",
      )}
    >
      <span
        aria-hidden
        className={clsx(
          "size-1.5 rounded-full",
          status === "open" && "bg-accent",
          status === "in_review" && "bg-warning",
          status === "closed" && "bg-muted",
        )}
      />
      {label}
    </span>
  );
}

/** Calibrated fraud probability as a thin meter: the fill carries severity, the track is the same ramp. */
export function ProbabilityBar({ p, className }: { p: number; className?: string }) {
  const tone = p >= 0.5 ? "bg-critical" : p >= 0.1 ? "bg-warning" : "bg-accent";
  const track = p >= 0.5 ? "bg-critical/15" : p >= 0.1 ? "bg-warning/20" : "bg-accent/15";
  return (
    <div className={clsx("flex items-center gap-2", className)}>
      <div
        className={clsx("h-1.5 w-20 overflow-hidden rounded-full", track)}
        role="meter"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(p * 100)}
        aria-label="Fraud probability"
      >
        <div className={clsx("h-full rounded-full", tone)} style={{ width: `${Math.max(p * 100, 2)}%` }} />
      </div>
      <span className="tabular w-12 text-right text-sm text-ink">{(p * 100).toFixed(p < 0.1 ? 1 : 0)}%</span>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-1 px-6 py-14 text-center">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children ? <div className="max-w-md text-sm text-muted">{children}</div> : null}
    </div>
  );
}
