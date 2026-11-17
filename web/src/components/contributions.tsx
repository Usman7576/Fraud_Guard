// Diverging bars for SHAP contributions (log-odds). Right of the zero line, in red, raises the
// fraud risk; left, in blue, lowers it. Direction is also spelled out in words, so colour is
// never the only cue. Each row shows a tooltip on hover or keyboard focus.
import clsx from "clsx";

import { featureValue } from "@/lib/format";

export function ContributionBars({
  contributions,
  values,
  limit = 12,
}: {
  contributions: [string, number][];
  values: Record<string, unknown>;
  limit?: number;
}) {
  const rows = [...contributions].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, limit);
  const max = Math.max(...rows.map(([, c]) => Math.abs(c)), 1e-6);
  if (!rows.length) return <p className="px-5 pb-5 text-sm text-muted">No contributions recorded.</p>;

  return (
    <div className="px-5 pb-5">
      <div className="mb-2 grid grid-cols-[minmax(0,6.5rem)_1fr] gap-3 text-[11px] text-muted sm:grid-cols-[minmax(0,11rem)_1fr]">
        <span />
        <div className="flex justify-between">
          <span>← lowers risk</span>
          <span>raises risk →</span>
        </div>
      </div>
      <ul className="space-y-1.5">
        {rows.map(([feature, c]) => {
          const widthPct = (Math.abs(c) / max) * 38;
          const up = c > 0;
          return (
            <li
              key={feature}
              tabIndex={0}
              className="group relative grid grid-cols-[minmax(0,6.5rem)_1fr] items-center gap-3 rounded outline-none sm:grid-cols-[minmax(0,11rem)_1fr]"
              aria-label={`${feature} = ${featureValue(values[feature])}: ${up ? "raises" : "lowers"} risk by ${Math.abs(c).toFixed(3)} log-odds`}
            >
              <span className="truncate font-mono text-xs text-ink-2" title={feature}>
                {feature}
              </span>
              <div className="relative h-5">
                <span aria-hidden className="absolute inset-y-0 left-1/2 w-px bg-[var(--axis)]" />
                <span
                  aria-hidden
                  className={clsx(
                    "absolute top-1/2 h-3 -translate-y-1/2",
                    up ? "left-1/2 rounded-r-[4px] bg-[var(--risk-up)]" : "right-1/2 rounded-l-[4px] bg-[var(--risk-down)]",
                  )}
                  style={{ width: `${Math.max(widthPct, 0.8)}%` }}
                />
                <span
                  className={clsx(
                    "tabular absolute top-1/2 -translate-y-1/2 text-[11px] text-ink-2",
                    up ? "pl-1.5" : "pr-1.5 text-right",
                  )}
                  style={up ? { left: `${50 + widthPct}%` } : { right: `${50 + widthPct}%` }}
                >
                  {c > 0 ? "+" : ""}
                  {c.toFixed(2)}
                </span>
              </div>
              <span
                role="tooltip"
                className="pointer-events-none absolute left-0 top-6 z-10 hidden rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink shadow-[var(--shadow)] group-hover:block group-focus:block"
              >
                <span className="font-mono">{feature}</span> = {featureValue(values[feature])} · {up ? "raises" : "lowers"} risk
                by {Math.abs(c).toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
