// Number and text formatting shared by the console.

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** 1,284 / 12.9K / 4.2M: in full below 10,000, compact above. */
export function count(n: number): string {
  return Math.abs(n) < 10_000 ? whole.format(n) : compact.format(n);
}

export function money(n: number, digits = 2): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
}

export function moneyCompact(n: number): string {
  return Math.abs(n) < 10_000 ? money(n, 0) : `$${compact.format(n)}`;
}

/** 0.0342 -> "3.4%"; tiny non-zero shares keep a significant digit. */
export function percent(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const p = x * 100;
  if (p !== 0 && Math.abs(p) < 0.1) return `${p.toFixed(2)}%`;
  return `${p.toFixed(digits)}%`;
}

export function ms(x: number | null | undefined): string {
  if (x === null || x === undefined) return "—";
  return x < 10 ? `${x.toFixed(1)} ms` : `${Math.round(x)} ms`;
}

/** Time to a deadline: "in 3h", "2h overdue". */
export function due(iso: string | null, now: Date = new Date()): string {
  if (!iso) return "—";
  const diff = new Date(iso).getTime() - now.getTime();
  const hours = Math.round(Math.abs(diff) / 3_600_000);
  const span = hours >= 48 ? `${Math.round(hours / 24)}d` : hours >= 1 ? `${hours}h` : "<1h";
  return diff >= 0 ? `in ${span}` : `${span} overdue`;
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

export function dateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** Feature values for display: missing as an em dash, long floats trimmed, booleans as yes/no. */
export function featureValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return whole.format(v);
    return Math.abs(v) >= 100 ? v.toFixed(1) : Number(v.toPrecision(4)).toString();
  }
  return String(v);
}
