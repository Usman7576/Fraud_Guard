// Pure helpers for the public landing page (kept free of React and server-only imports so
// they are unit-testable).

export const GITHUB_URL = "https://github.com/Usman7576/Fraud_Guard";

// Gauge ------------------------------------------------------------------------------------------

/** Fraud probabilities span four orders of magnitude, so the gauge is logarithmic. */
export const GAUGE_MIN = 1e-4;
export const GAUGE_TICKS = [1e-4, 1e-3, 1e-2, 1e-1, 1] as const;

/** Position on the gauge, 0 (left, 0.01% or less) to 1 (right, 100%). */
export function gaugePosition(p: number): number {
  if (!Number.isFinite(p)) return 0;
  const clamped = Math.min(Math.max(p, GAUGE_MIN), 1);
  return (Math.log10(clamped) - Math.log10(GAUGE_MIN)) / -Math.log10(GAUGE_MIN);
}

export function tickLabel(p: number): string {
  return `${(p * 100).toLocaleString("en-US", { maximumSignificantDigits: 2 })}%`;
}

// Live demo --------------------------------------------------------------------------------------

export const PRODUCTS = [
  { value: "W", label: "Physical goods (W)" },
  { value: "C", label: "Digital goods (C)" },
  { value: "H", label: "Services (H)" },
  { value: "R", label: "Retail (R)" },
  { value: "S", label: "Subscriptions (S)" },
] as const;
export const NETWORKS = ["visa", "mastercard", "american express", "discover"] as const;
export const CARD_TYPES = ["debit", "credit"] as const;
export const EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com", "protonmail.com"] as const;
export const DEVICES = ["desktop", "mobile"] as const;

export interface DemoInput {
  TransactionAmt: number;
  ProductCD: (typeof PRODUCTS)[number]["value"];
  card4: (typeof NETWORKS)[number];
  card6: (typeof CARD_TYPES)[number];
  P_emaildomain: (typeof EMAILS)[number];
  DeviceType: (typeof DEVICES)[number];
  card_age_days: number;
}

export const PRESETS: { label: string; input: DemoInput }[] = [
  {
    label: "Everyday purchase",
    input: {
      TransactionAmt: 42.5,
      ProductCD: "W",
      card4: "visa",
      card6: "debit",
      P_emaildomain: "gmail.com",
      DeviceType: "desktop",
      card_age_days: 420,
    },
  },
  {
    label: "Risky first purchase",
    input: {
      TransactionAmt: 1999,
      ProductCD: "C",
      card4: "mastercard",
      card6: "credit",
      P_emaildomain: "protonmail.com",
      DeviceType: "mobile",
      card_age_days: 0,
    },
  },
];

export interface DemoReason {
  feature: string;
  text: string;
  contribution: number;
}

export interface DemoResult {
  probability: number;
  decision: "approve" | "review" | "decline";
  reasons: DemoReason[];
  model_version: string;
  latency_ms: number;
}

/** Sliding one-minute window per client, in memory (one web server process). */
export class RateWindow {
  private hits = new Map<string, number[]>();

  constructor(
    private readonly perMinute: number,
    private readonly maxClients = 10_000,
  ) {}

  allow(client: string, now = Date.now()): boolean {
    const recent = (this.hits.get(client) ?? []).filter((t) => t > now - 60_000);
    if (recent.length >= this.perMinute) {
      this.hits.set(client, recent);
      return false;
    }
    recent.push(now);
    this.hits.delete(client); // re-insert so the map stays ordered by last use
    this.hits.set(client, recent);
    if (this.hits.size > this.maxClients) {
      const oldest = this.hits.keys().next().value;
      if (oldest !== undefined) this.hits.delete(oldest);
    }
    return true;
  }
}

export function fmt(v: number, format: "percent" | "money" | "decimal"): string {
  if (format === "percent") return `${(v * 100).toFixed(1)}%`;
  if (format === "money") return `${v < 0 ? "-" : ""}$${Math.abs(Math.round(v)).toLocaleString("en-US")}`;
  return v.toFixed(3);
}

// Results ----------------------------------------------------------------------------------------

export interface Interval {
  estimate: number;
  low: number;
  high: number;
}

export interface LandingResults {
  modelVersion: string;
  nRows: number;
  nFraud: number;
  nResamples: number;
  recall: Interval;
  rulesRecall: number;
  costSaved: Interval;
  prAuc: Interval;
  rulesPrAuc: number;
  policy: { falseDeclineCost: number; reviewCost: number; reviewCapacity: number };
  /** Rank (1 = most important) of the two-hop fraud-card count by SHAP, if it is in the top list. */
  graphRank: number | null;
  /** Fewer rows than a real IEEE-CIS test month: the synthetic development data. */
  isSample: boolean;
}

const REAL_TEST_MONTH_MIN_ROWS = 50_000;

function interval(v: unknown): Interval | null {
  if (typeof v !== "object" || v === null) return null;
  const { estimate, low, high } = v as Record<string, unknown>;
  return [estimate, low, high].every((x) => typeof x === "number" && Number.isFinite(x))
    ? { estimate: estimate as number, low: low as number, high: high as number }
    : null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function graphRank(shap: unknown): number | null {
  if (!Array.isArray(shap)) return null;
  const i = shap.findIndex((x) => Array.isArray(x) && x[0] === "graph_2hop_fraud_cards");
  return i >= 0 ? i + 1 : null;
}

/** The landing page's figures from reports/evaluation.json, or null if it is missing or incomplete. */
export function parseResults(raw: unknown): LandingResults | null {
  if (typeof raw !== "object" || raw === null) return null;
  const d = raw as Record<string, Record<string, unknown> | undefined> & Record<string, unknown>;
  const iv = (d.intervals ?? {}) as Record<string, unknown>;
  const recall = interval(iv.model_recall_at_fpr);
  const rulesRecall = interval(iv.rules_recall_at_fpr);
  const costSaved = interval(iv.cost_saved_per_10k);
  const prAuc = interval(iv.model_pr_auc);
  const rulesPrAuc = interval(iv.rules_pr_auc);
  const policy = (d.policy ?? {}) as Record<string, unknown>;
  const nRows = num(d.n_rows);
  const nFraud = num(d.n_fraud);
  const falseDeclineCost = num(policy.false_decline_cost);
  const reviewCost = num(policy.review_cost);
  const reviewCapacity = num(policy.review_capacity_per_day);
  if (!recall || !rulesRecall || !costSaved || !prAuc || !rulesPrAuc || nRows === null || nFraud === null) return null;
  if (falseDeclineCost === null || reviewCost === null || reviewCapacity === null) return null;
  return {
    modelVersion: String(d.model_version ?? "?"),
    nRows,
    nFraud,
    nResamples: num(d.n_resamples) ?? 0,
    recall,
    rulesRecall: rulesRecall.estimate,
    costSaved,
    prAuc,
    rulesPrAuc: rulesPrAuc.estimate,
    policy: { falseDeclineCost, reviewCost, reviewCapacity },
    graphRank: graphRank(d.shap_global),
    isSample: nRows < REAL_TEST_MONTH_MIN_ROWS,
  };
}
