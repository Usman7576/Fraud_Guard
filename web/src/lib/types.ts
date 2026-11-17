// Shapes returned by the FastAPI service (see /docs on the API).

export type Decision = "approve" | "review" | "decline";
export type CaseStatus = "open" | "in_review" | "closed";
export type Verdict = "fraud" | "genuine";

export interface Reason {
  feature: string;
  value: unknown;
  contribution: number;
  text: string;
}

export interface ScoreResponse {
  transaction_id: number;
  probability: number;
  decision: Decision;
  reasons: Reason[];
  model_version: string;
  latency_ms: number;
  case_opened: boolean;
}

export interface PreviewResponse extends ScoreResponse {
  transaction_dt: number;
  features: Record<string, unknown>;
  contributions: [string, number][];
}

export interface CaseSummary {
  txn_id: number;
  status: CaseStatus;
  probability: number;
  amount: number;
  decision: Decision;
  reasons: Reason[];
  assignee: string | null;
  verdict: Verdict | null;
  created_at: string;
  sla_due: string | null;
}

export interface Note {
  author: string;
  text: string;
  at: string;
}

export interface Neighbour {
  id: string;
  cards: number;
  fraud_cards: number;
}

export interface CaseDetail extends CaseSummary {
  card_uid: string;
  dt: number;
  features: Record<string, unknown>;
  contributions: [string, number][];
  notes: Note[];
  neighbourhood: Record<"device" | "address" | "email", Neighbour[]>;
}

export interface CasePage {
  items: CaseSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface KpiWeek {
  week: number;
  week_start: string;
  volume: number;
  decline_rate: number;
  review_rate: number;
  labelled_fraud: number;
  recall: number | null;
  cost: number;
  cost_per_10k: number;
  p95_latency_ms: number | null;
}

export interface Policy {
  false_decline_cost: number;
  review_cost: number;
  review_capacity_per_day: number;
  risk_multiplier: number;
}

export interface Health {
  status: string;
  model_version: string;
  transactions_in_memory: number;
  latest_transaction_dt: number | null;
}

export interface DriftWeek {
  week: number;
  week_start: string | null;
  n_rows: number;
  n_features: number;
  n_drifted: number;
  drift_share: number;
  threshold: number | null;
  top_drifted: string[];
  report_file: string | null;
}

export interface Interval {
  estimate: number;
  low: number;
  high: number;
  n_valid: number;
}

export interface RetrainingEvent {
  week: number;
  week_start: string | null;
  ts: string;
  trigger: string;
  outcome: "promoted" | "kept_champion" | "skipped";
  detail: string;
  drift_share: number | null;
  recall: number | null;
  champion_version: string | null;
  challenger_version: string | null;
  comparison: {
    n_rows: number;
    n_fraud: number;
    pr_auc_gain: Interval;
    cost_saved_per_10k: Interval;
  } | null;
}

export interface ModelVersion {
  version: string;
  aliases: string[];
  run_id: string | null;
  created_at: string | null;
  metrics: Record<string, number>;
  serving: boolean;
}
