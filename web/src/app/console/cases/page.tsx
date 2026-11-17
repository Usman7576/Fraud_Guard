import type { Metadata } from "next";
import Link from "next/link";

import { QueueTable } from "./queue-table";
import { Card, EmptyState, Select } from "@/components/ui";
import { ApiError, apiGet } from "@/lib/api";
import { count } from "@/lib/format";
import type { CasePage } from "@/lib/types";

export const metadata: Metadata = { title: "Review queue" };

const STATUSES = [
  ["open", "Open"],
  ["in_review", "In review"],
  ["closed", "Closed"],
  ["all", "All"],
] as const;
const SORTS = [
  ["probability", "Highest risk"],
  ["amount", "Largest amount"],
  ["sla_due", "Due soonest"],
  ["created_at", "Newest"],
] as const;
const PAGE_SIZE = 25;

type Params = { status?: string; sort?: string; min_probability?: string; mine?: string; page?: string };

function pick<T extends readonly (readonly [string, string])[]>(options: T, v: string | undefined, fallback: T[number][0]) {
  return options.some(([k]) => k === v) ? (v as T[number][0]) : fallback;
}

export default async function Cases({ searchParams }: { searchParams: Promise<Params> }) {
  const sp = await searchParams;
  const status = pick(STATUSES, sp.status, "open");
  const sort = pick(SORTS, sp.sort, "probability");
  const minP = Math.min(Math.max(Number(sp.min_probability ?? 0) || 0, 0), 1);
  const page = Math.max(Number.parseInt(sp.page ?? "1", 10) || 1, 1);
  const qs = new URLSearchParams({
    status,
    sort,
    order: sort === "sla_due" ? "asc" : "desc",
    min_probability: String(minP),
    page: String(page),
    page_size: String(PAGE_SIZE),
  });
  if (sp.mine) qs.set("assignee", sp.mine);

  let data: CasePage;
  try {
    data = await apiGet<CasePage>(`/cases?${qs}`);
  } catch (e) {
    if (e instanceof ApiError) return <p className="text-sm text-critical-ink">{e.message}</p>;
    throw e;
  }
  const pages = Math.max(Math.ceil(data.total / PAGE_SIZE), 1);
  const link = (p: number) => `/console/cases?${new URLSearchParams({ ...sp, page: String(p) } as Record<string, string>)}`;

  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <header>
        <h1 className="text-xl font-semibold text-ink">Review queue</h1>
        <p className="mt-1 text-sm text-muted">
          {count(data.total)} {data.total === 1 ? "case" : "cases"}. Keyboard: <kbd className="font-mono">j</kbd>/
          <kbd className="font-mono">k</kbd> to move, <kbd className="font-mono">Enter</kbd> to open.
        </p>
      </header>

      <form className="flex flex-wrap items-end gap-3" action="/console/cases">
        <div>
          <label htmlFor="status" className="mb-1 block text-xs text-ink-2">Status</label>
          <Select id="status" name="status" defaultValue={status}>
            {STATUSES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </Select>
        </div>
        <div>
          <label htmlFor="sort" className="mb-1 block text-xs text-ink-2">Sort</label>
          <Select id="sort" name="sort" defaultValue={sort}>
            {SORTS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </Select>
        </div>
        <div>
          <label htmlFor="min_probability" className="mb-1 block text-xs text-ink-2">Minimum risk</label>
          <Select id="min_probability" name="min_probability" defaultValue={String(minP)}>
            {[0, 0.05, 0.1, 0.25, 0.5].map((v) => <option key={v} value={v}>{v === 0 ? "Any" : `${v * 100}%+`}</option>)}
          </Select>
        </div>
        <button type="submit" className="h-9 rounded-lg border border-line bg-surface px-3 text-sm text-ink hover:bg-surface-2">
          Apply
        </button>
      </form>

      <Card className="overflow-hidden">
        {data.items.length ? (
          <QueueTable items={data.items} />
        ) : (
          <EmptyState title="Nothing here">No cases match these filters. Review decisions open a case automatically.</EmptyState>
        )}
      </Card>

      {pages > 1 && (
        <nav className="flex items-center justify-between text-sm" aria-label="Pages">
          <span className="text-muted">Page {page} of {pages}</span>
          <div className="flex gap-2">
            {page > 1 && <Link className="rounded-lg border border-line px-3 py-1.5 hover:bg-surface-2" href={link(page - 1)}>Previous</Link>}
            {page < pages && <Link className="rounded-lg border border-line px-3 py-1.5 hover:bg-surface-2" href={link(page + 1)}>Next</Link>}
          </div>
        </nav>
      )}
    </div>
  );
}
