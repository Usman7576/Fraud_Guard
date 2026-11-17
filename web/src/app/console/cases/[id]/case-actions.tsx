"use client";

import { ShieldAlert, ShieldCheck, UserCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/ui";
import type { CaseStatus } from "@/lib/types";

export function CaseActions({
  txnId,
  status,
  assignee,
  me,
}: {
  txnId: number;
  status: CaseStatus;
  assignee: string | null;
  me: string;
}) {
  const router = useRouter();
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, start] = useTransition();
  const closed = status === "closed";

  function patch(body: Record<string, unknown>) {
    setError(null);
    start(async () => {
      const res = await fetch(`/api/fg/cases/${txnId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const b = (await res.json().catch(() => ({}))) as { detail?: string };
        setError(b.detail ?? `Update failed (${res.status})`);
        return;
      }
      if ("note" in body) setNote("");
      router.refresh();
    });
  }

  return (
    <div className="space-y-4 px-5 pb-5">
      {!closed && (
        <div className="grid grid-cols-2 gap-2">
          <Button variant="danger" disabled={pending} onClick={() => patch({ verdict: "fraud", ...(note ? { note } : {}) })}>
            <ShieldAlert aria-hidden className="size-4" /> Fraud
          </Button>
          <Button variant="good" disabled={pending} onClick={() => patch({ verdict: "genuine", ...(note ? { note } : {}) })}>
            <ShieldCheck aria-hidden className="size-4" /> Genuine
          </Button>
        </div>
      )}
      {!closed && assignee !== me && me && (
        <Button variant="secondary" className="w-full" disabled={pending} onClick={() => patch({ assignee: me })}>
          <UserCheck aria-hidden className="size-4" /> Assign to me
        </Button>
      )}
      <div>
        <label htmlFor="note" className="mb-1 block text-xs font-medium text-ink-2">
          Note
        </label>
        <textarea
          id="note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={3}
          maxLength={5000}
          placeholder="What did you check?"
          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-muted focus:border-accent focus:outline-none"
        />
        <Button variant="secondary" className="mt-2" disabled={pending || !note.trim()} onClick={() => patch({ note })}>
          Add note
        </Button>
      </div>
      {error ? (
        <p role="alert" className="rounded-lg bg-critical/10 px-3 py-2 text-sm text-critical-ink">
          {error}
        </p>
      ) : null}
    </div>
  );
}
