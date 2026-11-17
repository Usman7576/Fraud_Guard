"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { Button } from "@/components/ui";

export function PromoteButton({ version }: { version: string }) {
  const router = useRouter();
  const [pending, start] = useTransition();
  const [error, setError] = useState<string | null>(null);

  function promote() {
    if (!window.confirm(`Serve model v${version} for every new transaction? This is recorded in the audit log.`)) return;
    setError(null);
    start(async () => {
      const res = await fetch(`/api/fg/models/${version}/promote`, { method: "POST" });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as { detail?: string };
        setError(body.detail ?? `Promotion failed (${res.status})`);
        return;
      }
      router.refresh();
    });
  }

  return (
    <div className="inline-flex flex-col items-end gap-1">
      <Button variant="secondary" className="h-8 px-3 text-xs" disabled={pending} onClick={promote}>
        {pending ? "Promoting…" : "Promote"}
      </Button>
      {error ? <span className="text-xs text-critical-ink">{error}</span> : null}
    </div>
  );
}
