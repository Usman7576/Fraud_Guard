"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button, Input, Label } from "@/components/ui";

export function LoginForm({ next }: { next: string }) {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setPending(true);
    setError(null);
    const form = new FormData(e.currentTarget);
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: form.get("username"), password: form.get("password") }),
    });
    if (res.ok) {
      router.replace(next);
      router.refresh();
      return;
    }
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    setError(body.detail ?? "Sign-in failed");
    setPending(false);
  }

  return (
    <form onSubmit={onSubmit} className="mt-5 space-y-4">
      <div>
        <Label htmlFor="username">Username</Label>
        <Input id="username" name="username" autoComplete="username" required autoFocus />
      </div>
      <div>
        <Label htmlFor="password">Password</Label>
        <Input id="password" name="password" type="password" autoComplete="current-password" required />
      </div>
      {error ? (
        <p role="alert" className="rounded-lg bg-critical/10 px-3 py-2 text-sm text-critical-ink">
          {error}
        </p>
      ) : null}
      <Button type="submit" className="w-full" disabled={pending}>
        {pending ? "Signing in…" : "Sign in"}
      </Button>
    </form>
  );
}
