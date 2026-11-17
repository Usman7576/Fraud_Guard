import type { Metadata } from "next";
import { ShieldCheck } from "lucide-react";

import { LoginForm } from "./login-form";
import { safeNext } from "@/lib/session";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({ searchParams }: { searchParams: Promise<{ next?: string }> }) {
  const { next } = await searchParams;
  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-2 text-accent">
          <ShieldCheck aria-hidden className="size-5" />
          <span className="text-sm font-semibold">FraudGuard console</span>
        </div>
        <div className="rounded-xl border border-line bg-surface p-6 shadow-[var(--shadow)]">
          <h1 className="text-lg font-semibold text-ink">Sign in</h1>
          <p className="mt-1 text-sm text-muted">
            Accounts are created with <code className="font-mono text-xs">fraudguard db add-user</code>.
          </p>
          <LoginForm next={safeNext(next)} />
        </div>
      </div>
    </main>
  );
}
