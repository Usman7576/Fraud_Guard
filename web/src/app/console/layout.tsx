import { LogOut, ShieldCheck } from "lucide-react";
import Link from "next/link";

import { ConsoleNav } from "./nav";
import { currentUser } from "@/lib/api";

export default async function ConsoleLayout({ children }: { children: React.ReactNode }) {
  const user = await currentUser();
  return (
    <div className="min-h-screen md:grid md:grid-cols-[13.5rem_1fr]">
      <aside className="border-b border-line bg-surface md:sticky md:top-0 md:h-screen md:border-r md:border-b-0">
        <div className="flex h-full flex-col gap-1 p-3 md:p-4">
          <Link href="/console" className="mb-2 flex items-center gap-2 px-2 py-1.5 text-accent md:mb-5">
            <ShieldCheck aria-hidden className="size-5" />
            <span className="text-sm font-semibold text-ink">FraudGuard</span>
          </Link>
          <div className="flex items-center gap-2">
            <div className="min-w-0 flex-1 md:flex-none">
              <ConsoleNav isAdmin={user?.role === "admin"} />
            </div>
            <form action="/api/auth/logout" method="post" className="md:hidden">
              <button type="submit" aria-label="Sign out" className="rounded-lg p-2 text-ink-2 hover:bg-surface-2">
                <LogOut aria-hidden className="size-4" />
              </button>
            </form>
          </div>
          <div className="mt-auto hidden border-t border-line pt-3 md:block">
            {user ? (
              <div className="px-2 pb-2 text-xs">
                <div className="font-medium text-ink">{user.username}</div>
                <div className="text-muted capitalize">{user.role}</div>
              </div>
            ) : null}
            <form action="/api/auth/logout" method="post">
              <button
                type="submit"
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                <LogOut aria-hidden className="size-4" /> Sign out
              </button>
            </form>
          </div>
        </div>
      </aside>
      <main className="min-w-0 px-4 py-6 md:px-8 md:py-8">{children}</main>
    </div>
  );
}
