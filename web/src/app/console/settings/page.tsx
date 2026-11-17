import type { Metadata } from "next";

import { AuditPanel, KeysPanel, UsersPanel } from "./panels";
import { Card, EmptyState } from "@/components/ui";
import { ApiError, apiGet, currentUser } from "@/lib/api";
import type { ApiKey, AuditPage, User } from "@/lib/types";

export const metadata: Metadata = { title: "Settings" };

export default async function Settings() {
  const user = await currentUser();
  if (user?.role !== "admin") {
    return (
      <Card className="mx-auto max-w-xl">
        <EmptyState title="Admins only">Users, API keys and the audit log are managed by an admin.</EmptyState>
      </Card>
    );
  }
  let users: User[];
  let keys: ApiKey[];
  let audit: AuditPage;
  try {
    [users, keys, audit] = await Promise.all([
      apiGet<User[]>("/users"),
      apiGet<ApiKey[]>("/api-keys"),
      apiGet<AuditPage>("/audit?limit=25"),
    ]);
  } catch (e) {
    if (e instanceof ApiError) return <p className="text-sm text-critical-ink">{e.message}</p>;
    throw e;
  }
  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <header>
        <h1 className="text-xl font-semibold text-ink">Settings</h1>
        <p className="mt-1 text-sm text-muted">People who can sign in, keys for machine clients, and every change anyone made.</p>
      </header>
      <UsersPanel users={users} me={user.username} />
      <KeysPanel keys={keys} />
      <AuditPanel key={audit.entries[0]?.id ?? "empty"} first={audit} />
    </div>
  );
}
