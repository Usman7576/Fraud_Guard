"use client";

import clsx from "clsx";
import { Check, Copy, KeyRound } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

import { Button, Card, CardHeader, Input, Label, Select } from "@/components/ui";
import { dateTime } from "@/lib/format";
import type { ApiKey, AuditEntry, AuditPage, Role, Scope, User } from "@/lib/types";

async function send(path: string, method: string, body?: unknown): Promise<{ ok: true; data: unknown } | { ok: false; error: string }> {
  const res = await fetch(`/api/fg/${path}`, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = (await res.json().catch(() => ({}))) as { detail?: unknown };
  if (res.ok) return { ok: true, data };
  const detail = data.detail;
  const message =
    typeof detail === "string"
      ? detail
      : Array.isArray(detail)
        ? detail.map((d: { msg?: string; loc?: unknown[] }) => `${d.loc?.at(-1) ?? "field"}: ${d.msg ?? "invalid"}`).join("; ")
        : `Request failed (${res.status})`;
  return { ok: false, error: message };
}

function useAction() {
  const router = useRouter();
  const [pending, start] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const run = (fn: () => ReturnType<typeof send>, after?: (data: unknown) => void) =>
    start(async () => {
      setError(null);
      const r = await fn();
      if (!r.ok) return setError(r.error);
      after?.(r.data);
      router.refresh();
    });
  return { pending, error, run };
}

function Chip({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "muted" }) {
  return (
    <span
      className={clsx(
        "inline-flex rounded-full px-2 py-0.5 text-xs font-medium",
        tone === "good" && "bg-good/10 text-good-ink",
        tone === "neutral" && "bg-surface-2 text-ink-2",
        tone === "muted" && "bg-surface-2 text-muted",
      )}
    >
      {children}
    </span>
  );
}

// Users ------------------------------------------------------------------------------------------

function PasswordReset({ username, onDone }: { username: string; onDone: () => void }) {
  const { pending, error, run } = useAction();
  const [password, setPassword] = useState("");
  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        run(() => send(`users/${username}`, "PATCH", { password }), onDone);
      }}
    >
      <Input
        type="password"
        autoComplete="new-password"
        minLength={10}
        required
        placeholder="New password (10+ characters)"
        aria-label={`New password for ${username}`}
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="h-8 w-56"
      />
      <Button type="submit" className="h-8 px-3 text-xs" disabled={pending}>
        Save
      </Button>
      <Button type="button" variant="ghost" className="h-8 px-3 text-xs" onClick={onDone}>
        Cancel
      </Button>
      {error && <span className="text-xs text-critical-ink">{error}</span>}
    </form>
  );
}

export function UsersPanel({ users, me }: { users: User[]; me: string }) {
  const { pending, error, run } = useAction();
  const [form, setForm] = useState({ username: "", role: "analyst" as Role, password: "" });
  const [resetting, setResetting] = useState<string | null>(null);

  return (
    <Card className="overflow-x-auto">
      <CardHeader title="Users" subtitle="Analysts work the queue; admins also manage models, the cost policy and these settings" />
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted">
          <tr className="border-y border-line">
            <th className="px-5 py-2 font-medium">User</th>
            <th className="px-5 py-2 font-medium">Role</th>
            <th className="px-5 py-2 font-medium">Status</th>
            <th className="px-5 py-2 font-medium">Created</th>
            <th className="px-5 py-2 font-medium" />
          </tr>
        </thead>
        <tbody>
          {users.map((u) => (
            <tr key={u.username} className="border-b border-line last:border-0">
              <td className="px-5 py-2 font-medium text-ink">
                {u.username}
                {u.username === me && <span className="ml-2 text-xs font-normal text-muted">(you)</span>}
              </td>
              <td className="px-5 py-2">
                <Select
                  aria-label={`Role of ${u.username}`}
                  className="h-8 text-xs"
                  value={u.role}
                  disabled={pending || u.username === me}
                  onChange={(e) => run(() => send(`users/${u.username}`, "PATCH", { role: e.target.value }))}
                >
                  <option value="analyst">analyst</option>
                  <option value="admin">admin</option>
                </Select>
              </td>
              <td className="px-5 py-2">{u.active ? <Chip tone="good">active</Chip> : <Chip tone="muted">deactivated</Chip>}</td>
              <td className="tabular px-5 py-2 text-ink-2">{u.created_at ? dateTime(u.created_at) : "—"}</td>
              <td className="px-5 py-2 text-right">
                {resetting === u.username ? (
                  <PasswordReset username={u.username} onDone={() => setResetting(null)} />
                ) : (
                  <div className="flex justify-end gap-1">
                    <Button variant="ghost" className="h-8 px-2.5 text-xs" onClick={() => setResetting(u.username)}>
                      Set password
                    </Button>
                    {u.username !== me && (
                      <Button
                        variant="ghost"
                        className="h-8 px-2.5 text-xs"
                        disabled={pending}
                        onClick={() => run(() => send(`users/${u.username}`, "PATCH", { active: !u.active }))}
                      >
                        {u.active ? "Deactivate" : "Activate"}
                      </Button>
                    )}
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <form
        className="flex flex-wrap items-end gap-3 border-t border-line px-5 py-4"
        onSubmit={(e) => {
          e.preventDefault();
          run(() => send("users", "POST", form), () => setForm({ username: "", role: "analyst", password: "" }));
        }}
      >
        <div>
          <Label htmlFor="new-username">New user</Label>
          <Input
            id="new-username"
            required
            pattern="[a-z0-9][a-z0-9_.\-]{2,31}"
            title="3-32 characters: lowercase letters, digits, _ . -"
            autoComplete="off"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value.toLowerCase() })}
            className="w-40"
          />
        </div>
        <div>
          <Label htmlFor="new-role">Role</Label>
          <Select id="new-role" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
            <option value="analyst">analyst</option>
            <option value="admin">admin</option>
          </Select>
        </div>
        <div>
          <Label htmlFor="new-password">Password</Label>
          <Input
            id="new-password"
            type="password"
            required
            minLength={10}
            autoComplete="new-password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="w-52"
          />
        </div>
        <Button type="submit" disabled={pending}>
          Add user
        </Button>
        {error && <p className="w-full text-sm text-critical-ink">{error}</p>}
      </form>
    </Card>
  );
}

// API keys ---------------------------------------------------------------------------------------

const SCOPES: { scope: Scope; help: string }[] = [
  { scope: "score", help: "score live payments" },
  { scope: "labels", help: "send chargebacks" },
  { scope: "promote", help: "promote models" },
  { scope: "demo", help: "landing-page demo" },
];

function NewKey({ name, keyValue, onClose }: { name: string; keyValue: string; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mx-5 mb-4 rounded-lg border border-warning/50 bg-warning/10 p-4" role="status">
      <p className="text-sm font-medium text-ink">Key for {name}: copy it now, it is not shown again.</p>
      <div className="mt-2 flex items-center gap-2">
        <code className="min-w-0 flex-1 truncate rounded-md bg-surface px-2.5 py-1.5 font-mono text-xs text-ink" data-testid="new-api-key">
          {keyValue}
        </code>
        <Button
          variant="secondary"
          className="h-8 px-2.5 text-xs"
          onClick={async () => {
            await navigator.clipboard.writeText(keyValue);
            setCopied(true);
          }}
        >
          {copied ? <Check aria-hidden className="size-3.5" /> : <Copy aria-hidden className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button variant="ghost" className="h-8 px-2.5 text-xs" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  );
}

export function KeysPanel({ keys }: { keys: ApiKey[] }) {
  const { pending, error, run } = useAction();
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<Scope[]>(["score"]);
  const [created, setCreated] = useState<{ name: string; key: string } | null>(null);

  return (
    <Card className="overflow-x-auto">
      <CardHeader title="API keys" subtitle="For machine clients. Only a hash is stored, so a key is shown once, when it is created" />
      {created && <NewKey name={created.name} keyValue={created.key} onClose={() => setCreated(null)} />}
      {keys.length ? (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr className="border-y border-line">
              <th className="px-5 py-2 font-medium">Name</th>
              <th className="px-5 py-2 font-medium">Scopes</th>
              <th className="px-5 py-2 font-medium">Status</th>
              <th className="px-5 py-2 font-medium">Created</th>
              <th className="px-5 py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k.id} className={clsx("border-b border-line last:border-0", !k.active && "text-muted")}>
                <td className="px-5 py-2 font-medium">
                  <span className="inline-flex items-center gap-2">
                    <KeyRound aria-hidden className="size-3.5 text-muted" />
                    {k.name}
                  </span>
                </td>
                <td className="px-5 py-2">
                  <span className="flex flex-wrap gap-1">
                    {k.scopes.map((s) => (
                      <Chip key={s}>{s}</Chip>
                    ))}
                  </span>
                </td>
                <td className="px-5 py-2">{k.active ? <Chip tone="good">active</Chip> : <Chip tone="muted">revoked</Chip>}</td>
                <td className="tabular px-5 py-2 text-ink-2">{k.created_at ? dateTime(k.created_at) : "—"}</td>
                <td className="px-5 py-2 text-right">
                  {k.active && (
                    <Button
                      variant="ghost"
                      className="h-8 px-2.5 text-xs text-critical-ink"
                      disabled={pending}
                      onClick={() => {
                        if (window.confirm(`Revoke ${k.name}? Clients using it stop working at once.`))
                          run(() => send(`api-keys/${k.id}/revoke`, "POST"));
                      }}
                    >
                      Revoke
                    </Button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="px-5 pb-3 text-sm text-muted">No API keys yet.</p>
      )}
      <form
        className="flex flex-wrap items-end gap-3 border-t border-line px-5 py-4"
        onSubmit={(e) => {
          e.preventDefault();
          run(
            () => send("api-keys", "POST", { name, scopes }),
            (data) => {
              setCreated({ name, key: (data as { key: string }).key });
              setName("");
            },
          );
        }}
      >
        <div>
          <Label htmlFor="key-name">New key for</Label>
          <Input
            id="key-name"
            required
            maxLength={64}
            placeholder="e.g. checkout-service"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-52"
          />
        </div>
        <fieldset className="flex flex-wrap gap-x-4 gap-y-1 pb-2">
          <legend className="mb-1 text-xs font-medium text-ink-2">Scopes</legend>
          {SCOPES.map(({ scope, help }) => (
            <label key={scope} className="flex items-center gap-1.5 text-sm text-ink" title={help}>
              <input
                type="checkbox"
                className="accent-[var(--accent)]"
                checked={scopes.includes(scope)}
                onChange={(e) => setScopes((s) => (e.target.checked ? [...s, scope] : s.filter((x) => x !== scope)))}
              />
              {scope}
            </label>
          ))}
        </fieldset>
        <Button type="submit" disabled={pending || !scopes.length}>
          Create key
        </Button>
        {error && <p className="w-full text-sm text-critical-ink">{error}</p>}
      </form>
    </Card>
  );
}

// Audit log --------------------------------------------------------------------------------------

function change(e: AuditEntry): string {
  const show = (v: unknown) => (v === null || v === undefined ? "" : JSON.stringify(v));
  const before = show(e.before);
  const after = show(e.after);
  if (before && after) return `${before} → ${after}`;
  return after || before || "";
}

export function AuditPanel({ first }: { first: AuditPage }) {
  const [entries, setEntries] = useState(first.entries);
  const [next, setNext] = useState(first.next_before);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function older() {
    if (!next) return;
    setLoading(true);
    const res = await fetch(`/api/fg/audit?limit=25&before=${encodeURIComponent(next)}`);
    setLoading(false);
    if (!res.ok) return setError(`Could not load older entries (${res.status})`);
    const page = (await res.json()) as AuditPage;
    setEntries((e) => [...e, ...page.entries]);
    setNext(page.next_before);
  }

  return (
    <Card className="overflow-x-auto">
      <CardHeader title="Audit log" subtitle="Verdicts, policy changes, promotions, users and keys, newest first" />
      {entries.length ? (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr className="border-y border-line">
              <th className="px-5 py-2 font-medium">When</th>
              <th className="px-5 py-2 font-medium">Who</th>
              <th className="px-5 py-2 font-medium">What</th>
              <th className="px-5 py-2 font-medium">Target</th>
              <th className="px-5 py-2 font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className="border-b border-line align-top last:border-0">
                <td className="tabular whitespace-nowrap px-5 py-2 text-ink-2">{dateTime(e.ts)}</td>
                <td className="px-5 py-2 text-ink">{e.actor}</td>
                <td className="px-5 py-2">
                  <code className="font-mono text-xs text-ink">{e.action}</code>
                </td>
                <td className="px-5 py-2 text-ink-2">{e.target ?? "—"}</td>
                <td className="max-w-md px-5 py-2">
                  <span className="line-clamp-2 break-all font-mono text-xs text-muted">{change(e)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="px-5 pb-5 text-sm text-muted">Nothing has been changed yet.</p>
      )}
      {(next || error) && (
        <div className="border-t border-line px-5 py-3">
          {error ? (
            <p className="text-sm text-critical-ink">{error}</p>
          ) : (
            <Button variant="secondary" disabled={loading} onClick={older}>
              {loading ? "Loading…" : "Load older"}
            </Button>
          )}
        </div>
      )}
    </Card>
  );
}
