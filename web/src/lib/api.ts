// Server-side calls to the FastAPI service, authenticated with the session cookie.
import "server-only";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { API_PREFIX, API_URL } from "@/lib/config";
import { ACCESS_COOKIE, USER_COOKIE, parseUser } from "@/lib/session";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  const token = (await cookies()).get(ACCESS_COOKIE)?.value;
  if (!token) redirect("/login");
  let res: Response;
  try {
    res = await fetch(`${API_URL}${API_PREFIX}${path}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(503, `The FraudGuard API is not reachable at ${API_URL}. Is \`fraudguard serve\` running?`);
  }
  if (res.status === 401) redirect("/login");
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  return (await res.json()) as T;
}

export async function errorDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
  } catch {
    // not JSON
  }
  return `API error ${res.status}`;
}

export async function currentUser(): Promise<{ username: string; role: string } | null> {
  return parseUser((await cookies()).get(USER_COOKIE)?.value);
}
