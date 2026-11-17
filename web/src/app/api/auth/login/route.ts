import { NextResponse, type NextRequest } from "next/server";

import { errorDetail } from "@/lib/api";
import { API_PREFIX, API_URL } from "@/lib/config";
import { sessionCookies, type TokenResponse } from "@/lib/session";

export async function POST(req: NextRequest): Promise<NextResponse> {
  const body = (await req.json().catch(() => null)) as { username?: unknown; password?: unknown } | null;
  if (typeof body?.username !== "string" || typeof body?.password !== "string") {
    return NextResponse.json({ detail: "username and password are required" }, { status: 422 });
  }
  let res: Response;
  try {
    res = await fetch(`${API_URL}${API_PREFIX}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: body.username, password: body.password }),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ detail: "The FraudGuard API is not reachable." }, { status: 503 });
  }
  if (!res.ok) return NextResponse.json({ detail: await errorDetail(res) }, { status: res.status });

  const tokens = (await res.json()) as TokenResponse;
  const out = NextResponse.json({ role: tokens.role });
  for (const c of sessionCookies(tokens)) out.cookies.set(c.name, c.value, c.options);
  return out;
}
