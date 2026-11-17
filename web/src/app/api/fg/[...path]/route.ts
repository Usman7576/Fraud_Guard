// Browser -> Next.js -> FastAPI proxy for the console's interactive calls.
// Adds the bearer token from the httpOnly cookie, forwards only the endpoints the console
// needs, and refuses cross-site writes.
import { NextResponse, type NextRequest } from "next/server";

import { API_PREFIX, API_URL } from "@/lib/config";
import { isAllowed, sameOrigin } from "@/lib/proxy";
import { ACCESS_COOKIE } from "@/lib/session";

type Ctx = { params: Promise<{ path: string[] }> };

async function forward(req: NextRequest, { params }: Ctx): Promise<Response> {
  const path = (await params).path.join("/");
  if (!isAllowed(req.method, path)) return NextResponse.json({ detail: "not found" }, { status: 404 });
  if (req.method !== "GET" && !sameOrigin(req.headers.get("origin"), req.nextUrl.origin)) {
    return NextResponse.json({ detail: "cross-site request refused" }, { status: 403 });
  }
  const token = req.cookies.get(ACCESS_COOKIE)?.value;
  if (!token) return NextResponse.json({ detail: "not signed in" }, { status: 401 });

  const init: RequestInit = {
    method: req.method,
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    cache: "no-store",
  };
  if (req.method !== "GET") init.body = await req.text();
  try {
    const res = await fetch(`${API_URL}${API_PREFIX}/${path}${req.nextUrl.search}`, init);
    const headers: Record<string, string> = { "Content-Type": res.headers.get("Content-Type") ?? "application/json" };
    // Drift reports are HTML with their own scripts: keep the API's sandbox so they never run as the console.
    const csp = res.headers.get("Content-Security-Policy");
    if (csp) headers["Content-Security-Policy"] = csp;
    return new Response(await res.text(), { status: res.status, headers });
  } catch {
    return NextResponse.json({ detail: "The FraudGuard API is not reachable." }, { status: 503 });
  }
}

export const GET = forward;
export const PATCH = forward;
export const POST = forward;
export const PUT = forward;
