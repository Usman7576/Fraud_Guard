// The landing page's live demo: public, so it is rate limited per visitor here and per key by the
// API. The demo key (scope "demo" only) stays on the server; the API scores without storing anything.
import { NextResponse, type NextRequest } from "next/server";

import { API_PREFIX, API_URL } from "@/lib/config";
import { RateWindow } from "@/lib/landing";
import { sameOrigin } from "@/lib/proxy";

const DEMO_KEY = process.env.FRAUDGUARD_DEMO_API_KEY ?? "";
const MAX_BODY = 2_000;
const perVisitor = new RateWindow(10);

function visitor(req: NextRequest): string {
  // Behind a proxy the first X-Forwarded-For hop is the visitor; locally there is none.
  return req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || req.headers.get("x-real-ip") || "local";
}

export async function POST(req: NextRequest): Promise<Response> {
  if (!sameOrigin(req.headers.get("origin"), req.nextUrl.origin)) {
    return NextResponse.json({ detail: "cross-site request refused" }, { status: 403 });
  }
  if (!DEMO_KEY) {
    return NextResponse.json(
      { detail: "The live demo is not configured: set FRAUDGUARD_DEMO_API_KEY for the web server." },
      { status: 503 },
    );
  }
  if (!perVisitor.allow(visitor(req))) {
    return NextResponse.json({ detail: "Too many tries. Wait a minute and score again." }, { status: 429 });
  }
  const body = await req.text();
  if (body.length > MAX_BODY) return NextResponse.json({ detail: "request too large" }, { status: 413 });
  try {
    const res = await fetch(`${API_URL}${API_PREFIX}/demo/score`, {
      method: "POST",
      headers: { "X-API-Key": DEMO_KEY, "Content-Type": "application/json" },
      body,
      cache: "no-store",
    });
    return new Response(await res.text(), {
      status: res.status,
      headers: { "Content-Type": res.headers.get("Content-Type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json({ detail: "The FraudGuard API is not reachable." }, { status: 503 });
  }
}
