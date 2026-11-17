// Guards the console and the API proxy, and refreshes the access token before it expires.
//
// The access cookie is set to expire shortly before the JWT does. When it is gone but the
// refresh cookie remains, the middleware swaps the refresh token for new tokens, sets them
// on the response, and passes the new access token on to the page in this same request.
import { NextResponse, type NextRequest } from "next/server";

import { ACCESS_COOKIE, REFRESH_COOKIE, clearedCookies, sessionCookies, type TokenResponse } from "@/lib/session";

const API_URL = (process.env.FRAUDGUARD_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export const config = { matcher: ["/console/:path*", "/api/fg/:path*"] };

function denied(req: NextRequest): NextResponse {
  const isApi = req.nextUrl.pathname.startsWith("/api/");
  const res = isApi
    ? NextResponse.json({ detail: "not signed in" }, { status: 401 })
    : NextResponse.redirect(new URL(`/login?next=${encodeURIComponent(req.nextUrl.pathname + req.nextUrl.search)}`, req.url));
  for (const c of clearedCookies()) res.cookies.set(c.name, c.value, c.options);
  return res;
}

export async function middleware(req: NextRequest): Promise<NextResponse> {
  if (req.cookies.get(ACCESS_COOKIE)?.value) return NextResponse.next();
  const refresh = req.cookies.get(REFRESH_COOKIE)?.value;
  if (!refresh) return denied(req);

  let tokens: TokenResponse;
  try {
    const r = await fetch(`${API_URL}/api/v1/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
      cache: "no-store",
    });
    if (!r.ok) return denied(req);
    tokens = (await r.json()) as TokenResponse;
  } catch {
    return denied(req);
  }

  const cookies = sessionCookies(tokens);
  for (const c of cookies) req.cookies.set(c.name, c.value); // visible to this request's handlers
  const res = NextResponse.next({ request: { headers: req.headers } });
  for (const c of cookies) res.cookies.set(c.name, c.value, c.options); // stored by the browser
  return res;
}
