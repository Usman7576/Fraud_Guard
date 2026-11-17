import { NextResponse, type NextRequest } from "next/server";

import { clearedCookies } from "@/lib/session";

export function POST(req: NextRequest): NextResponse {
  const res = NextResponse.redirect(new URL("/login", req.url), { status: 303 });
  for (const c of clearedCookies()) res.cookies.set(c.name, c.value, c.options);
  return res;
}
