// Session cookies. Tokens live only in httpOnly cookies, so page scripts never see them.
// No Node-only imports here: middleware (edge runtime) uses this module too.

export const ACCESS_COOKIE = "fg_access";
export const REFRESH_COOKIE = "fg_refresh";
export const USER_COOKIE = "fg_user";

/** The access cookie expires this many seconds before the token, so it is refreshed in time. */
export const EXPIRY_MARGIN_S = 30;
const REFRESH_MAX_AGE_S = 7 * 24 * 3600;

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  role: string;
  expires_in: number;
}

export interface CookieSpec {
  name: string;
  value: string;
  options: { httpOnly: true; sameSite: "lax"; secure: boolean; path: "/"; maxAge: number };
}

function opts(maxAge: number): CookieSpec["options"] {
  return { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", path: "/", maxAge };
}

/** Cookies to set after a login or a refresh. */
export function sessionCookies(t: TokenResponse): CookieSpec[] {
  const user = JSON.stringify({ username: tokenSubject(t.access_token), role: t.role });
  return [
    { name: ACCESS_COOKIE, value: t.access_token, options: opts(Math.max(t.expires_in - EXPIRY_MARGIN_S, 1)) },
    { name: REFRESH_COOKIE, value: t.refresh_token, options: opts(REFRESH_MAX_AGE_S) },
    { name: USER_COOKIE, value: user, options: opts(REFRESH_MAX_AGE_S) },
  ];
}

export function clearedCookies(): CookieSpec[] {
  return [ACCESS_COOKIE, REFRESH_COOKIE, USER_COOKIE].map((name) => ({ name, value: "", options: opts(0) }));
}

/** The JWT's `sub` claim, for display only (the API verifies every token itself). */
export function tokenSubject(jwt: string): string {
  const part = jwt.split(".")[1];
  if (!part) return "";
  try {
    const padded = part.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(part.length / 4) * 4, "=");
    const sub = (JSON.parse(atob(padded)) as { sub?: unknown }).sub;
    return typeof sub === "string" ? sub : "";
  } catch {
    return "";
  }
}

export function parseUser(raw: string | undefined): { username: string; role: string } | null {
  if (!raw) return null;
  try {
    const u = JSON.parse(raw) as { username?: unknown; role?: unknown };
    return typeof u.username === "string" && typeof u.role === "string"
      ? { username: u.username, role: u.role }
      : null;
  } catch {
    return null;
  }
}

/** Only same-site relative paths are allowed as a post-login destination. */
export function safeNext(next: string | null | undefined): string {
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : "/console";
}
