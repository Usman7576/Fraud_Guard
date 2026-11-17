import { describe, expect, it } from "vitest";

import { count, due, featureValue, money, percent } from "@/lib/format";
import { isAllowed, sameOrigin } from "@/lib/proxy";
import { EXPIRY_MARGIN_S, parseUser, safeNext, sessionCookies, tokenSubject } from "@/lib/session";

function jwt(payload: object): string {
  const b64 = (s: string) => btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return `${b64('{"alg":"HS256"}')}.${b64(JSON.stringify(payload))}.sig`;
}

describe("format", () => {
  it("counts in full below 10,000 and compact above", () => {
    expect(count(1284)).toBe("1,284");
    expect(count(12_900)).toBe("12.9K");
  });
  it("formats money and percentages", () => {
    expect(money(1234.5)).toBe("$1,234.50");
    expect(percent(0.0342)).toBe("3.4%");
    expect(percent(0.0004)).toBe("0.04%");
    expect(percent(null)).toBe("—");
  });
  it("describes deadlines", () => {
    const now = new Date("2026-01-01T12:00:00Z");
    expect(due("2026-01-01T15:00:00Z", now)).toBe("in 3h");
    expect(due("2026-01-01T10:00:00Z", now)).toBe("2h overdue");
    expect(due(null, now)).toBe("—");
  });
  it("shows feature values plainly", () => {
    expect(featureValue(null)).toBe("—");
    expect(featureValue(true)).toBe("yes");
    expect(featureValue(0.123456)).toBe("0.1235");
    expect(featureValue(4200)).toBe("4,200");
  });
});

describe("session", () => {
  it("reads the subject from a JWT, including unpadded base64url", () => {
    expect(tokenSubject(jwt({ sub: "ana", role: "analyst" }))).toBe("ana");
    expect(tokenSubject("garbage")).toBe("");
  });
  it("expires the access cookie before the token", () => {
    const cookies = sessionCookies({
      access_token: jwt({ sub: "ana" }),
      refresh_token: "r",
      token_type: "bearer",
      role: "analyst",
      expires_in: 900,
    });
    const access = cookies.find((c) => c.name === "fg_access");
    expect(access?.options.maxAge).toBe(900 - EXPIRY_MARGIN_S);
    expect(cookies.every((c) => c.options.httpOnly && c.options.sameSite === "lax")).toBe(true);
    expect(parseUser(cookies.find((c) => c.name === "fg_user")?.value)).toEqual({ username: "ana", role: "analyst" });
  });
  it("only redirects to same-site paths after login", () => {
    expect(safeNext("/console/cases?status=open")).toBe("/console/cases?status=open");
    expect(safeNext("https://evil.example")).toBe("/console");
    expect(safeNext("//evil.example")).toBe("/console");
    expect(safeNext("/\\evil.example")).toBe("/console");
    expect(safeNext(undefined)).toBe("/console");
  });
});

describe("proxy", () => {
  it("forwards only the console's endpoints", () => {
    expect(isAllowed("GET", "cases")).toBe(true);
    expect(isAllowed("PATCH", "cases/123")).toBe(true);
    expect(isAllowed("POST", "score/preview")).toBe(true);
    expect(isAllowed("PUT", "policy")).toBe(false); // admin settings are not exposed here
    expect(isAllowed("POST", "models/3/promote")).toBe(false);
    expect(isAllowed("GET", "cases/../policy")).toBe(false);
    expect(isAllowed("POST", "score")).toBe(false); // live scoring needs an API key, not a person
  });
  it("refuses cross-site writes", () => {
    expect(sameOrigin("http://localhost:3000", "http://localhost:3000")).toBe(true);
    expect(sameOrigin("https://evil.example", "http://localhost:3000")).toBe(false);
    expect(sameOrigin(null, "http://localhost:3000")).toBe(false);
  });
});
