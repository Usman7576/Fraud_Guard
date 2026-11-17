import { describe, expect, it } from "vitest";

import { count, due, featureValue, money, percent } from "@/lib/format";
import { RateWindow, fmt, gaugePosition, parseResults, tickLabel } from "@/lib/landing";
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
    expect(isAllowed("PUT", "policy")).toBe(true); // the API itself requires the admin role
    expect(isAllowed("PATCH", "users/bea")).toBe(true);
    expect(isAllowed("PATCH", "users/../policy")).toBe(false);
    expect(isAllowed("POST", "api-keys/0123456789abcdef01234567/revoke")).toBe(true);
    expect(isAllowed("POST", "api-keys/x/revoke")).toBe(false);
    expect(isAllowed("DELETE", "users/bea")).toBe(false);
    expect(isAllowed("POST", "models/3/promote")).toBe(true); // the API itself requires the admin role
    expect(isAllowed("POST", "ops/models/3/promote")).toBe(false); // machine-key route never via the browser
    expect(isAllowed("POST", "labels")).toBe(false);
    expect(isAllowed("GET", "monitoring/drift/24/report")).toBe(true);
    expect(isAllowed("GET", "cases/../policy")).toBe(false);
    expect(isAllowed("POST", "score")).toBe(false); // live scoring needs an API key, not a person
  });
  it("refuses cross-site writes", () => {
    expect(sameOrigin("http://localhost:3000", "http://localhost:3000")).toBe(true);
    expect(sameOrigin("https://evil.example", "http://localhost:3000")).toBe(false);
    expect(sameOrigin(null, "http://localhost:3000")).toBe(false);
  });
});

describe("landing", () => {
  it("places scores on a log gauge", () => {
    expect(gaugePosition(1e-4)).toBe(0);
    expect(gaugePosition(0)).toBe(0);
    expect(gaugePosition(0.01)).toBeCloseTo(0.5);
    expect(gaugePosition(1)).toBe(1);
    expect(gaugePosition(Number.NaN)).toBe(0);
    expect(tickLabel(0.001)).toBe("0.1%");
  });
  it("formats result figures", () => {
    expect(fmt(0.91667, "percent")).toBe("91.7%");
    expect(fmt(-1102.4, "money")).toBe("-$1,102");
    expect(fmt(0.88445, "decimal")).toBe("0.884");
  });
  it("limits each visitor per minute", () => {
    const w = new RateWindow(2, 2);
    expect([w.allow("a", 0), w.allow("a", 1), w.allow("a", 2)]).toEqual([true, true, false]);
    expect(w.allow("a", 60_001)).toBe(true); // the window slid past the first two
    w.allow("b", 60_002);
    w.allow("c", 60_003); // evicts the least recently used visitor ("a")
    expect(w.allow("a", 60_004)).toBe(true);
  });
  it("reads the evaluation report, or nothing", () => {
    const iv = (estimate: number) => ({ estimate, low: estimate - 0.1, high: estimate + 0.1 });
    const report = {
      model_version: "5",
      n_rows: 3362,
      n_fraud: 108,
      n_resamples: 1000,
      policy: { false_decline_cost: 15, review_cost: 3, review_capacity_per_day: 200 },
      intervals: {
        model_recall_at_fpr: iv(0.9),
        rules_recall_at_fpr: iv(0.8),
        cost_saved_per_10k: iv(8900),
        model_pr_auc: iv(0.88),
        rules_pr_auc: iv(0.69),
      },
      shap_global: [["C1", 0.7], ["graph_2hop_fraud_cards", 0.6]],
    };
    const r = parseResults(report);
    expect(r?.recall.estimate).toBe(0.9);
    expect(r?.graphRank).toBe(2);
    expect(r?.isSample).toBe(true);
    expect(parseResults({ ...report, intervals: {} })).toBeNull();
    expect(parseResults(null)).toBeNull();
  });
});
