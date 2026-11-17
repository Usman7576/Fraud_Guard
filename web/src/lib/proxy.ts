// Rules for the browser -> FastAPI proxy (kept here so they are unit-testable).

const ALLOWED: [method: string, pattern: RegExp][] = [
  ["GET", /^cases$/],
  ["GET", /^cases\/\d+$/],
  ["PATCH", /^cases\/\d+$/],
  ["GET", /^kpi\/weekly$/],
  ["GET", /^policy$/],
  ["POST", /^score\/preview$/],
  ["GET", /^auth\/me$/],
  ["GET", /^monitoring\/drift\/\d+\/report$/],
  // Admin-only on the API side; the proxy only forwards the person's own token.
  ["POST", /^models\/\d+\/promote$/],
  ["POST", /^policy\/simulate$/],
  ["PUT", /^policy$/],
  ["GET", /^users$/],
  ["POST", /^users$/],
  ["PATCH", /^users\/[a-z0-9][a-z0-9_.-]{2,31}$/],
  ["GET", /^api-keys$/],
  ["POST", /^api-keys$/],
  ["POST", /^api-keys\/[0-9a-f]{24}\/revoke$/],
  ["GET", /^audit$/],
];

/** Only the endpoints the console uses are reachable through the proxy. */
export function isAllowed(method: string, path: string): boolean {
  return ALLOWED.some(([m, re]) => m === method && re.test(path));
}

/** Writes must come from the console's own origin (browsers always send Origin on POST/PATCH). */
export function sameOrigin(origin: string | null, ownOrigin: string): boolean {
  return origin === ownOrigin;
}
