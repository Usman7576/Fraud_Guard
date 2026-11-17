// Reads the evaluation report for the landing page. The page is static, so this runs at build
// time (and in development on each request): rebuild after `fraudguard evaluate` to refresh it.
import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

import { parseResults, type LandingResults } from "@/lib/landing";

const REPORTS_DIR = process.env.FRAUDGUARD_REPORTS_DIR ?? path.join(process.cwd(), "..", "reports");

export async function loadResults(): Promise<LandingResults | null> {
  try {
    return parseResults(JSON.parse(await readFile(path.join(REPORTS_DIR, "evaluation.json"), "utf-8")));
  } catch {
    return null; // not evaluated yet (reports/ is git-ignored)
  }
}
