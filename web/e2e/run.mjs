// End-to-end run: a throwaway MongoDB database, the real API and the production web build, then
// Playwright against them, then everything is stopped and the database dropped.
//
//   npm run e2e            (from web/; needs MongoDB running and the model trained)
//   npm run e2e -- --headed   (extra arguments go to Playwright)
import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.resolve(web, "..");
const isWin = process.platform === "win32";
const venvPython = path.join(root, ".venv", isWin ? "Scripts/python.exe" : "bin/python");
const python = existsSync(venvPython) ? [venvPython] : ["uv", "run", "python"];
const API_PORT = 8765;
const WEB_PORT = 3765;

const secret = () => randomBytes(18).toString("base64url");
const env = {
  ...process.env,
  FRAUDGUARD_MONGO_DB: `fraudguard_e2e_${randomBytes(3).toString("hex")}`,
  FRAUDGUARD_JWT_SECRET: secret() + secret(),
  E2E_ADMIN_PASSWORD: secret(),
  E2E_ANALYST_PASSWORD: secret(),
  E2E_API_URL: `http://127.0.0.1:${API_PORT}`,
};

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { cwd: root, env, encoding: "utf-8", shell: false, ...opts });
  if (r.status !== 0) throw new Error(`${cmd} ${args.join(" ")} failed:\n${r.stdout ?? ""}${r.stderr ?? ""}`);
  return r.stdout ?? "";
}

function start(name, cmd, args, opts) {
  const child = spawn(cmd, args, { env, stdio: ["ignore", "pipe", "pipe"], ...opts });
  let log = "";
  child.stdout.on("data", (d) => (log += d));
  child.stderr.on("data", (d) => (log += d));
  child.on("exit", (code) => code && console.error(`${name} exited with ${code}\n${log.slice(-3000)}`));
  return { child, log: () => log };
}

function stop(proc) {
  if (!proc || proc.child.exitCode !== null) return;
  if (isWin) spawnSync("taskkill", ["/pid", String(proc.child.pid), "/T", "/F"], { stdio: "ignore" });
  else proc.child.kill("SIGTERM");
}

async function waitFor(url, what, ms = 180_000) {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`${what} did not start within ${ms / 1000}s`);
}

let api;
let site;
let code = 1;
try {
  console.log(`e2e database: ${env.FRAUDGUARD_MONGO_DB}`);
  const keys = JSON.parse(run(python[0], [...python.slice(1), "scripts/e2e_env.py", "setup"]).trim().split("\n").at(-1));
  env.FRAUDGUARD_DEMO_API_KEY = keys["e2e-demo"];
  env.E2E_SCORE_KEY = keys["e2e-score"];
  env.FRAUDGUARD_API_URL = env.E2E_API_URL;

  api = start("API", python[0], [...python.slice(1), "-m", "fraudguard.cli", "serve", "--port", String(API_PORT)], { cwd: root });
  if (!existsSync(path.join(web, ".next", "BUILD_ID")) || process.env.E2E_REBUILD) {
    console.log("building the web app…");
    run(isWin ? "npm.cmd" : "npm", ["run", "build"], { cwd: web, stdio: "inherit", shell: isWin });
  }
  site = start("web", process.execPath, [path.join(web, "node_modules", "next", "dist", "bin", "next"), "start", "--port", String(WEB_PORT)], {
    cwd: web,
  });
  await waitFor(`${env.E2E_API_URL}/api/v1/health`, "the API");
  console.log("scoring test-month payments:", run(python[0], [...python.slice(1), "scripts/e2e_env.py", "score"]).trim());
  await waitFor(`http://127.0.0.1:${WEB_PORT}/`, "the web app");

  const pw = spawnSync(
    process.execPath,
    [path.join(web, "node_modules", "@playwright", "test", "cli.js"), "test", ...process.argv.slice(2)],
    { cwd: web, env: { ...env, E2E_BASE_URL: `http://localhost:${WEB_PORT}` }, stdio: "inherit" },
  );
  code = pw.status ?? 1;
} catch (e) {
  console.error(e instanceof Error ? e.message : e);
  if (api) console.error(`--- API log ---\n${api.log().slice(-3000)}`);
} finally {
  stop(site);
  stop(api);
  try {
    run(python[0], [...python.slice(1), "scripts/e2e_env.py", "teardown"]);
  } catch (e) {
    console.error(`could not drop ${env.FRAUDGUARD_MONGO_DB}: ${e}`);
  }
}
process.exit(code);
