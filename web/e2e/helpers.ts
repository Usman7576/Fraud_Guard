import { expect, type Page } from "@playwright/test";

export const ADMIN = { username: "e2e-admin", password: process.env.E2E_ADMIN_PASSWORD ?? "" };
export const ANALYST = { username: "e2e-analyst", password: process.env.E2E_ANALYST_PASSWORD ?? "" };

export async function signIn(page: Page, who: { username: string; password: string }, next = "/console") {
  await page.goto(`/login?next=${encodeURIComponent(next)}`);
  await page.getByLabel("Username").fill(who.username);
  await page.getByLabel("Password").fill(who.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(new RegExp(`${next.replace(/[/?]/g, "\\$&")}$`));
}

/** Fails the test on uncaught page errors (hydration problems, runtime exceptions). */
export function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error" && !/Failed to load resource/.test(m.text())) errors.push(m.text());
  });
  return errors;
}
