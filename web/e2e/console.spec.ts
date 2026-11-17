import { expect, test } from "@playwright/test";

import { ADMIN, ANALYST, signIn, watchErrors } from "./helpers";

test("the console needs a sign-in and rejects a wrong password", async ({ page }) => {
  await page.goto("/console/cases");
  await expect(page).toHaveURL(/\/login\?next=%2Fconsole%2Fcases/);
  await page.getByLabel("Username").fill(ANALYST.username);
  await page.getByLabel("Password").fill("not-the-password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("alert")).toBeVisible();
});

test("an analyst works a case from the queue", async ({ page }) => {
  const errors = watchErrors(page);
  await signIn(page, ANALYST);
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
  const nav = page.getByRole("navigation", { name: "Console" });
  await expect(nav.getByRole("link", { name: "Review queue" })).toBeVisible();
  for (const adminOnly of ["Models", "Cost explorer", "Settings"]) {
    await expect(nav.getByRole("link", { name: adminOnly })).toHaveCount(0);
  }

  await nav.getByRole("link", { name: "Review queue" }).click();
  await expect(page.getByRole("heading", { name: "Review queue" })).toBeVisible();
  const firstCase = page.locator('a[href^="/console/cases/"]').first();
  await expect(firstCase).toBeVisible();
  await firstCase.click();
  await expect(page).toHaveURL(/\/console\/cases\/\d+$/);
  await page.getByRole("button", { name: "Genuine" }).click();
  await expect(page.getByText(/closed/i).first()).toBeVisible();

  await page.goto("/console/settings");
  await expect(page.getByText("Admins only")).toBeVisible();
  expect(errors).toEqual([]);
});

test("an admin explores costs without applying them", async ({ page }) => {
  const errors = watchErrors(page);
  await signIn(page, ADMIN, "/console/cost");
  await expect(page.getByRole("heading", { name: "Cost explorer" })).toBeVisible();
  await expect(page.getByText("Thresholds this policy implies")).toBeVisible({ timeout: 20_000 });
  const tile = page.getByText("Cost per 10,000 payments").first().locator("..");
  await expect(tile).toContainText("same as now");

  const slider = page.getByLabel("False decline cost");
  await slider.focus();
  for (let i = 0; i < 20; i++) await slider.press("ArrowRight");
  await expect(tile).toContainText("vs now");
  await expect(page.getByRole("button", { name: "Apply this policy" })).toBeEnabled();
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(tile).toContainText("same as now");
  expect(errors).toEqual([]);
});

test("an admin adds a user, creates and revokes an API key, and sees it audited", async ({ page }) => {
  const errors = watchErrors(page);
  await signIn(page, ADMIN, "/console/settings");
  const name = `e2e-key-${Date.now()}`;

  await page.getByLabel("New user").fill("e2e-newcomer");
  await page.locator("#new-password").fill("newcomer-password-1");
  await page.getByRole("button", { name: "Add user" }).click();
  await expect(page.getByRole("cell", { name: "e2e-newcomer", exact: true })).toBeVisible();

  await page.getByLabel("New key for").fill(name);
  await page.getByRole("button", { name: "Create key" }).click();
  await expect(page.getByTestId("new-api-key")).toHaveText(/^fg_/);
  await page.getByRole("button", { name: "Done" }).click();
  await expect(page.getByTestId("new-api-key")).toHaveCount(0); // shown once

  const keys = page.locator("section", { has: page.getByRole("heading", { name: "API keys" }) });
  const row = keys.getByRole("row", { name: new RegExp(name) });
  page.once("dialog", (d) => d.accept());
  await row.getByRole("button", { name: "Revoke" }).click();
  await expect(row).toContainText("revoked");

  const audit = page.locator("section", { has: page.getByRole("heading", { name: "Audit log" }) });
  await expect(audit).toContainText("api_key.revoke");
  await expect(audit).toContainText("user.create");
  expect(errors).toEqual([]);
});
