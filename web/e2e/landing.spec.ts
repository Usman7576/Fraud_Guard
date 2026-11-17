import { expect, test } from "@playwright/test";

import { watchErrors } from "./helpers";

test("landing page renders, scores a payment live and has no horizontal scroll", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Catch the fraud.");
  const hero = page.locator("[data-hero-art]");
  if (test.info().project.name === "phone") {
    // Phones keep the SVG globe, animated with CSS: no WebGL start-up on the main thread.
    await page.waitForTimeout(2500);
    await expect(hero).toHaveAttribute("data-hero-art", "svg");
    await expect(hero.locator("canvas")).toHaveCount(0);
    await expect(hero.locator(".fg-pulse").first()).toBeVisible();
  } else {
    // Laptops and desktops get the WebGL globe once the browser is idle.
    await expect(hero.locator("canvas")).toBeVisible({ timeout: 15_000 });
    await expect(hero).toHaveAttribute("data-hero-art", "webgl");
  }

  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "Risky first purchase" }).click();
  await page.getByRole("button", { name: "Score this payment" }).click();
  await expect(page.getByText("chance of fraud", { exact: true })).toBeVisible();
  await expect(page.locator("#demo li")).toHaveCount(3);
  await expect(page.locator("#demo").getByText(/^model v\d+|· model v\d+/)).toBeVisible();

  await page.locator("#results").scrollIntoViewIfNeeded();
  await expect(page.locator("#results h2")).toHaveText("Measured on a month the model never saw.");

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  expect(errors).toEqual([]);
});

test("reduced motion gets static art, a plain step list and final numbers", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.waitForTimeout(2500); // past the idle callback that would load the 3D scenes
  await expect(page.locator("canvas")).toHaveCount(0);
  const graph = page.locator("#graph");
  await expect(graph.getByRole("img", { name: /Card A's chargeback/ })).toBeVisible();
  for (const step of ["Every payment leaves links", "A chargeback confirms card A", "One hop: what card A used", "Two hops: the ring"]) {
    await expect(graph.getByText(step)).toBeVisible();
  }
  const height = await graph.evaluate((el) => el.getBoundingClientRect().height);
  expect(height).toBeLessThan(2 * (page.viewportSize()?.height ?? 800)); // not the tall scroll story
});

test("the demo endpoint refuses cross-site posts", async ({ request }) => {
  const r = await request.post("/api/demo/score", {
    headers: { Origin: "https://evil.example" },
    data: { TransactionAmt: 10 },
  });
  expect(r.status()).toBe(403);
});
