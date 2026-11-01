import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const PUBLIC_IMAGE = process.env.VERIFAI_E2E_IMAGE ?? "";
const PRIVATE_IMAGE = process.env.VERIFAI_E2E_PRIVATE_IMAGE ?? "";
const PRIVATE_VIDEO = process.env.VERIFAI_E2E_PRIVATE_VIDEO ?? "";
const SHOTS = process.env.VERIFAI_E2E_SHOTS ?? path.resolve("e2e-shots");

type Tracked = { __problems: string[] };

// Any console error or uncaught exception fails the test that caused it.
test.beforeEach(async ({ page }) => {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    // an expected 4xx/5xx response (e.g. a rejected upload) is logged by the browser; it is asserted on explicitly where it matters
    if (m.type() === "error" && !/Failed to load resource/.test(m.text())) problems.push(`console.error: ${m.text()}`);
  });
  (page as unknown as Tracked).__problems = problems;
});
test.afterEach(async ({ page }) => {
  expect((page as unknown as Tracked).__problems, "console errors / page errors").toEqual([]);
});

const shot = async (page: Page, name: string) => {
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: true });
};
const noHScroll = async (page: Page) => expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);

async function upload(page: Page, file: string) {
  await page.goto("/#/");
  await page.locator("input[type=file]").setInputFiles(file);
  await expect(page.getByTestId("preview")).toBeVisible();
  await page.getByRole("button", { name: /^analyse (image|video)$/i }).click();
}

test("health, API docs and the app shell are reachable", async ({ page, request }) => {
  const h = await (await request.get("/api/health")).json();
  expect(h.status).toBe("ok");
  expect(h.model_loaded).toBe(true);
  expect((await request.get("/docs")).status()).toBe(200);
  expect((await request.get("/openapi.json")).status()).toBe(200);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels");
});

test("navigation reaches every section and marks the current page", async ({ page }) => {
  await page.goto("/#/");
  const nav = page.getByRole("navigation", { name: "Main" });
  for (const [label, heading] of [["How It Works", /how it works/i], ["Model", /^model$/i], ["Reports", /^reports$/i], ["About", /about verif-ai/i], ["Home", /see beyond the pixels/i]] as const) {
    await nav.getByRole("link", { name: label }).click();
    await expect(page.getByRole("heading", { level: 1 })).toContainText(heading);
    await expect(nav.getByRole("link", { name: label })).toHaveAttribute("aria-current", "page");
  }
  await page.goto("/#/model");
  await expect(page.getByTestId("model-card")).toContainText("EfficientNet-B0"); // read live from /api/model
});

test("upload tabs switch between image and video and show the real limits", async ({ page }) => {
  await page.goto("/#/");
  await expect(page.getByRole("tab", { name: /upload image/i })).toHaveAttribute("aria-selected", "true");
  await page.getByRole("tab", { name: /upload video/i }).click();
  await expect(page.getByText(/drag and drop a video here/i)).toBeVisible();
  await expect(page.getByText(/MP4/i).first()).toBeVisible();
});

test("unsupported and invalid uploads are rejected with a clear message", async ({ page }, info) => {
  await page.goto("/#/");
  const txt = info.outputPath("notes.txt");
  fs.mkdirSync(path.dirname(txt), { recursive: true });
  fs.writeFileSync(txt, "not media");
  await page.locator("input[type=file]").setInputFiles(txt);
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(page.getByTestId("preview")).toHaveCount(0);

  // a file that claims to be a video but cannot be decoded: the server must refuse it and the UI must say so
  const broken = info.outputPath("broken.mp4");
  fs.writeFileSync(broken, Buffer.from("this is not a real mp4 file".repeat(50)));
  await page.getByRole("button", { name: /dismiss/i }).click();
  await page.locator("input[type=file]").setInputFiles(broken);
  await page.getByRole("button", { name: /analyse video/i }).click();
  await expect(page.getByRole("alert").first()).toBeVisible({ timeout: 60_000 });
});

test("unknown analysis ids show a not-found message, not a crash", async ({ page }) => {
  await page.goto(`/#/analysis/${"0".repeat(32)}`);
  await expect(page.getByRole("alert")).toContainText(/no longer exists|deleted|expired/i);
});

test("public image: upload, progress, results, Grad-CAM, PDF, deletion", async ({ page, request }) => {
  test.skip(!PUBLIC_IMAGE, "VERIFAI_E2E_IMAGE not set");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#/");
  await shot(page, "01_home");
  await page.locator("input[type=file]").setInputFiles(PUBLIC_IMAGE);
  await expect(page.getByAltText("Selected file preview")).toBeVisible();
  await shot(page, "02_preview");
  await page.getByRole("button", { name: /analyse image/i }).click();
  await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 120_000 });
  expect(page.url()).toMatch(/#\/analysis\/[0-9a-f]{32}$/);
  const id = page.url().split("/").pop()!;

  await expect(page.getByTestId("verdict").first()).toHaveText(/REAL|FAKE|INCONCLUSIVE/);
  await expect(page.getByTestId("fake-score")).toHaveText(/\d\.\d{3}/);
  await expect(page.getByText(/not a probability that the verdict is correct/i)).toBeVisible();
  await shot(page, "03_result_overview");

  await page.getByRole("tab", { name: /forensic evidence/i }).click();
  await expect(page.getByAltText(/image with face bounding boxes/i)).toBeVisible();
  await expect(page.getByRole("table")).toBeVisible();
  const cam = page.getByAltText(/Grad-CAM heat-map/i).first();
  await expect(cam).toBeVisible();
  expect(await cam.evaluate((i: HTMLImageElement) => i.complete && i.naturalWidth > 0), "Grad-CAM image decodes").toBe(true);
  await shot(page, "04_result_evidence");

  await page.getByRole("tab", { name: /report/i }).click();
  await expect(page.getByText(/not definitive proof/i)).toBeVisible();
  await shot(page, "05_result_report");

  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("link", { name: /download forensic report/i }).click()]);
  const pdf = fs.readFileSync(await dl.path());
  expect(pdf.subarray(0, 5).toString()).toBe("%PDF-");
  expect(pdf.length).toBeGreaterThan(5_000);

  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels");
  expect((await request.get(`/api/analyses/${id}`)).status()).toBe(404);
  expect((await request.get(`/api/analyses/${id}/report.pdf`)).status()).toBe(404);
});

test("responsive layouts: no horizontal scroll, navigation usable", async ({ page }) => {
  for (const [w, h] of [[1440, 900], [1024, 768], [768, 1024], [390, 844]] as const) {
    await page.setViewportSize({ width: w, height: h });
    for (const hash of ["#/", "#/how-it-works", "#/model", "#/reports", "#/about"]) {
      await page.goto(`/${hash}`);
      await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
      await noHScroll(page);
    }
    if (w < 768) {
      await page.goto("/#/");
      await page.getByRole("button", { name: /open menu/i }).click();
      await page.getByRole("navigation", { name: /mobile/i }).getByRole("link", { name: "About" }).click();
      await expect(page.getByRole("heading", { level: 1 })).toContainText(/about/i);
      await expect(page.getByRole("navigation", { name: /mobile/i })).toHaveCount(0); // the menu closes after navigating
    }
  }
  if (PUBLIC_IMAGE) {
    for (const [w, h, name] of [[390, 844, "06_mobile_result"], [768, 1024, "07_tablet_result"]] as const) {
      await page.setViewportSize({ width: w, height: h });
      await upload(page, PUBLIC_IMAGE);
      await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 120_000 });
      await noHScroll(page);
      await shot(page, name);
      await page.getByRole("button", { name: /delete results/i }).click();
    }
  }
});

// ---- personal media: workflow only. No screenshots, no filenames and no scores are written anywhere. ----
test("private image: the real-image workflow completes and can be deleted", async ({ page, request }) => {
  test.skip(!PRIVATE_IMAGE, "VERIFAI_E2E_PRIVATE_IMAGE not set");
  await upload(page, PRIVATE_IMAGE);
  await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 120_000 });
  const id = page.url().split("/").pop()!;
  await expect(page.getByTestId("verdict").first()).toHaveText(/REAL|FAKE|INCONCLUSIVE/);
  await page.getByRole("tab", { name: /forensic evidence/i }).click();
  expect((await request.get(`/api/analyses/${id}/report.pdf`)).status()).toBe(200);
  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels"); // the redirect happens after the DELETE succeeded
  expect((await request.get(`/api/analyses/${id}`)).status()).toBe(404);
});

test("private video: background job, timeline, tracks, evidence, PDF and deletion", async ({ page, request }) => {
  test.skip(!PRIVATE_VIDEO, "VERIFAI_E2E_PRIVATE_VIDEO not set");
  await page.goto("/#/");
  await page.getByRole("tab", { name: /upload video/i }).click();
  await page.locator("input[type=file]").setInputFiles(PRIVATE_VIDEO);
  await expect(page.getByLabel("Selected video preview")).toBeVisible();
  await page.getByRole("button", { name: /analyse video/i }).click();
  await expect(page.getByTestId("progress")).toBeVisible();
  await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 12 * 60_000 });
  const id = page.url().split("/").pop()!;
  await expect(page.getByTestId("verdict").first()).toHaveText(/REAL|FAKE|INCONCLUSIVE/);
  await expect(page.getByRole("tab")).toHaveCount(5);
  await page.getByRole("tab", { name: /frame analysis/i }).click();
  await expect(page.getByRole("img", { name: /fake-score timeline/i })).toBeVisible();
  await expect(page.getByRole("heading", { name: /suspicious timestamps/i })).toBeVisible();
  await page.getByRole("tab", { name: /face tracks/i }).click();
  await expect(page.getByRole("heading", { name: /face tracks/i })).toBeVisible();
  await page.getByRole("tab", { name: /forensic evidence/i }).click();
  await expect(page.getByRole("heading", { name: /highest-scoring frames/i })).toBeVisible();
  const pdf = await request.get(`/api/analyses/${id}/report.pdf`);
  expect(pdf.status()).toBe(200);
  expect((await pdf.body()).subarray(0, 5).toString()).toBe("%PDF-");
  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels"); // the redirect happens after the DELETE succeeded
  expect((await request.get(`/api/analyses/${id}`)).status()).toBe(404);
});
