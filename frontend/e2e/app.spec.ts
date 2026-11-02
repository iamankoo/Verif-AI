import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const PUBLIC_IMAGE = process.env.VERIFAI_E2E_IMAGE ?? "";
// Cross-origin deployments (Vercel UI + hosted API): set VERIFAI_E2E_API to the backend origin. Empty = same origin.
const API = (process.env.VERIFAI_E2E_API ?? "").replace(/\/+$/, "");
// Grad-CAM can be switched off on small hosts (VERIFAI_ENABLE_HEATMAPS=false, e.g. a 512 MB free instance). The UI must then say so honestly.
async function gradCamOn(request: import("@playwright/test").APIRequestContext): Promise<boolean> {
  const m = await (await request.get(`${API}/api/model`)).json();
  return !!m.explainability?.grad_cam_available;
}
const PUBLIC_VIDEO = process.env.VERIFAI_E2E_PUBLIC_VIDEO ?? "";   // synthetic clip made from the public test image
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
const noHScroll = async (page: Page) => {
  const m = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, view: window.innerWidth, url: location.hash }));
  expect(m.scroll, `horizontal overflow on ${m.url} at ${m.view}px (scrollWidth ${m.scroll})`).toBeLessThanOrEqual(m.view + 1);
};

async function upload(page: Page, file: string) {
  await page.goto("/#/");
  await page.locator("input[type=file]").setInputFiles(file);
  await expect(page.getByTestId("preview")).toBeVisible();
  await page.getByRole("button", { name: /^analyse (image|video)$/i }).click();
}

test("health, API docs and the app shell are reachable", async ({ page, request }) => {
  const h = await (await request.get(`${API}/api/health`)).json();
  expect(h.status).toBe("ok");
  expect(h.model_loaded).toBe(true);
  expect((await request.get(`${API}/docs`)).status()).toBe(200);
  expect((await request.get(`${API}/openapi.json`)).status()).toBe(200);
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

test("one unified upload card (no tabs) shows the supported formats and limits", async ({ page }) => {
  await page.goto("/#/");
  await expect(page.getByRole("heading", { name: "Upload Media" })).toBeVisible();
  await expect(page.getByRole("tab", { name: /upload (image|video)/i })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Choose File" })).toBeVisible();
  await expect(page.getByText(/Images: .*JPG/)).toBeVisible();
  await expect(page.getByText(/Videos: .*MP4/)).toBeVisible();
  await expect(page.getByText(/Max size: .*MB/)).toBeVisible();
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

test("direct navigation and a page refresh keep the same page (deep links work on the hosted URL)", async ({ page }) => {
  for (const [hash, heading] of [["#/model", /^model$/i], ["#/how-it-works", /how it works/i], ["#/reports", /^reports$/i], ["#/about", /about verif-ai/i]] as const) {
    await page.goto(`/${hash}`);
    await expect(page.getByRole("heading", { level: 1 })).toContainText(heading);
    await page.reload();
    await expect(page.getByRole("heading", { level: 1 })).toContainText(heading);
    expect(new URL(page.url()).hash).toBe(hash);
  }
  await page.goto(`/#/analysis/${"0".repeat(32)}`);
  await page.reload();
  await expect(page.getByRole("alert")).toContainText(/no longer exists|deleted|expired/i);   // an unknown id after a refresh: a message, not a blank page
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

  // heat-map viewer on the Overview: original crop + Grad-CAM side by side, both really decoded
  const viewer = page.getByTestId("face-heatmap-viewer");
  await expect(viewer).toBeVisible();
  if (!(await gradCamOn(request))) {
    // Grad-CAM is off on this server: the viewer must say so, show only the real face crop and invent nothing
    await expect(viewer.getByTestId("heatmap-unavailable")).toBeVisible();
    await expect(viewer.getByAltText(/Grad-CAM heat-map/)).toHaveCount(0);
    await expect(viewer.getByAltText(/original face crop/)).toBeVisible();
    await shot(page, "03b_heatmap_unavailable_honest");
  } else {
  const decoded = (alt: RegExp) => viewer.getByAltText(alt).evaluate((i: HTMLImageElement) => i.complete && i.naturalWidth > 0);
  await expect(viewer.getByAltText(/Face #1 original face crop/)).toBeVisible();
  await expect(viewer.getByAltText(/Face #1 Grad-CAM heat-map/)).toBeVisible();
  expect(await decoded(/Face #1 original face crop/), "original crop decodes").toBe(true);
  expect(await decoded(/Face #1 Grad-CAM heat-map/), "Grad-CAM decodes").toBe(true);
  await shot(page, "03b_heatmap_side_by_side");
  await viewer.getByRole("button", { name: "Original" }).click();
  await expect(viewer.getByAltText(/Grad-CAM heat-map/)).toHaveCount(0);
  await viewer.getByRole("button", { name: "Heatmap" }).click();
  await expect(viewer.getByAltText(/original face crop/)).toHaveCount(0);
  await viewer.getByRole("button", { name: "Overlay" }).click();
  await expect(viewer.getByTestId("overlay-stack")).toBeVisible();
  const slider = viewer.getByRole("slider", { name: /heat-map strength/i });
  await slider.fill("25");
  expect(await viewer.getByTestId("overlay-top").evaluate((e) => getComputedStyle(e).opacity)).toBe("0.25");
  await shot(page, "03c_heatmap_overlay");
  await expect(viewer.getByText(/does not independently establish that anything was manipulated/)).toBeVisible();
  }

  await page.getByRole("tab", { name: /forensic evidence/i }).click();
  await expect(page.getByAltText(/image with face bounding boxes/i)).toBeVisible();
  await expect(page.getByRole("table")).toBeVisible();
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
  expect((await request.get(`${API}/api/analyses/${id}`)).status()).toBe(404);
  expect((await request.get(`${API}/api/analyses/${id}/report.pdf`)).status()).toBe(404);
});

test("model selection: the production default is preselected; the E4 candidate is labelled and recorded in the result", async ({ page, request }) => {
  test.skip(!PUBLIC_IMAGE, "VERIFAI_E2E_IMAGE not set");
  const list = await (await request.get(`${API}/api/models`)).json();
  test.skip(!list.selection_enabled || list.models.filter((m: { selectable: boolean }) => m.selectable).length < 2, "this server offers only its default model");
  await page.goto("/#/");
  await page.locator("input[type=file]").setInputFiles(PUBLIC_IMAGE);
  const select = page.getByTestId("model-select");
  await expect(select).toBeVisible();
  await expect(select).toHaveValue(list.default);
  const cand = list.models.find((m: { role: string; selectable: boolean }) => m.role === "candidate" && m.selectable);
  test.skip(!cand, "no candidate model on this server");
  await select.selectOption(cand.id);
  await expect(page.getByTestId("candidate-note")).toContainText(/not approved for production/i);
  await page.getByRole("button", { name: /analyse image/i }).click();
  await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 180_000 });
  const id = page.url().split("/").pop()!;
  const res = await (await request.get(`${API}/api/analyses/${id}`)).json();
  expect(res.model.registry_id).toBe(cand.id);
  expect(res.model.role).toBe("candidate");
  await expect(page.getByTestId("model-chip")).toContainText(/Candidate - not approved for production/i);
  // the default is unchanged by using a candidate
  expect((await (await request.get(`${API}/api/models`)).json()).default).toBe(list.default);
  await request.delete(`${API}/api/analyses/${id}`);
});

test("public video: unified upload, timeline, frame selection and synchronised heat-map, PDF, deletion", async ({ page, request }) => {
  test.skip(!PUBLIC_VIDEO, "VERIFAI_E2E_PUBLIC_VIDEO not set");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/#/");
  await page.locator("input[type=file]").setInputFiles(PUBLIC_VIDEO);          // the same single card; the type is detected from the file
  await expect(page.getByText("Video detected")).toBeVisible();
  await expect(page.getByLabel("Selected video preview")).toBeVisible();
  await page.getByRole("button", { name: /^analyse video$/i }).click();
  await expect(page.getByTestId("progress")).toBeVisible();
  await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 5 * 60_000 });
  const id = page.url().split("/").pop()!;
  await expect(page.getByTestId("verdict").first()).toHaveText(/REAL|FAKE|INCONCLUSIVE/);

  // Overview already shows the stored frame next to its own Grad-CAM
  const viewer = page.getByTestId("frame-evidence-viewer");
  await expect(viewer).toBeVisible();
  const camOn = await gradCamOn(request);
  await expect(viewer.getByAltText(/Evidence frame at/)).toBeVisible();
  if (camOn) await expect(viewer.getByAltText(/Grad-CAM heat-map at/)).toBeVisible();
  else await expect(viewer.getByTestId("heatmap-unavailable")).toBeVisible();   // honest: frame stored, heat-map not generated on this server
  await shot(page, "08_video_overview_heatmap");

  await page.getByRole("tab", { name: /frame analysis/i }).click();
  const v2 = page.getByTestId("frame-evidence-viewer");
  const pos = async () => (await v2.getByTestId("frame-position").textContent())!.replace(/\s+/g, " ");
  const first = await pos();
  // every analysed frame is a clickable dot; the viewer must follow it and never show a stale heat-map
  const dots = page.locator("[data-testid^=tl-dot-]");
  const n = await dots.count();
  expect(n).toBeGreaterThan(1);
  let sawStored = 0, sawMissing = 0;
  for (let i = 0; i < n; i++) {
    await dots.nth(i).click();
    const label = await pos();
    const frameNo = /Frame (\d+)/.exec(label)![1];
    expect(await dots.nth(i).getAttribute("data-testid")).toBe(`tl-dot-${frameNo}`);       // the selected dot and the viewer agree
    const stored = await v2.getByAltText(/Evidence frame at/).count();
    const cam = await v2.getByAltText(/Grad-CAM heat-map at/).count();
    if (stored) { sawStored++; expect(cam, camOn ? "stored frame shows its heat-map" : "no heat-map when Grad-CAM is off").toBe(camOn ? 1 : 0); }
    else { sawMissing++; expect(cam, "no heat-map without stored evidence").toBe(0); await expect(v2.getByTestId("heatmap-unavailable")).toBeVisible(); }
  }
  expect(sawStored).toBeGreaterThan(0);
  expect(first).toMatch(/Frame \d+/);
  await v2.getByRole("button", { name: /next analysed frame/i }).isEnabled();
  await shot(page, "09_video_frame_analysis");
  if (camOn) await expect(v2.getByRole("button", { name: "Overlay" })).toBeDisabled();
  console.log(`video frames: ${n} analysed, ${sawStored} with stored heat-map, ${sawMissing} without`);

  // the viewer must stay usable on phones: no horizontal scroll at 390 px and at 320 px, controls still reachable
  for (const [w, h, name] of [[390, 844, "10_video_mobile"], [320, 640, "11_video_narrow"]] as const) {
    await page.setViewportSize({ width: w, height: h });
    await noHScroll(page);
    await expect(v2.getByRole("button", { name: /next analysed frame/i })).toBeVisible();
    await expect(v2.getByRole("button", { name: "Heatmap" })).toBeVisible();
    await shot(page, name);
  }
  await page.setViewportSize({ width: 1440, height: 900 });

  const pdf = await request.get(`${API}/api/analyses/${id}/report.pdf`);
  expect(pdf.status()).toBe(200);
  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels");
  expect((await request.get(`${API}/api/analyses/${id}`)).status()).toBe(404);
});

test("responsive layouts: no horizontal scroll, navigation usable", async ({ page }) => {
  for (const [w, h] of [[1440, 900], [1024, 768], [768, 1024], [390, 844], [320, 640]] as const) {
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
  expect((await request.get(`${API}/api/analyses/${id}/report.pdf`)).status()).toBe(200);
  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels"); // the redirect happens after the DELETE succeeded
  expect((await request.get(`${API}/api/analyses/${id}`)).status()).toBe(404);
});

test("private video: background job, timeline, tracks, evidence, PDF and deletion", async ({ page, request }) => {
  test.skip(!PRIVATE_VIDEO, "VERIFAI_E2E_PRIVATE_VIDEO not set");
  await page.goto("/#/");
  await page.locator("input[type=file]").setInputFiles(PRIVATE_VIDEO);
  await expect(page.getByText("Video detected")).toBeVisible();
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
  const pdf = await request.get(`${API}/api/analyses/${id}/report.pdf`);
  expect(pdf.status()).toBe(200);
  expect((await pdf.body()).subarray(0, 5).toString()).toBe("%PDF-");
  await page.getByRole("button", { name: /delete results/i }).click();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("See Beyond the Pixels"); // the redirect happens after the DELETE succeeded
  expect((await request.get(`${API}/api/analyses/${id}`)).status()).toBe(404);
});
