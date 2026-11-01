import { defineConfig } from "@playwright/test";

// E2E runs against an already running Verif-AI server (the backend serving the built frontend).
//   VERIFAI_E2E_URL             default http://127.0.0.1:8000
//   VERIFAI_E2E_IMAGE           public test image (screenshots are only ever taken with this one)
//   VERIFAI_E2E_PRIVATE_IMAGE   optional local personal image: workflow only, no screenshots, nothing logged
//   VERIFAI_E2E_PRIVATE_VIDEO   optional local personal video: same rules
//   VERIFAI_E2E_SHOTS           directory for screenshots (default ./e2e-shots, git-ignored)
export default defineConfig({
  testDir: "./e2e",
  timeout: 15 * 60_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: process.env.VERIFAI_E2E_URL ?? "http://127.0.0.1:8000", channel: "chrome", headless: true, screenshot: "off", video: "off", trace: "off" },
});
