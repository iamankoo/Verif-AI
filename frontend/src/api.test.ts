import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, parseErrorBody } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("parseErrorBody", () => {
  it("uses the server detail string and code", () => {
    expect(parseErrorBody({ detail: "Too big", code: "payload_too_large" }, 413)).toEqual({ message: "Too big", code: "payload_too_large" });
  });
  it("handles validation-error arrays and empty bodies", () => {
    expect(parseErrorBody({ detail: [{ loc: ["body", "file"] }] }, 422).message).toMatch(/file is required/);
    expect(parseErrorBody(null, 500).message).toBe("Request failed (500).");
  });
});

describe("api helpers", () => {
  it("builds report and artifact URLs", () => {
    expect(api.reportUrl("abc")).toBe("/api/analyses/abc/report.pdf");
    expect(api.artifactUrl("abc", "annotated.jpg")).toBe("/api/analyses/abc/artifacts/annotated.jpg");
  });

  it("throws ApiError with the server message on HTTP errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 409, json: async () => ({ detail: "Analysis is running.", code: "not_ready" }) }));
    await expect(api.result("x")).rejects.toMatchObject({ name: "ApiError", status: 409, code: "not_ready", message: "Analysis is running." });
  });

  it("survives a non-JSON error body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 502, json: async () => { throw new Error("html"); } }));
    await expect(api.health()).rejects.toBeInstanceOf(ApiError);
  });

  it("returns parsed JSON on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ status: "ok", model_loaded: true }) }));
    await expect(api.health()).resolves.toMatchObject({ status: "ok" });
  });
});
