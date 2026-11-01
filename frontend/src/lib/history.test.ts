import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";

vi.mock("../api", async (orig) => {
  const real = await orig<typeof import("../api")>();
  return { ...real, api: { list: vi.fn(), result: vi.fn(), status: vi.fn() } };
});
import { api } from "../api";
import { forget, loadRecent, remember, rememberedIds, summaryFor } from "./history";
const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

const id = (c: string) => c.repeat(32);
const res = (c: string, t: string) => ({ analysis_id: id(c), media_type: "image", filename: `${c}.png`, status: "completed", classification: "REAL", fake_probability: 0.1, created_at: t });

beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
  m.list.mockResolvedValue([]);
});

describe("per-browser history", () => {
  it("remembers ids newest first, de-duplicates, forgets, and ignores junk", () => {
    remember(id("a")); remember(id("b")); remember(id("a"));
    expect(rememberedIds()).toEqual([id("a"), id("b")]);
    forget(id("a"));
    expect(rememberedIds()).toEqual([id("b")]);
    localStorage.setItem("verifai.history.v1", JSON.stringify([id("c"), "not-an-id", 5]));
    expect(rememberedIds()).toEqual([id("c")]);
    localStorage.setItem("verifai.history.v1", "{broken");
    expect(rememberedIds()).toEqual([]);
  });

  it("builds the list from this browser's own ids when the server list is empty (public deployment)", async () => {
    remember(id("a")); remember(id("b"));
    m.result.mockImplementation(async (x: string) => (x === id("a") ? res("a", "2026-11-01T10:00:00Z") : res("b", "2026-11-02T10:00:00Z")));
    const rows = await loadRecent();
    expect(rows.map((r) => r.analysis_id)).toEqual([id("b"), id("a")]);            // newest first
    expect(m.list).toHaveBeenCalledTimes(1);
  });

  it("drops deleted or expired ids (404) from the stored list, and keeps running jobs via the status endpoint", async () => {
    remember(id("a")); remember(id("b"));
    m.result.mockImplementation(async (x: string) => { if (x === id("a")) throw new ApiError("gone", 404); throw new ApiError("not ready", 409); });
    m.status.mockResolvedValue({ analysis_id: id("b"), status: "running", media_type: "video", filename: "v.mp4", progress: 0.3, stage: "analysing", created_at: "2026-11-02T10:00:00Z" });
    const rows = await loadRecent();
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ analysis_id: id("b"), status: "running", classification: null });
    expect(rememberedIds()).toEqual([id("b")]);
  });

  it("uses the server list when it is enabled and does not refetch entries it already has", async () => {
    remember(id("a"));
    m.list.mockResolvedValue([res("a", "2026-11-01T10:00:00Z")]);
    const rows = await loadRecent();
    expect(rows).toHaveLength(1);
    expect(m.result).not.toHaveBeenCalled();
  });

  it("summaryFor returns null for unknown ids", async () => {
    m.result.mockRejectedValue(new ApiError("gone", 404));
    expect(await summaryFor(id("z"))).toBeNull();
  });
});
