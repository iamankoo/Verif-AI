import { describe, expect, it } from "vitest";
import { fmtBytes, fmtTime, validateFile } from "./format";

const f = (name: string, size = 1000) => new File([new Uint8Array(size)], name);

describe("validateFile", () => {
  it("accepts supported images and videos", () => {
    expect(validateFile(f("a.JPG")).kind).toBe("image");
    expect(validateFile(f("clip.mp4")).kind).toBe("video");
    expect(validateFile(f("clip.mov")).kind).toBe("video");
  });
  it("rejects unsupported types and empty files", () => {
    expect(validateFile(f("a.exe")).error).toMatch(/Unsupported/);
    expect(validateFile(f("noext")).error).toMatch(/Unsupported/);
    expect(validateFile(f("a.png", 0)).error).toMatch(/empty/);
  });
  it("enforces per-kind size limits", () => {
    const limits = { max_image_mb: 1, max_video_mb: 2, image_formats: [".png"], video_formats: [".mp4"] };
    expect(validateFile(f("a.png", 1_100_000), limits).error).toMatch(/limit for images is 1 MB/);
    expect(validateFile(f("a.mp4", 1_100_000), limits).kind).toBe("video");
    expect(validateFile(f("a.mp4", 2_100_000), limits).error).toMatch(/videos/);
  });
});

describe("formatters", () => {
  it("formats bytes and time", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(2048)).toBe("2.0 KB");
    expect(fmtBytes(5 * 1024 ** 2)).toBe("5.0 MB");
    expect(fmtTime(4.25)).toBe("4.3s");
    expect(fmtTime(65.5)).toBe("1:05.5");
  });
});
