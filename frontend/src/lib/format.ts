import type { Classification, Limits } from "../types";

export const DEFAULT_LIMITS: Pick<Limits, "max_image_mb" | "max_video_mb" | "image_formats" | "video_formats"> = {
  max_image_mb: 15,
  max_video_mb: 200,
  image_formats: [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
  video_formats: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"],
};

export const fmt3 = (n: number | null | undefined) => (n == null ? "n/a" : n.toFixed(3));
export const pct = (n: number | null | undefined) => (n == null ? "n/a" : `${(n * 100).toFixed(1)}%`);

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return m > 0 ? `${m}:${r.toFixed(1).padStart(4, "0")}` : `${r.toFixed(1)}s`;
}

export const VERDICT_TEXT: Record<Classification, { title: string; blurb: string }> = {
  REAL: { title: "Real", blurb: "The model found no patterns it associates with fabricated faces. This does not prove authenticity." },
  FAKE: { title: "Fake", blurb: "The model found patterns it associates with fabricated faces. This is model-based evidence, not proof." },
  INCONCLUSIVE: { title: "Inconclusive", blurb: "There is not enough usable evidence for a reliable verdict. This does not mean real or fake." },
};

export type Kind = "image" | "video";

/** Client-side pre-check. The server re-validates everything; this only gives fast feedback. */
export function validateFile(
  file: File,
  limits: Pick<Limits, "max_image_mb" | "max_video_mb" | "image_formats" | "video_formats"> = DEFAULT_LIMITS,
): { kind?: Kind; error?: string } {
  const ext = "." + (file.name.split(".").pop() ?? "").toLowerCase();
  const kind: Kind | undefined = limits.image_formats.includes(ext) ? "image" : limits.video_formats.includes(ext) ? "video" : undefined;
  if (!kind) return { error: `Unsupported file type "${ext}". Use an image (${limits.image_formats.join(", ")}) or a video (${limits.video_formats.join(", ")}).` };
  if (file.size === 0) return { error: "The selected file is empty." };
  const max = (kind === "image" ? limits.max_image_mb : limits.max_video_mb) * 1e6;
  if (file.size > max) return { error: `The file is ${fmtBytes(file.size)}; the limit for ${kind}s is ${kind === "image" ? limits.max_image_mb : limits.max_video_mb} MB.` };
  return { kind };
}

/** Short display name for a registry id (the full label is shown in the model card). */
export function shortModelName(id?: string | null, fallback = ""): string {
  const map: Record<string, string> = { e4: "E4", model4: "Model 4", v2: "v2" };
  return (id && map[id]) || id || fallback;
}
