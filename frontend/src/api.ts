import type { AnalysisResult, AnalysisSummary, Health, JobState, ModelResponse } from "./types";

/** Backend origin for production builds, e.g. https://verifai-api.onrender.com (no trailing slash needed). Empty = same origin (dev proxy / single-container deployment).
 *  VITE_API_BASE_URL is the documented name; VITE_API_BASE is still read for older setups. These values are public (they are baked into the bundle): never put secrets here. */
export function apiBase(env: Record<string, unknown> = import.meta.env): string {
  const raw = String(env.VITE_API_BASE_URL ?? env.VITE_API_BASE ?? "").trim();
  return `${raw.replace(/\/+$/, "")}/api`;
}
const BASE = apiBase();

export class ApiError extends Error {
  status: number;
  code: string;
  constructor(message: string, status: number, code = "error") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** Turns an error response body (FastAPI `detail` string or validation list) into a readable message. */
export function parseErrorBody(body: unknown, status: number): { message: string; code: string } {
  const b = body as { detail?: unknown; code?: string } | null;
  let message = `Request failed (${status}).`;
  if (b && typeof b.detail === "string") message = b.detail;
  else if (b && Array.isArray(b.detail) && b.detail.length) message = "The request was not valid (a file is required).";
  return { message, code: b?.code ?? "error" };
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    const { message, code } = parseErrorBody(body, res.status);
    throw new ApiError(message, res.status, code);
  }
  return (await res.json()) as T;
}

/** XHR upload so that large videos can report upload progress. */
function upload<T>(path: string, file: File, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError("Could not reach the Verif-AI server.", 0, "network"));
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as T);
      else {
        const { message, code } = parseErrorBody(xhr.response, xhr.status);
        reject(new ApiError(message, xhr.status, code));
      }
    };
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
}

export const api = {
  health: () => fetch(`${BASE}/health`).then((r) => json<Health>(r)),
  model: () => fetch(`${BASE}/model`).then((r) => json<ModelResponse>(r)),
  analyzeImage: (file: File, onProgress?: (f: number) => void) => upload<AnalysisResult>("/analyze/image", file, onProgress),
  analyzeVideo: (file: File, onProgress?: (f: number) => void) => upload<JobState>("/analyze/video", file, onProgress),
  status: (id: string) => fetch(`${BASE}/analyses/${id}/status`).then((r) => json<JobState>(r)),
  result: (id: string) => fetch(`${BASE}/analyses/${id}`).then((r) => json<AnalysisResult>(r)),
  list: () => fetch(`${BASE}/analyses?limit=20`).then((r) => json<AnalysisSummary[]>(r)),
  remove: (id: string) =>
    fetch(`${BASE}/analyses/${id}`, { method: "DELETE" }).then((r) => {
      if (!r.ok) throw new ApiError("Could not delete the analysis.", r.status);
    }),
  reportUrl: (id: string) => `${BASE}/analyses/${id}/report.pdf`,
  artifactUrl: (id: string, name: string) => `${BASE}/analyses/${id}/artifacts/${name}`,
};
