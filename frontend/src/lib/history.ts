import { api, ApiError } from "../api";
import type { AnalysisSummary } from "../types";

/** Per-browser history. A public deployment switches the server-wide list off (VERIFAI_PUBLIC_HISTORY=false), so each browser keeps only the ids of
 *  its own analyses and asks the server for those one by one. Ids are unguessable; nothing here lists other visitors' analyses. */
const KEY = "verifai.history.v1";
const MAX = 30;

const read = (): string[] => {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string" && /^[0-9a-f]{32}$/.test(x)) : [];
  } catch {
    return [];
  }
};
const write = (ids: string[]) => {
  try {
    localStorage.setItem(KEY, JSON.stringify(ids.slice(0, MAX)));
  } catch {
    /* storage unavailable (private window): history simply is not remembered */
  }
};

export const remember = (id: string) => write([id, ...read().filter((x) => x !== id)]);
export const forget = (id: string) => write(read().filter((x) => x !== id));
export const rememberedIds = read;

/** One entry for the history table, built from the real result (or the job status while a video is still running). null = gone (deleted / expired). */
export async function summaryFor(id: string): Promise<AnalysisSummary | null> {
  try {
    const r = await api.result(id);
    return { analysis_id: r.analysis_id, media_type: r.media_type, filename: r.filename, status: r.status, classification: r.classification, fake_probability: r.fake_probability, created_at: r.created_at };
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    try {
      const j = await api.status(id);
      return { analysis_id: j.analysis_id, media_type: j.media_type, filename: j.filename, status: j.status, classification: null, fake_probability: null, created_at: j.created_at };
    } catch (e2) {
      if (e2 instanceof ApiError && e2.status === 404) return null;
      return null;
    }
  }
}

/** Server list (empty on public deployments) merged with this browser's own analyses, newest first. Expired or deleted ids are dropped from the local list. */
export async function loadRecent(): Promise<AnalysisSummary[]> {
  const server = await api.list().catch(() => [] as AnalysisSummary[]);
  const have = new Set(server.map((s) => s.analysis_id));
  const local = read().filter((id) => !have.has(id));
  const got = await Promise.all(local.map(async (id) => ({ id, s: await summaryFor(id) })));
  for (const g of got) if (g.s === null) forget(g.id);
  return [...server, ...got.flatMap((g) => (g.s ? [g.s] : []))].sort((a, b) => b.created_at.localeCompare(a.created_at));
}
