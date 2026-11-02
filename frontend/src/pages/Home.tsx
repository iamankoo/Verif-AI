import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { IconImage, IconVideo } from "../components/Icons";
import { ErrorBanner, HistoryList, ModelCard, ProgressPanel } from "../components/Panels";
import { UploadZone } from "../components/UploadZone";
import { fmtBytes, type Kind } from "../lib/format";
import { loadRecent, remember } from "../lib/history";
import type { AnalysisSummary, Health, ModelList, ModelResponse } from "../types";

interface Selected {
  file: File;
  kind: Kind;
  url: string;
}

export function Home({ onOpen }: { onOpen: (id: string) => void }) {
  const [info, setInfo] = useState<ModelResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [models, setModels] = useState<ModelList | null>(null);
  const [modelId, setModelId] = useState<string>("");
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [sel, setSel] = useState<Selected | null>(null);
  const [busy, setBusy] = useState<{ phase: "uploading" | "analysing"; progress: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.health().then((h) => alive && setHealth(h)).catch(() => alive && setError("Cannot reach the Verif-AI server. Is the backend running?"));
    api.model().then((m) => alive && setInfo(m)).catch(() => undefined);
    // Older servers have no /models endpoint: then no selector is shown and the server default is used.
    Promise.resolve().then(() => api.models()).then((l) => { if (alive && l) { setModels(l); setModelId(l.default); } }).catch(() => undefined);
    loadRecent().then((l) => alive && setHistory(l)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => () => { if (sel) URL.revokeObjectURL(sel.url); }, [sel]);

  const choose = useCallback((file: File, kind: Kind) => {
    setError(null);
    setSel({ file, kind, url: URL.createObjectURL(file) });
  }, []);

  const analyse = async () => {
    if (!sel) return;
    setError(null);
    setBusy({ phase: "uploading", progress: 0 });
    try {
      // Only send a model id when the user picked something other than the server default.
      const chosenModel = models && modelId && modelId !== models.default ? modelId : undefined;
      const onProgress = (f: number) => setBusy({ phase: f >= 1 ? "analysing" : "uploading", progress: f });
      if (sel.kind === "image") {
        const r = await api.analyzeImage(sel.file, onProgress, chosenModel);
        remember(r.analysis_id);
        onOpen(r.analysis_id);
      } else {
        const j = await api.analyzeVideo(sel.file, onProgress, chosenModel);
        remember(j.analysis_id);
        onOpen(j.analysis_id);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The analysis failed unexpectedly.");
      setBusy(null);
    }
  };

  const modelReady = health ? health.model_loaded : true;
  const selectable = models?.selection_enabled ? models.models.filter((o) => o.selectable) : [];
  const picked = selectable.find((o) => o.id === modelId);

  return (
    <div className="space-y-8 pb-4">
      <section className="mx-auto max-w-6xl pt-2 text-center">
        <h1 className="text-3xl font-extrabold leading-tight sm:text-4xl lg:text-[2.6rem]">
          See Beyond the Pixels. <span className="text-accent">Verify What&apos;s Real.</span>
        </h1>
        <p className="mx-auto mt-4 max-w-3xl text-base text-ink-soft sm:text-lg">
          Analyze images and videos with AI-powered facial forensics. Detect suspicious manipulation, inspect visual evidence and generate detailed forensic reports.
        </p>
      </section>

      <section className="mx-auto w-full max-w-5xl" aria-label="Upload">
        {error && <div className="mb-4"><ErrorBanner message={error} onClose={() => setError(null)} /></div>}
        <div className="card-glow p-4 sm:p-6">
          {busy ? (
            <ProgressPanel
              stage={busy.phase}
              progress={busy.phase === "uploading" ? busy.progress : 1}
              label={busy.phase === "uploading" ? `Uploading ${sel?.file.name}` : "Analysing - this can take a while for videos..."}
            />
          ) : sel ? (
            <section data-testid="preview">
              <div className="flex flex-wrap items-start gap-5">
                <div className="w-full max-w-sm overflow-hidden rounded-xl border border-paper-line bg-paper">
                  {sel.kind === "image" ? (
                    <img src={sel.url} alt="Selected file preview" className="max-h-72 w-full object-contain" />
                  ) : (
                    <video src={sel.url} controls muted className="max-h-72 w-full" aria-label="Selected video preview" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <p className="flex items-center gap-2 text-sm text-accent">{sel.kind === "image" ? <IconImage className="h-4 w-4" /> : <IconVideo className="h-4 w-4" />} {sel.kind === "image" ? "Image detected" : "Video detected"}</p>
                  <p className="mt-1 break-all text-lg font-semibold">{sel.file.name}</p>
                  <p className="text-sm text-ink-mute">{sel.kind} &middot; {fmtBytes(sel.file.size)}</p>
                  {selectable.length > 1 && (
                    <div className="mt-4" data-testid="model-picker">
                      <label htmlFor="model-select" className="block text-sm font-medium">Model</label>
                      <select id="model-select" data-testid="model-select" className="mt-1 w-full max-w-md rounded-lg border border-paper-line bg-paper px-3 py-2 text-sm"
                        value={modelId} onChange={(e) => setModelId(e.target.value)}>
                        {selectable.map((o) => <option key={o.id} value={o.id}>{o.label} - {o.role_label}</option>)}
                      </select>
                      {picked && picked.role === "candidate" && (
                        <p className="mt-2 text-xs text-ink-soft" data-testid="candidate-note">Candidate model for evaluation only. It is not approved for production: it failed the project's promotion rule on DeepFakeFace detection. Results are labelled with this model.</p>
                      )}
                    </div>
                  )}
                  <div className="mt-4 flex flex-wrap gap-2">
                    <button className="btn-primary" onClick={analyse} disabled={!modelReady}>Analyse {sel.kind}</button>
                    <button className="btn-secondary" onClick={() => setSel(null)}>Remove</button>
                  </div>
                  <p className="mt-3 text-xs text-ink-mute">Uploaded files are deleted right after analysis; only derived evidence images and results are kept, for {info?.limits.retention_hours ?? 72} hours.</p>
                </div>
              </div>
            </section>
          ) : (
            <UploadZone limits={info?.limits} disabled={!modelReady} onSelect={choose} onError={setError} />
          )}
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2" aria-label="Model and history">
        <ModelCard info={info} health={health} compact />
        <HistoryList items={history} onOpen={onOpen} limit={5} viewAllHref="#/reports" />
      </section>
    </div>
  );
}
