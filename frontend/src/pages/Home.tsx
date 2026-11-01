import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { IconImage, IconVideo } from "../components/Icons";
import { ErrorBanner, HistoryList, ModelCard, ProgressPanel } from "../components/Panels";
import { UploadZone } from "../components/UploadZone";
import { fmtBytes, type Kind } from "../lib/format";
import type { AnalysisSummary, Health, ModelResponse } from "../types";

interface Selected {
  file: File;
  kind: Kind;
  url: string;
}

export function Home({ onOpen }: { onOpen: (id: string) => void }) {
  const [info, setInfo] = useState<ModelResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [sel, setSel] = useState<Selected | null>(null);
  const [mode, setMode] = useState<Kind>("image");
  const [busy, setBusy] = useState<{ phase: "uploading" | "analysing"; progress: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    api.health().then((h) => alive && setHealth(h)).catch(() => alive && setError("Cannot reach the Verif-AI server. Is the backend running?"));
    api.model().then((m) => alive && setInfo(m)).catch(() => undefined);
    api.list().then((l) => alive && setHistory(l)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => () => { if (sel) URL.revokeObjectURL(sel.url); }, [sel]);

  const choose = useCallback((file: File, kind: Kind) => {
    setError(null);
    setMode(kind);                                   // a file of the other kind switches the visible tab
    setSel({ file, kind, url: URL.createObjectURL(file) });
  }, []);

  const analyse = async () => {
    if (!sel) return;
    setError(null);
    setBusy({ phase: "uploading", progress: 0 });
    try {
      const onProgress = (f: number) => setBusy({ phase: f >= 1 ? "analysing" : "uploading", progress: f });
      if (sel.kind === "image") {
        const r = await api.analyzeImage(sel.file, onProgress);
        onOpen(r.analysis_id);
      } else {
        const j = await api.analyzeVideo(sel.file, onProgress);
        onOpen(j.analysis_id);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The analysis failed unexpectedly.");
      setBusy(null);
    }
  };

  const modelReady = health ? health.model_loaded : true;
  const switchTab = (k: Kind) => {
    setMode(k);
    if (sel && sel.kind !== k) setSel(null);          // a preview of the other kind is dropped when the tab changes
  };

  return (
    <div className="space-y-10 pb-4">
      <section className="mx-auto max-w-3xl text-center">
        <h1 className="text-3xl font-bold leading-tight sm:text-4xl">See Beyond the Pixels. Verify What&apos;s Real.</h1>
        <p className="mx-auto mt-3 max-w-2xl text-ink-soft">
          Upload a photo or video of a person. Verif-AI finds the faces, scores them with a two-stream deep-learning model, shows the evidence and gives you a PDF report. Everything runs on this computer.
        </p>
      </section>

      <section className="mx-auto w-full max-w-3xl" aria-label="Upload">
        {error && <div className="mb-4"><ErrorBanner message={error} onClose={() => setError(null)} /></div>}
        <div className="card p-4 sm:p-6">
          <div role="tablist" aria-label="Media type" className="mb-5 grid grid-cols-2 gap-2 rounded-lg border border-paper-line bg-paper-raised p-1">
            {([["image", "Upload Image", IconImage], ["video", "Upload Video", IconVideo]] as const).map(([k, label, Icon]) => (
              <button
                key={k}
                role="tab"
                id={`tab-${k}`}
                aria-selected={mode === k}
                aria-controls="upload-panel"
                disabled={!!busy}
                onClick={() => switchTab(k)}
                className={`flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-semibold transition ${mode === k ? "bg-white text-accent shadow-sm" : "text-ink-mute hover:text-ink"}`}
              >
                <Icon className="h-4 w-4" /> {label}
              </button>
            ))}
          </div>
          <div id="upload-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
            {busy ? (
              <ProgressPanel
                stage={busy.phase}
                progress={busy.phase === "uploading" ? busy.progress : 1}
                label={busy.phase === "uploading" ? `Uploading ${sel?.file.name}` : "Analysing - this can take a while for videos..."}
              />
            ) : sel ? (
              <section data-testid="preview">
                <div className="flex flex-wrap items-start gap-5">
                  <div className="w-full max-w-xs overflow-hidden rounded-xl bg-paper">
                    {sel.kind === "image" ? (
                      <img src={sel.url} alt="Selected file preview" className="max-h-64 w-full object-contain" />
                    ) : (
                      <video src={sel.url} controls muted className="max-h-64 w-full" aria-label="Selected video preview" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="break-all font-semibold">{sel.file.name}</p>
                    <p className="text-sm text-ink-mute">{sel.kind} &middot; {fmtBytes(sel.file.size)}</p>
                    <div className="mt-4 flex flex-wrap gap-2">
                      <button className="btn-primary" onClick={analyse} disabled={!modelReady}>Analyse {sel.kind}</button>
                      <button className="btn-secondary" onClick={() => setSel(null)}>Remove</button>
                    </div>
                    <p className="mt-3 text-xs text-ink-mute">Uploaded files are deleted right after analysis; only derived evidence images and results are kept, for {info?.limits.retention_hours ?? 72} hours.</p>
                  </div>
                </div>
              </section>
            ) : (
              <UploadZone limits={info?.limits} mode={mode} disabled={!modelReady} onSelect={choose} onError={setError} />
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2" aria-label="Model and history">
        <ModelCard info={info} health={health} />
        <HistoryList items={history} onOpen={onOpen} />
      </section>
    </div>
  );
}
