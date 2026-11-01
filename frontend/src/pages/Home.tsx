import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
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

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <div className="space-y-5 lg:col-span-3">
        <section className="card p-6">
          <h1 className="text-3xl">See beyond the pixels.</h1>
          <p className="mt-2 max-w-2xl text-ink-soft">
            Upload a photo or video of a person. Verif-AI detects faces, analyses them with a two-stream deep-learning model, highlights the evidence and produces a forensic report. It runs entirely on this server - nothing is sent to third-party AI services.
          </p>
        </section>

        {error && <ErrorBanner message={error} onClose={() => setError(null)} />}

        {busy ? (
          <ProgressPanel
            stage={busy.phase}
            progress={busy.phase === "uploading" ? busy.progress : 1}
            label={busy.phase === "uploading" ? `Uploading ${sel?.file.name}` : "Analysing - this can take a while for videos..."}
          />
        ) : (
          <>
            <UploadZone limits={info?.limits} disabled={!modelReady} onSelect={choose} onError={setError} />
            {sel && (
              <section className="card p-5" data-testid="preview">
                <div className="flex flex-wrap items-start gap-5">
                  <div className="w-full max-w-xs overflow-hidden rounded-lg bg-paper">
                    {sel.kind === "image" ? (
                      <img src={sel.url} alt="Selected file preview" className="max-h-64 w-full object-contain" />
                    ) : (
                      <video src={sel.url} controls muted className="max-h-64 w-full" aria-label="Selected video preview" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="break-all font-medium">{sel.file.name}</p>
                    <p className="text-sm text-ink-mute">{sel.kind} &middot; {fmtBytes(sel.file.size)}</p>
                    <div className="mt-4 flex gap-2">
                      <button className="btn-primary" onClick={analyse} disabled={!modelReady}>Analyse {sel.kind}</button>
                      <button className="btn-secondary" onClick={() => setSel(null)}>Remove</button>
                    </div>
                    <p className="mt-3 text-xs text-ink-mute">Uploaded files are deleted right after analysis; only derived evidence images and results are kept, for {info?.limits.retention_hours ?? 72} hours.</p>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </div>

      <aside className="space-y-5 lg:col-span-2">
        <ModelCard info={info} health={health} />
        <HistoryList items={history} onOpen={onOpen} />
      </aside>
    </div>
  );
}
