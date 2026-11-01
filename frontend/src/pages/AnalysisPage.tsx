import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import { ErrorBanner, ProgressPanel } from "../components/Panels";
import { ResultView } from "../components/ResultView";
import type { AnalysisResult, JobState } from "../types";

export function AnalysisPage({ id, onHome, pollMs = 1000 }: { id: string; onHome: () => void; pollMs?: number }) {
  const [job, setJob] = useState<JobState | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setResult(null);
    setError(null);
    const tick = async () => {
      try {
        const st = await api.status(id);
        if (!alive) return;
        setJob(st);
        if (st.status === "completed") {
          const r = await api.result(id);
          if (alive) setResult(r);
        } else if (st.status === "failed") {
          setError(st.error ?? "The analysis failed.");
        } else {
          timer = setTimeout(tick, pollMs);
        }
      } catch (e) {
        if (alive) setError(e instanceof ApiError && e.status === 404 ? "This analysis no longer exists (it may have expired or been deleted)." : "Lost contact with the server.");
      }
    };
    tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [id, pollMs]);

  const remove = async () => {
    try {
      await api.remove(id);
      onHome();
    } catch {
      setError("Could not delete this analysis.");
    }
  };

  return (
    <div className="space-y-5">
      <button className="btn-secondary" onClick={onHome}>&larr; Back</button>
      {error && <ErrorBanner message={error} />}
      {!result && !error && <ProgressPanel stage={job?.stage ?? "queued"} progress={job?.progress ?? 0} label={`Analysing ${job?.filename ?? "media"}...`} />}
      {result && <ResultView result={result} onNew={onHome} onDelete={remove} />}
    </div>
  );
}
