import type { AnalysisSummary, Health, ModelResponse } from "../types";
import { VerdictBadge } from "./Verdict";

export function ModelCard({ info, health }: { info: ModelResponse | null; health: Health | null }) {
  if (health && !health.model_loaded) {
    return (
      <div role="alert" className="card border-fake/30 bg-fake-soft p-5 text-sm text-fake">
        <p className="font-semibold">The analysis model is not loaded.</p>
        <p className="mt-1 break-words">{health.detail ?? "Check the server log and the VERIFAI_MODEL_PATH setting."}</p>
      </div>
    );
  }
  if (!info) return <div className="card p-5 text-sm text-ink-mute" aria-busy="true">Loading model information...</div>;
  const m = info.model;
  const rows: [string, string][] = [
    ["Model", `${m.name} v${m.version}`],
    ["Architecture", "Two-stream EfficientNet-B0 (RGB + SRM noise residual)"],
    ["Parameters", m.parameters.toLocaleString()],
    ["Device", m.device],
    ["Decision threshold", String(m.decision_threshold)],
    ["Inconclusive band", `+/- ${info.limits.uncertainty_margin} around 0.5`],
    ["Checkpoint", `${m.checkpoint_file} (${m.checkpoint_sha256.slice(0, 10)}...)`],
    ["Grad-CAM", info.explainability.grad_cam_available ? "available" : "unavailable"],
  ];
  return (
    <div className="card p-5" data-testid="model-card">
      <h2 className="mb-3 text-lg">Model information</h2>
      <dl className="space-y-1.5 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4"><dt className="text-ink-mute">{k}</dt><dd className="text-right font-medium">{v}</dd></div>
        ))}
      </dl>
      {m.trained_on && <p className="mt-3 text-xs text-ink-soft"><strong>Trained on:</strong> {m.trained_on}</p>}
      {m.metrics_note && <p className="mt-1 text-xs text-ink-soft">{m.metrics_note}</p>}
      <p className="mt-3 rounded-lg bg-warn-soft p-3 text-xs text-warn">{info.scope_note}</p>
    </div>
  );
}

export function HistoryList({ items, onOpen }: { items: AnalysisSummary[]; onOpen: (id: string) => void }) {
  return (
    <div className="card p-5" data-testid="history">
      <h2 className="mb-3 text-lg">Recent analyses</h2>
      {items.length === 0 ? (
        <p className="text-sm text-ink-mute">No analyses yet. Results are kept on this server for a limited time and can be deleted at any point.</p>
      ) : (
        <ul className="divide-y divide-paper-line">
          {items.map((a) => (
            <li key={a.analysis_id}>
              <button className="flex w-full items-center justify-between gap-3 py-2.5 text-left hover:bg-paper" onClick={() => onOpen(a.analysis_id)}>
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium">{a.filename}</span>
                  <span className="text-xs text-ink-mute">{a.media_type} &middot; {new Date(a.created_at).toLocaleString()}</span>
                </span>
                {a.classification ? <VerdictBadge value={a.classification} size="sm" /> : <span className="text-xs text-ink-mute">{a.status}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function ProgressPanel({ stage, progress, label }: { stage: string; progress: number; label: string }) {
  const pct = Math.round(progress * 100);
  return (
    <div className="card p-6" role="status" aria-live="polite" data-testid="progress">
      <p className="font-medium">{label}</p>
      <p className="mt-1 text-sm text-ink-mute">Stage: {stage}</p>
      <div className="mt-4 h-3 overflow-hidden rounded-full bg-paper" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
        <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-right text-xs tabular-nums text-ink-mute">{pct}%</p>
    </div>
  );
}

export function ErrorBanner({ message, onClose }: { message: string; onClose?: () => void }) {
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-xl border border-fake/30 bg-fake-soft p-4 text-sm text-fake">
      <span>{message}</span>
      {onClose && <button className="font-semibold underline" onClick={onClose}>Dismiss</button>}
    </div>
  );
}
