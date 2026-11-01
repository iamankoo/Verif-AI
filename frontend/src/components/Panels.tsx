import type { AnalysisSummary, Health, ModelResponse } from "../types";
import { IconActivity, IconArrowRight, IconChip, IconClock, IconDatabase, IconImage, IconLayers, IconStack, IconVideo } from "./Icons";
import { VerdictBadge } from "./Verdict";

function PanelTitle({ icon, title, right }: { icon: React.ReactNode; title: string; right?: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <h2 className="flex items-center gap-3 text-xl">
        <span className="text-accent">{icon}</span>
        {title}
      </h2>
      {right}
    </div>
  );
}

function Row({ icon, k, children }: { icon: React.ReactNode; k: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1 border-t border-paper-line py-2.5 first:border-t-0 sm:grid-cols-[9.5rem_minmax(0,1fr)] sm:items-center sm:gap-3">
      <dt className="flex items-center gap-2 text-sm font-medium text-ink-soft"><span className="text-accent">{icon}</span>{k}</dt>
      <dd className="min-w-0 break-words text-sm font-medium">{children}</dd>
    </div>
  );
}

export function ModelCard({ info, health, compact = false }: { info: ModelResponse | null; health: Health | null; compact?: boolean }) {
  if (health && !health.model_loaded) {
    return (
      <div role="alert" className="card border-fake/40 bg-fake-soft p-5 text-sm text-fake">
        <p className="font-semibold">The analysis model is not loaded.</p>
        <p className="mt-1 break-words">{health.detail ?? "Check the server log and the VERIFAI_MODEL_PATH setting."}</p>
      </div>
    );
  }
  if (!info) return <div className="card p-5 text-sm text-ink-mute" aria-busy="true">Loading model information...</div>;
  const m = info.model;
  const extra: [string, string][] = [
    ["Parameters", m.parameters.toLocaleString()],
    ["Device", m.device],
    ["Decision threshold", String(m.decision_threshold)],
    ["Inconclusive band", `+/- ${info.limits.uncertainty_margin} around 0.5`],
    ["Checkpoint", `${m.checkpoint_file} (${m.checkpoint_sha256.slice(0, 10)}...)`],
    ["Grad-CAM", info.explainability.grad_cam_available ? "available" : "unavailable"],
  ];
  return (
    <div className="card-glow min-w-0 p-5" data-testid="model-card">
      <PanelTitle icon={<IconChip className="h-7 w-7" />} title="Model Information" />
      <dl>
        <Row icon={<IconLayers className="h-5 w-5" />} k="Model">{m.name} v{m.version}</Row>
        <Row icon={<IconStack className="h-5 w-5" />} k="Architecture">Two-stream EfficientNet-B0 (RGB + SRM noise residual)</Row>
        <Row icon={<IconActivity className="h-5 w-5" />} k="Status">
          <span className="inline-flex items-center gap-2 rounded-full border border-real/40 bg-real-soft px-3 py-0.5 text-real"><span className="h-2 w-2 rounded-full bg-real" aria-hidden="true" /> {health?.model_loaded ? "Active" : "Loaded"}</span>
        </Row>
        {m.trained_on && <Row icon={<IconDatabase className="h-5 w-5" />} k="Trained On">{m.trained_on}</Row>}
        {!compact && extra.map(([k, v]) => <Row key={k} icon={<span className="block h-5 w-5" />} k={k}>{v}</Row>)}
      </dl>
      {!compact && m.metrics_note && <p className="mt-2 text-xs text-ink-soft">{m.metrics_note}</p>}
      <p className="mt-3 rounded-xl border border-warn/30 bg-warn-soft p-3 text-xs text-warn">{info.scope_note}</p>
    </div>
  );
}

const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString(undefined, { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
};

export function HistoryList({ items, onOpen, limit, viewAllHref }: { items: AnalysisSummary[]; onOpen: (id: string) => void; limit?: number; viewAllHref?: string }) {
  const shown = limit ? items.slice(0, limit) : items;
  return (
    <div className="card-glow min-w-0 p-5" data-testid="history">
      <PanelTitle
        icon={<IconClock className="h-7 w-7" />}
        title="Recent Analyses"
        right={viewAllHref ? <a href={viewAllHref} className="flex items-center gap-1 text-sm font-semibold text-accent hover:underline">View All <IconArrowRight className="h-4 w-4" /></a> : undefined}
      />
      {shown.length === 0 ? (
        <p className="text-sm text-ink-mute">No analyses yet. Results are kept on this server for a limited time and can be deleted at any point.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[30rem] text-left text-sm">
            <thead className="text-xs uppercase tracking-wide text-ink-mute">
              <tr><th className="w-8 pb-2 pr-2">#</th><th className="pb-2 pr-3">Filename</th><th className="pb-2 pr-3">Type</th><th className="pb-2 pr-3">Verdict</th><th className="pb-2">Date</th></tr>
            </thead>
            <tbody>
              {shown.map((a, i) => (
                <tr key={a.analysis_id} className="border-t border-paper-line">
                  <td className="py-2.5 pr-2 text-ink-mute">{i + 1}</td>
                  <td className="max-w-[11rem] py-2.5 pr-3"><button className="block max-w-full truncate text-left font-medium hover:text-accent" onClick={() => onOpen(a.analysis_id)}>{a.filename}</button></td>
                  <td className="py-2.5 pr-3"><span className="inline-flex items-center gap-1.5 text-ink-soft">{a.media_type === "video" ? <IconVideo className="h-4 w-4" /> : <IconImage className="h-4 w-4" />}{a.media_type === "video" ? "Video" : "Image"}</span></td>
                  <td className="py-2.5 pr-3">{a.classification ? <VerdictBadge value={a.classification} size="sm" /> : <span className="text-xs text-ink-mute">{a.status}</span>}</td>
                  <td className="whitespace-nowrap py-2.5 text-ink-soft">{fmtDate(a.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
    <div role="alert" className="flex items-start justify-between gap-3 rounded-xl border border-fake/40 bg-fake-soft p-4 text-sm text-fake">
      <span>{message}</span>
      {onClose && <button className="font-semibold underline" onClick={onClose}>Dismiss</button>}
    </div>
  );
}
