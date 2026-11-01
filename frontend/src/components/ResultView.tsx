import { api } from "../api";
import type { AnalysisResult } from "../types";
import { fmt3, fmtBytes, fmtTime } from "../lib/format";
import { ScoreBars, VerdictBadge, VerdictBlurb } from "./Verdict";
import { VideoTimeline } from "./VideoTimeline";

const LIMITATIONS = [
  "This is automated, model-based evidence - not definitive proof of manipulation or authenticity.",
  "Scores are uncalibrated neural-network outputs, not real-world probabilities.",
  "The model analyses faces only and is validated only on the data and generators listed in the model card; other generators, heavy compression, low resolution and adversarial edits can cause errors.",
  "Grad-CAM shows regions that influenced the RGB stream's fake score. It does not prove those pixels were altered and does not explain the noise-residual stream.",
  "A REAL result does not prove authenticity; a FAKE result does not say who or what created the media.",
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card p-5">
      <h3 className="mb-3 text-lg">{title}</h3>
      {children}
    </section>
  );
}

function Stat({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-ink-mute">{k}</dt>
      <dd className="font-medium tabular-nums">{v}</dd>
    </div>
  );
}

export function ResultView({ result, onNew, onDelete }: { result: AnalysisResult; onNew: () => void; onDelete?: () => void }) {
  const id = result.analysis_id;
  const cls = result.classification ?? "INCONCLUSIVE";
  const art = (n?: string | null) => (n ? api.artifactUrl(id, n) : undefined);
  const v = result.video;

  return (
    <div className="space-y-5" data-testid="result-view">
      <div className="card flex flex-wrap items-center justify-between gap-4 p-5">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <VerdictBadge value={cls} size="lg" />
            <div>
              <p className="font-medium break-all">{result.filename}</p>
              <p className="text-xs text-ink-mute">
                {result.media_type} &middot; {fmtBytes(result.file_size_bytes)} &middot; analysis {id.slice(0, 8)}
                {result.processing_time_s != null && <> &middot; {result.processing_time_s.toFixed(2)} s</>}
              </p>
            </div>
          </div>
          <div className="mt-3 max-w-2xl"><VerdictBlurb value={cls} /></div>
        </div>
        <div className="flex flex-wrap gap-2">
          <a className="btn-primary" href={api.reportUrl(id)} download>Download forensic report (PDF)</a>
          <button className="btn-secondary" onClick={onNew}>New analysis</button>
          {onDelete && <button className="btn-secondary" onClick={onDelete}>Delete results</button>}
        </div>
      </div>

      {result.warnings.length > 0 && (
        <div role="alert" className="rounded-xl border border-warn/30 bg-warn-soft p-4 text-sm text-warn">
          <p className="mb-1 font-semibold">Warnings</p>
          <ul className="list-disc space-y-0.5 pl-5">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-5">
        <div className="space-y-5 lg:col-span-2">
          <Section title="Scores">
            {result.fake_probability != null && result.real_probability != null ? (
              <ScoreBars fake={result.fake_probability} real={result.real_probability} decisiveness={result.decisiveness} label={result.decisiveness_label} />
            ) : (
              <p className="text-sm text-ink-soft">No score could be produced because no usable face was found.</p>
            )}
            <p className="mt-4 rounded-lg bg-paper p-3 text-xs text-ink-soft">{result.score_semantics}</p>
            {result.aggregation_note && <p className="mt-2 text-xs text-ink-mute">{result.aggregation_note}</p>}
          </Section>
          <Section title="Explanation">
            <ul className="list-disc space-y-1.5 pl-5 text-sm">{result.explanation.map((e, i) => <li key={i}>{e}</li>)}</ul>
          </Section>
        </div>

        <div className="space-y-5 lg:col-span-3">
          {result.media_type === "image" && (
            <>
              {result.annotated_artifact && (
                <Section title="Detected faces">
                  <img src={art(result.annotated_artifact)} alt="Analysed image with face bounding boxes" className="max-h-[28rem] w-full rounded-lg object-contain bg-paper" />
                </Section>
              )}
              {result.faces.length > 0 && (
                <Section title={`Per-face results (${result.faces.length})`}>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="text-xs uppercase text-ink-mute"><tr><th className="py-1 pr-3">Face</th><th className="pr-3">Verdict</th><th className="pr-3">Fake</th><th className="pr-3">Real</th><th>Detector</th></tr></thead>
                      <tbody>
                        {result.faces.map((f) => (
                          <tr key={f.index} className="border-t border-paper-line">
                            <td className="py-2 pr-3">#{f.index + 1}</td>
                            <td className="pr-3"><VerdictBadge value={f.label} size="sm" /></td>
                            <td className="pr-3 tabular-nums">{fmt3(f.fake_probability)}</td>
                            <td className="pr-3 tabular-nums">{fmt3(f.real_probability)}</td>
                            <td className="tabular-nums">{fmt3(f.detection_confidence)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {result.faces.some((f) => f.heatmap_artifact) && (
                    <div className="mt-5 grid gap-4 sm:grid-cols-2">
                      {result.faces.filter((f) => f.crop_artifact).slice(0, 6).map((f) => (
                        <figure key={f.index} className="rounded-lg border border-paper-line p-2">
                          <div className="grid grid-cols-2 gap-2">
                            <img src={art(f.crop_artifact)} alt={`Face ${f.index + 1} model input`} className="rounded" />
                            {f.heatmap_artifact && <img src={art(f.heatmap_artifact)} alt={`Face ${f.index + 1} Grad-CAM heat-map`} className="rounded" />}
                          </div>
                          <figcaption className="mt-1 text-xs text-ink-mute">Face #{f.index + 1}: model input (left) and Grad-CAM (right)</figcaption>
                        </figure>
                      ))}
                    </div>
                  )}
                </Section>
              )}
            </>
          )}

          {v && (
            <>
              <Section title="Video summary">
                <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <Stat k="Duration" v={fmtTime(v.duration_s)} />
                  <Stat k="Resolution" v={`${v.width}x${v.height}`} />
                  <Stat k="Frames sampled" v={v.frames_sampled} />
                  <Stat k="With a face" v={v.frames_with_faces} />
                  <Stat k="Face tracks" v={v.tracks.length} />
                  <Stat k="Aggregation" v={v.aggregation_method} />
                  <Stat k="Corrupt skipped" v={v.corrupt_frames_skipped} />
                  <Stat k="FPS" v={v.fps.toFixed(2)} />
                </dl>
                {!v.aggregation_validated && <p className="mt-3 text-xs text-ink-mute">The aggregation method was chosen a priori and has not been validated on labelled videos.</p>}
              </Section>
              <Section title="Frame timeline">
                <VideoTimeline frames={v.frames} duration={v.duration_s} />
                {v.temporal && (
                  <p className="mt-2 text-xs text-ink-soft">
                    {Math.round(v.temporal.suspicious_fraction * 100)}% of frames above 0.5 &middot; longest consecutive run {v.temporal.longest_suspicious_run} &middot; score std {v.temporal.score_std.toFixed(3)} &middot; {v.temporal.sign_flips} boundary crossings
                  </p>
                )}
              </Section>
              <Section title="Suspicious timestamps">
                {v.suspicious_segments.length ? (
                  <ul className="space-y-1 text-sm" data-testid="segments">
                    {v.suspicious_segments.map((s, i) => (
                      <li key={i} className="flex justify-between rounded-lg bg-fake-soft px-3 py-2">
                        <span>{fmtTime(s.start_s)} - {fmtTime(s.end_s)} <span className="text-ink-mute">({s.frames} frame{s.frames > 1 ? "s" : ""})</span></span>
                        <span className="tabular-nums">max {s.max_fake_probability.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-ink-soft">No analysed frame scored above the 0.5 decision boundary.</p>
                )}
                <p className="mt-2 text-xs text-ink-mute">Model-flagged time ranges are leads for review, not proof.</p>
              </Section>
              {v.tracks.length > 0 && (
                <Section title="Face tracks">
                  <table className="w-full text-left text-sm">
                    <thead className="text-xs uppercase text-ink-mute"><tr><th className="py-1">Track</th><th>Frames</th><th>Time span</th><th>Median</th><th>Frames &gt; 0.5</th></tr></thead>
                    <tbody>
                      {v.tracks.slice(0, 8).map((t) => (
                        <tr key={t.track_id} className="border-t border-paper-line">
                          <td className="py-2">#{t.track_id + 1}</td><td>{t.frames}</td>
                          <td>{fmtTime(t.first_timestamp_s)} - {fmtTime(t.last_timestamp_s)}</td>
                          <td className="tabular-nums">{fmt3(t.median_fake_probability)}</td><td>{Math.round(t.suspicious_fraction * 100)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Section>
              )}
              {v.evidence_frames.length > 0 && (
                <Section title="Highest-scoring frames">
                  <div className="grid gap-4 sm:grid-cols-2">
                    {v.evidence_frames.map((e, i) => (
                      <figure key={i} className="rounded-lg border border-paper-line p-2">
                        <img src={art(e.frame_artifact)} alt={`Evidence frame at ${e.timestamp_s.toFixed(1)} seconds`} className="w-full rounded" />
                        {e.heatmap_artifact && <img src={art(e.heatmap_artifact)} alt={`Grad-CAM heat-map at ${e.timestamp_s.toFixed(1)} seconds`} className="mt-2 w-24 rounded" />}
                        <figcaption className="mt-1 text-xs text-ink-mute">t = {fmtTime(e.timestamp_s)} (frame {e.frame_index}) &middot; fake-score {e.fake_probability.toFixed(3)}</figcaption>
                      </figure>
                    ))}
                  </div>
                </Section>
              )}
            </>
          )}
        </div>
      </div>

      <Section title="Scope and limitations">
        <p className="mb-2 text-sm">{result.scope_note}</p>
        <ul className="list-disc space-y-1 pl-5 text-sm text-ink-soft">{LIMITATIONS.map((l, i) => <li key={i}>{l}</li>)}</ul>
        {result.model && (
          <p className="mt-3 text-xs text-ink-mute">
            Model {result.model.name} v{result.model.version} &middot; checkpoint {result.model.checkpoint_file} (sha256 {result.model.checkpoint_sha256.slice(0, 12)}...) &middot; {result.model.device} &middot; threshold {result.model.decision_threshold}
          </p>
        )}
      </Section>
    </div>
  );
}
