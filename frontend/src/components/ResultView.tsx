import { useRef, useState } from "react";
import { api } from "../api";
import type { AnalysisResult, Classification } from "../types";
import { fmt3, fmtBytes, fmtTime } from "../lib/format";
import { IconArrowLeft, IconCalendar, IconChart, IconClock, IconDownload, IconFile, IconGauge, IconImage, IconLayers, IconScan, IconStack, IconTrash, IconUser, IconVideo } from "./Icons";
import { ScoreBars, VerdictBadge, VerdictBlurb } from "./Verdict";
import { VideoTimeline } from "./VideoTimeline";

const LIMITATIONS = [
  "This is automated, model-based evidence - not definitive proof of manipulation or authenticity.",
  "Scores are uncalibrated neural-network outputs, not real-world probabilities.",
  "The model analyses faces only and is validated only on the data and generators listed in the model card; other generators, heavy compression, low resolution and adversarial edits can cause errors.",
  "Grad-CAM shows regions that influenced the RGB stream's fake score. It does not prove those pixels were altered and does not explain the noise-residual stream.",
  "A REAL result does not prove authenticity; a FAKE result does not say who or what created the media.",
];

const VERDICT_CAPTION: Record<Classification, string> = {
  REAL: "No patterns associated with fabricated faces were found (not proof of authenticity)",
  FAKE: "Patterns associated with fabricated faces were found (model-based evidence, not proof)",
  INCONCLUSIVE: "Not enough usable evidence for a reliable verdict",
};

function Section({ title, icon, children, className = "" }: { title: string; icon?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`card p-5 ${className}`}>
      <h3 className="mb-3 flex items-center gap-2.5 text-lg">
        {icon && <span className="grid h-9 w-9 place-items-center rounded-lg bg-accent-soft text-accent">{icon}</span>}
        {title}
      </h3>
      {children}
    </section>
  );
}

function Stat({ k, v, icon }: { k: string; v: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-paper-line bg-paper/50 p-3">
      {icon && <span className="text-accent">{icon}</span>}
      <div>
        <dd className="text-lg font-semibold tabular-nums leading-tight">{v}</dd>
        <dt className="text-xs text-ink-mute">{k}</dt>
      </div>
    </div>
  );
}

type TabId = "overview" | "frames" | "tracks" | "evidence" | "report";

export function ResultView({ result, onNew, onDelete }: { result: AnalysisResult; onNew: () => void; onDelete?: () => void }) {
  const id = result.analysis_id;
  const cls = result.classification ?? "INCONCLUSIVE";
  const art = (n?: string | null) => (n ? api.artifactUrl(id, n) : undefined);
  const v = result.video;
  const tabs: { id: TabId; label: string; Icon: typeof IconChart }[] = [
    { id: "overview", label: "Overview", Icon: IconScan },
    ...(v ? [{ id: "frames" as const, label: "Frame Analysis", Icon: IconLayers }, { id: "tracks" as const, label: "Face Tracks", Icon: IconUser }] : []),
    { id: "evidence", label: "Forensic Evidence", Icon: IconStack },
    { id: "report", label: "Report", Icon: IconFile },
  ];
  const [tab, setTab] = useState<TabId>("overview");
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const next = e.key === "ArrowRight" ? (i + 1) % tabs.length : e.key === "ArrowLeft" ? (i - 1 + tabs.length) % tabs.length : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : -1;
    if (next >= 0) {
      e.preventDefault();
      setTab(tabs[next].id);
      tabRefs.current[tabs[next].id]?.focus();
    }
  };
  const hasHeatmaps = result.faces.some((f) => f.heatmap_artifact) || !!v?.evidence_frames.some((e) => e.heatmap_artifact);
  const thumb = result.media_type === "image" ? result.annotated_artifact : v?.evidence_frames[0]?.frame_artifact;
  const created = new Date(result.created_at);

  const detectedFaces = result.media_type === "image" && result.annotated_artifact && (
    <img src={art(result.annotated_artifact)} alt="Analysed image with face bounding boxes" className="max-h-[28rem] w-full rounded-xl bg-paper object-contain" />
  );

  const evidenceFrames = v && v.evidence_frames.length > 0 && (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {v.evidence_frames.map((e, i) => (
        <figure key={i} className="rounded-xl border border-paper-line bg-paper/50 p-2">
          <img src={art(e.frame_artifact)} alt={`Evidence frame at ${e.timestamp_s.toFixed(1)} seconds`} className="w-full rounded-lg" />
          <div className="mt-2 flex items-center gap-3">
            {e.heatmap_artifact && <img src={art(e.heatmap_artifact)} alt={`Grad-CAM heat-map at ${e.timestamp_s.toFixed(1)} seconds`} className="h-16 w-16 rounded-lg object-cover" />}
            <figcaption className="text-xs text-ink-soft">t = {fmtTime(e.timestamp_s)} (frame {e.frame_index})<br />fake-score {e.fake_probability.toFixed(3)}</figcaption>
          </div>
        </figure>
      ))}
    </div>
  );

  return (
    <div className="space-y-5" data-testid="result-view">
      <div className="card grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="flex flex-wrap items-start gap-5 sm:flex-nowrap">
          <div className="grid h-32 w-full max-w-[14rem] shrink-0 place-items-center overflow-hidden rounded-xl border border-paper-line bg-paper sm:h-36 sm:w-56" aria-hidden={!thumb}>
            {thumb ? <img src={art(thumb)} alt="" className="h-full w-full object-cover" /> : result.media_type === "image" ? <IconImage className="h-10 w-10 text-ink-mute" /> : <IconVideo className="h-10 w-10 text-ink-mute" />}
          </div>
          <div className="min-w-0">
            <p className="break-all text-xl font-semibold sm:text-2xl">{result.filename}</p>
            <ul className="mt-2 flex flex-wrap gap-2">
              <li className="chip">{result.media_type === "image" ? <IconImage className="h-3.5 w-3.5" /> : <IconVideo className="h-3.5 w-3.5" />} {result.media_type === "image" ? "Image" : "Video"}</li>
              <li className="chip"><IconFile className="h-3.5 w-3.5" /> {fmtBytes(result.file_size_bytes)}</li>
              {result.processing_time_s != null && <li className="chip"><IconClock className="h-3.5 w-3.5" /> {result.processing_time_s.toFixed(2)} s</li>}
              <li className="chip"><IconCalendar className="h-3.5 w-3.5" /> {Number.isNaN(created.getTime()) ? "" : created.toLocaleString()}</li>
              <li className="chip">analysis {id.slice(0, 8)}</li>
            </ul>
            <div className="mt-3 max-w-xl"><VerdictBlurb value={cls} /></div>
          </div>
        </div>
        <div className="rounded-2xl border border-paper-line bg-paper/60 p-4 text-center lg:border-l lg:bg-transparent">
          <p className="mb-2 text-sm text-ink-mute">Verdict</p>
          <VerdictBadge value={cls} size="lg" />
          <p className="mt-3 text-xs text-ink-soft">{VERDICT_CAPTION[cls]}</p>
        </div>
      </div>

      {result.warnings.length > 0 && (
        <div role="alert" className="rounded-2xl border border-warn/40 bg-warn-soft p-4 text-sm text-warn">
          <p className="mb-1 font-semibold">Warnings</p>
          <ul className="list-disc space-y-0.5 pl-5">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}

      <div role="tablist" aria-label="Result sections" className="flex gap-1 overflow-x-auto border-b border-paper-line pb-0">
        {tabs.map(({ id: tid, label, Icon }, i) => (
          <button
            key={tid}
            ref={(el) => { tabRefs.current[tid] = el; }}
            role="tab"
            id={`tab-${tid}`}
            aria-selected={tab === tid}
            aria-controls={`panel-${tid}`}
            tabIndex={tab === tid ? 0 : -1}
            onClick={() => setTab(tid)}
            onKeyDown={(e) => onKey(e, i)}
            className={`-mb-px flex shrink-0 items-center gap-2 rounded-t-xl border border-b-0 px-4 py-2.5 text-sm font-semibold transition ${tab === tid ? "border-accent/60 bg-accent-soft text-accent" : "border-transparent text-ink-soft hover:text-ink"}`}
          >
            <Icon className="h-4 w-4" /> {label}
          </button>
        ))}
      </div>

      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="space-y-5">
        {tab === "overview" && (
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="space-y-5">
              <Section title="Detection Scores" icon={<IconChart className="h-5 w-5" />}>
                {result.fake_probability != null && result.real_probability != null ? (
                  <ScoreBars fake={result.fake_probability} real={result.real_probability} decisiveness={result.decisiveness} label={result.decisiveness_label} />
                ) : (
                  <p className="text-sm text-ink-soft">No score could be produced because no usable face was found.</p>
                )}
                <p className="mt-4 rounded-xl bg-paper/60 p-3 text-xs text-ink-soft">{result.score_semantics}</p>
                {result.aggregation_note && <p className="mt-2 text-xs text-ink-mute">{result.aggregation_note}</p>}
              </Section>
              <Section title="Analysis Summary" icon={<IconGauge className="h-5 w-5" />}>
                <dl className="grid grid-cols-2 gap-3 xl:grid-cols-4">
                  {v ? (
                    <>
                      <Stat k="Frames sampled" v={v.frames_sampled} icon={<IconScan className="h-5 w-5" />} />
                      <Stat k="Faces detected" v={result.faces_detected} icon={<IconUser className="h-5 w-5" />} />
                      <Stat k="Face track(s)" v={v.tracks.length} icon={<IconLayers className="h-5 w-5" />} />
                      <Stat k="Sampling rate" v={`~${v.sampling_fps.toFixed(1)} fps`} icon={<IconGauge className="h-5 w-5" />} />
                    </>
                  ) : (
                    <>
                      <Stat k="Faces detected" v={result.faces_detected} icon={<IconUser className="h-5 w-5" />} />
                      <Stat k="Resolution" v={result.image_width && result.image_height ? `${result.image_width}x${result.image_height}` : "n/a"} icon={<IconImage className="h-5 w-5" />} />
                      <Stat k="Processing" v={result.processing_time_s != null ? `${result.processing_time_s.toFixed(2)} s` : "n/a"} icon={<IconClock className="h-5 w-5" />} />
                      <Stat k="File size" v={fmtBytes(result.file_size_bytes)} icon={<IconFile className="h-5 w-5" />} />
                    </>
                  )}
                </dl>
              </Section>
              <Section title="Explanation">
                <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-soft">{result.explanation.map((e, i) => <li key={i}>{e}</li>)}</ul>
              </Section>
            </div>
            <div className="space-y-5">
              {result.media_type === "image" ? (
                detectedFaces && <Section title="Detected faces" icon={<IconScan className="h-5 w-5" />}>{detectedFaces}</Section>
              ) : (
                evidenceFrames && <Section title="Highest-Scoring Frames" icon={<IconImage className="h-5 w-5" />}>{evidenceFrames}</Section>
              )}
              {result.model && (
                <Section title="Model Information" icon={<IconStack className="h-5 w-5" />}>
                  <dl className="space-y-1.5 text-sm">
                    {([
                      ["Model", `${result.model.name} v${result.model.version}`],
                      ["Architecture", "Two-stream EfficientNet-B0 (RGB + SRM)"],
                      ["Parameters", result.model.parameters.toLocaleString()],
                      ["Decision threshold", String(result.model.decision_threshold)],
                      ["Grad-CAM", hasHeatmaps ? "Available for this result" : "Not available for this result"],
                    ] as const).map(([k, val]) => (
                      <div key={k} className="flex justify-between gap-4"><dt className="text-ink-mute">{k}</dt><dd className="text-right font-medium">{val}</dd></div>
                    ))}
                  </dl>
                  <p className="mt-3 text-xs text-ink-mute">Grad-CAM highlights the image regions that most influenced the RGB stream&apos;s score.</p>
                </Section>
              )}
            </div>
          </div>
        )}

        {tab === "frames" && v && (
          <>
            <Section title="Video summary" icon={<IconVideo className="h-5 w-5" />}>
              <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
            <Section title="Frame timeline" icon={<IconChart className="h-5 w-5" />}>
              <VideoTimeline frames={v.frames} duration={v.duration_s} />
              {v.temporal && (
                <p className="mt-2 text-xs text-ink-soft">
                  {Math.round(v.temporal.suspicious_fraction * 100)}% of frames above 0.5 &middot; longest consecutive run {v.temporal.longest_suspicious_run} &middot; score std {v.temporal.score_std.toFixed(3)} &middot; {v.temporal.sign_flips} boundary crossings
                </p>
              )}
            </Section>
            <Section title="Suspicious timestamps" icon={<IconClock className="h-5 w-5" />}>
              {v.suspicious_segments.length ? (
                <ul className="space-y-1.5 text-sm" data-testid="segments">
                  {v.suspicious_segments.map((s, i) => (
                    <li key={i} className="flex justify-between rounded-xl bg-fake-soft px-3 py-2">
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
          </>
        )}

        {tab === "tracks" && v && (
          <Section title="Face tracks" icon={<IconUser className="h-5 w-5" />}>
            {v.tracks.length > 0 ? (
              <div className="overflow-x-auto">
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
              </div>
            ) : (
              <p className="text-sm text-ink-soft">No face track was long enough to summarise.</p>
            )}
            <p className="mt-3 text-xs text-ink-mute">A track follows one face across the sampled frames. The video score is the highest track median.</p>
          </Section>
        )}

        {tab === "evidence" && (
          <>
            {result.media_type === "image" ? (
              <>
                {detectedFaces && <Section title="Detected faces" icon={<IconScan className="h-5 w-5" />}>{detectedFaces}</Section>}
                {result.faces.length > 0 ? (
                  <Section title={`Per-face results (${result.faces.length})`} icon={<IconUser className="h-5 w-5" />}>
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
                          <figure key={f.index} className="rounded-xl border border-paper-line bg-paper/50 p-2">
                            <div className="grid grid-cols-2 gap-2">
                              <img src={art(f.crop_artifact)} alt={`Face ${f.index + 1} model input`} className="rounded-lg" />
                              {f.heatmap_artifact && <img src={art(f.heatmap_artifact)} alt={`Face ${f.index + 1} Grad-CAM heat-map`} className="rounded-lg" />}
                            </div>
                            <figcaption className="mt-1 text-xs text-ink-mute">Face #{f.index + 1}: model input (left) and Grad-CAM (right)</figcaption>
                          </figure>
                        ))}
                      </div>
                    )}
                  </Section>
                ) : (
                  <Section title="Per-face results"><p className="text-sm text-ink-soft">No usable face was found, so there is no per-face evidence.</p></Section>
                )}
              </>
            ) : (
              <Section title="Highest-scoring frames" icon={<IconImage className="h-5 w-5" />}>
                {evidenceFrames || <p className="text-sm text-ink-soft">No evidence frames were produced for this video.</p>}
                <p className="mt-3 text-xs text-ink-mute">Grad-CAM thumbnails show regions that influenced the RGB stream&apos;s score; they are not proof that pixels were altered.</p>
              </Section>
            )}
          </>
        )}

        {tab === "report" && (
          <>
            <Section title="Forensic report" icon={<IconFile className="h-5 w-5" />}>
              <p className="text-sm text-ink-soft">The PDF records the verdict and scores, the file name, size and SHA-256, the model checkpoint and its hash, the evidence images, tables and the limitations below. It is generated on request from the stored result.</p>
              <a className="btn-secondary mt-4" href={api.reportUrl(id)} download><IconDownload className="h-4 w-4" /> Download PDF</a>
              <dl className="mt-4 space-y-1.5 text-sm">
                <div className="flex justify-between gap-4"><dt className="text-ink-mute">Analysis id</dt><dd className="break-all text-right font-mono text-xs">{id}</dd></div>
                {result.file_sha256 && <div className="flex justify-between gap-4"><dt className="text-ink-mute">File SHA-256</dt><dd className="break-all text-right font-mono text-xs">{result.file_sha256}</dd></div>}
              </dl>
              <p className="mt-4 text-xs text-ink-mute">Your original upload has already been deleted. The evidence images and result are kept on this server for a limited time and are removed immediately when you delete the result.</p>
            </Section>
            <Section title="Scope and limitations">
              <p className="mb-2 text-sm">{result.scope_note}</p>
              <ul className="list-disc space-y-1 pl-5 text-sm text-ink-soft">{LIMITATIONS.map((l, i) => <li key={i}>{l}</li>)}</ul>
              {result.model && (
                <p className="mt-3 text-xs text-ink-mute">
                  Model {result.model.name} v{result.model.version} &middot; checkpoint {result.model.checkpoint_file} (sha256 {result.model.checkpoint_sha256.slice(0, 12)}...) &middot; {result.model.device} &middot; threshold {result.model.decision_threshold}
                </p>
              )}
            </Section>
          </>
        )}
      </div>

      <div className="card flex flex-wrap items-center justify-between gap-3 p-4" role="group" aria-label="Actions">
        <button className="btn-secondary" onClick={onNew}><IconArrowLeft className="h-4 w-4" /> New analysis</button>
        <div className="flex flex-wrap gap-2">
          <a className="btn-primary" href={api.reportUrl(id)} download><IconDownload className="h-4 w-4" /> Download forensic report (PDF)</a>
          {onDelete && <button className="btn-danger" onClick={onDelete}><IconTrash className="h-4 w-4" /> Delete results</button>}
        </div>
      </div>
    </div>
  );
}
