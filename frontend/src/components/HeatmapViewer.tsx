import { useEffect, useMemo, useState } from "react";
import type { AnalysisResult, FaceResult, FrameResult, SuspiciousSegment, VideoSummary } from "../types";
import { fmt3, fmtTime } from "../lib/format";
import { IconChevronLeft, IconChevronRight, IconEye } from "./Icons";
import { VerdictBadge } from "./Verdict";

export const GRADCAM_NOTE =
  "Grad-CAM marks the regions that most influenced the model's prediction (RGB stream). It does not independently establish that anything was manipulated, and it does not explain the noise-residual stream.";

type Mode = "side" | "original" | "heatmap" | "overlay";

function ModeSwitch({ mode, onMode, overlayDisabledReason }: { mode: Mode; onMode: (m: Mode) => void; overlayDisabledReason?: string }) {
  const items: [Mode, string][] = [["side", "Side by side"], ["original", "Original"], ["heatmap", "Heatmap"], ["overlay", "Overlay"]];
  return (
    <div className="seg flex-wrap" role="group" aria-label="View mode">
      {items.map(([m, label]) => {
        const disabled = m === "overlay" && !!overlayDisabledReason;
        return (
          <button key={m} type="button" className="seg-btn" aria-pressed={mode === m} disabled={disabled} title={disabled ? overlayDisabledReason : undefined} onClick={() => onMode(m)}>
            {label}
          </button>
        );
      })}
    </div>
  );
}

function Pane({ title, src, alt, caption }: { title: string; src: string; alt: string; caption?: string }) {
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-mute">{title}</figcaption>
      <div className="overflow-hidden rounded-xl border border-paper-line bg-paper">
        <img src={src} alt={alt} className="mx-auto max-h-[26rem] w-full object-contain" />
      </div>
      {caption && <p className="mt-1.5 text-xs text-ink-mute">{caption}</p>}
    </figure>
  );
}

function Unavailable({ children }: { children: React.ReactNode }) {
  return <div role="status" className="rounded-xl border border-warn/40 bg-warn-soft p-4 text-sm text-warn" data-testid="heatmap-unavailable">{children}</div>;
}

/* ---------------------------------------------------------------- images */

/** Original face crop / Grad-CAM / overlay for one detected face. Both images are genuine backend artifacts; the overlay only blends them. */
export function FaceHeatmapViewer({ result, art, faceIndex, onFace }: { result: AnalysisResult; art: (n?: string | null) => string | undefined; faceIndex: number; onFace: (i: number) => void }) {
  const faces = result.faces.filter((f) => f.crop_artifact);
  const [mode, setMode] = useState<Mode>("side");
  const [strength, setStrength] = useState(60);
  const face: FaceResult | undefined = faces.find((f) => f.index === faceIndex) ?? faces[0];
  if (!face) return <Unavailable>No face crop was stored for this image, so there is nothing to show here.</Unavailable>;
  const crop = art(face.crop_artifact), cam = art(face.heatmap_artifact);
  const label = `Face #${face.index + 1}`;
  return (
    <div className="space-y-4" data-testid="face-heatmap-viewer">
      {faces.length > 1 && (
        <div role="group" aria-label="Choose a detected face" className="flex flex-wrap gap-2">
          {faces.map((f) => (
            <button key={f.index} type="button" className="seg-btn border border-paper-line" aria-pressed={f.index === face.index} onClick={() => onFace(f.index)}>
              Face #{f.index + 1} &middot; {f.label}
            </button>
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="flex flex-wrap items-center gap-2 text-sm text-ink-soft">
          <strong className="text-ink">{label}</strong>
          <VerdictBadge value={face.label} size="sm" />
          <span>fake-score {fmt3(face.fake_probability)} &middot; detector confidence {fmt3(face.detection_confidence)}</span>
        </p>
        {cam && <ModeSwitch mode={mode} onMode={setMode} />}
      </div>
      {!cam ? (
        <>
          <Unavailable>A Grad-CAM heat-map is not available for {label} (it was not generated, or the library is not installed on the server). Nothing has been approximated.</Unavailable>
          {crop && <Pane title="Original (face crop)" src={crop} alt={`${label} original face crop`} />}
        </>
      ) : mode === "side" ? (
        <div className="grid gap-4 sm:grid-cols-2">
          <Pane title="Original (face crop)" src={crop!} alt={`${label} original face crop`} caption="The face crop exactly as the model received it." />
          <Pane title="Heatmap (Grad-CAM)" src={cam} alt={`${label} Grad-CAM heat-map`} caption="Warm colours = regions with more influence on the score." />
        </div>
      ) : mode === "original" ? (
        <Pane title="Original (face crop)" src={crop!} alt={`${label} original face crop`} />
      ) : mode === "heatmap" ? (
        <Pane title="Heatmap (Grad-CAM)" src={cam} alt={`${label} Grad-CAM heat-map`} caption="As generated by the server: the Grad-CAM colours are already blended onto this face crop." />
      ) : (
        <figure>
          <figcaption className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-mute">Overlay (original + Grad-CAM)</figcaption>
          <div className="relative mx-auto aspect-square w-full max-w-[26rem] overflow-hidden rounded-xl border border-paper-line bg-paper" data-testid="overlay-stack">
            <img src={crop} alt={`${label} original face crop (overlay base)`} className="absolute inset-0 h-full w-full object-contain" />
            <img src={cam} alt={`${label} Grad-CAM overlay`} className="absolute inset-0 h-full w-full object-contain" style={{ opacity: strength / 100 }} data-testid="overlay-top" />
          </div>
          <label className="mx-auto mt-3 flex max-w-[26rem] items-center gap-3 text-xs text-ink-soft">
            Heat-map strength
            <input type="range" min={0} max={100} value={strength} onChange={(e) => setStrength(Number(e.target.value))} className="w-full accent-[#1fe3c4]" aria-label="Heat-map strength" />
            <span className="w-9 text-right tabular-nums">{strength}%</span>
          </label>
        </figure>
      )}
      <p className="flex gap-2 rounded-xl border border-paper-line bg-paper/60 p-3 text-xs text-ink-soft"><IconEye className="mt-0.5 h-4 w-4 shrink-0 text-accent" />{GRADCAM_NOTE}</p>
    </div>
  );
}

/* ---------------------------------------------------------------- video */

/** Picks the frame to show for a time range: the stored evidence frame with the highest score inside it, else the analysed frame with the highest score. */
export function pickFrameForRange(video: VideoSummary, seg: Pick<SuspiciousSegment, "start_s" | "end_s">): number | null {
  const eps = 1e-6;
  const inside = (t: number) => t >= seg.start_s - eps && t <= seg.end_s + eps;
  const ev = video.evidence_frames.filter((e) => inside(e.timestamp_s)).sort((a, b) => b.fake_probability - a.fake_probability);
  if (ev.length) return ev[0].frame_index;
  const fr = video.frames.filter((f) => inside(f.timestamp_s)).sort((a, b) => (b.frame_fake_probability ?? 0) - (a.frame_fake_probability ?? 0));
  return fr.length ? fr[0].frame_index : null;
}

const OVERLAY_REASON = "Not available for video frames: only the face-level heat-map is stored, not its exact position in the frame, so a registered overlay would be an approximation.";

/** The selected analysed frame with its stored evidence image and Grad-CAM. Frame and heat-map come from the same evidence entry, so they cannot drift apart. */
export function FrameEvidenceViewer({ video, art, selected, onSelect }: { video: VideoSummary; art: (n?: string | null) => string | undefined; selected: number | null; onSelect: (frameIndex: number) => void }) {
  const [mode, setMode] = useState<Mode>("side");
  const frames: FrameResult[] = video.frames;
  const pos = Math.max(0, frames.findIndex((f) => f.frame_index === selected));
  const frame = frames[pos];
  const evidence = useMemo(() => video.evidence_frames.find((e) => e.frame_index === frame?.frame_index), [video, frame]);
  useEffect(() => { if (mode === "overlay") setMode("side"); }, [mode]);
  if (!frame) return <Unavailable>No analysed frames are available for this video.</Unavailable>;
  const top = frame.faces.length ? frame.faces.reduce((a, b) => (b.fake_probability > a.fake_probability ? b : a)) : null;
  const sortedEv = [...video.evidence_frames].sort((a, b) => a.timestamp_s - b.timestamp_s);
  const nearest = sortedEv.length ? sortedEv.reduce((a, b) => (Math.abs(b.timestamp_s - frame.timestamp_s) < Math.abs(a.timestamp_s - frame.timestamp_s) ? b : a)) : null;
  const cam = art(evidence?.heatmap_artifact), img = art(evidence?.frame_artifact);
  const t = fmtTime(frame.timestamp_s);
  return (
    <div className="space-y-4" data-testid="frame-evidence-viewer">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button type="button" className="btn-secondary !px-3" aria-label="Previous analysed frame" disabled={pos <= 0} onClick={() => onSelect(frames[pos - 1].frame_index)}><IconChevronLeft className="h-4 w-4" /></button>
          <p className="min-w-[11rem] text-center text-sm text-ink-soft" aria-live="polite" data-testid="frame-position">
            Frame <strong className="text-ink">{frame.frame_index}</strong> &middot; t = {t} &middot; {pos + 1} of {frames.length}
          </p>
          <button type="button" className="btn-secondary !px-3" aria-label="Next analysed frame" disabled={pos >= frames.length - 1} onClick={() => onSelect(frames[pos + 1].frame_index)}><IconChevronRight className="h-4 w-4" /></button>
        </div>
        {evidence && (img || cam) && <ModeSwitch mode={mode} onMode={setMode} overlayDisabledReason={OVERLAY_REASON} />}
      </div>

      {sortedEv.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-ink-soft">
          Stored frames:
          {sortedEv.map((e) => (
            <button key={e.frame_index} type="button" className="seg-btn border border-paper-line !py-1 text-xs" aria-pressed={e.frame_index === frame.frame_index} onClick={() => onSelect(e.frame_index)}>
              {fmtTime(e.timestamp_s)} &middot; {e.fake_probability.toFixed(2)}
            </button>
          ))}
        </div>
      )}

      <p className="text-sm text-ink-soft">
        Frame fake-score <strong className="text-ink">{fmt3(frame.frame_fake_probability)}</strong>
        {frame.suspicious ? <span className="ml-2 text-fake">above the 0.5 boundary</span> : <span className="ml-2">at or below the 0.5 boundary</span>}
        {frame.faces.length > 0 && <> &middot; {frame.faces.length} face{frame.faces.length > 1 ? "s" : ""}: {frame.faces.map((f) => `track #${f.track_id + 1} (${f.fake_probability.toFixed(2)})`).join(", ")}</>}
      </p>

      {!evidence ? (
        <Unavailable>
          No frame image or heat-map was stored for this timestamp. The server keeps them only for the {video.evidence_frames.length} highest-scoring frames, and nothing has been approximated.
          {nearest && (
            <> <button type="button" className="font-semibold underline" onClick={() => onSelect(nearest.frame_index)}>Jump to the nearest stored frame ({fmtTime(nearest.timestamp_s)})</button></>
          )}
        </Unavailable>
      ) : (
        <>
          {!cam && <Unavailable>A Grad-CAM heat-map is not available for this frame. Only the frame image is shown.</Unavailable>}
          {mode === "side" || mode === "overlay" ? (
            <div className={`grid gap-4 ${cam ? "md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]" : ""}`}>
              {img && <Pane title="Original frame (faces boxed)" src={img} alt={`Evidence frame at ${evidence.timestamp_s.toFixed(1)} seconds`} caption="The analysed video frame with the detected face boxes drawn by the server." />}
              {cam && <Pane title="Heatmap (Grad-CAM)" src={cam} alt={`Grad-CAM heat-map at ${evidence.timestamp_s.toFixed(1)} seconds`} caption={top ? `For the highest-scoring face in this frame (track #${top.track_id + 1}, fake-score ${top.fake_probability.toFixed(3)}).` : undefined} />}
            </div>
          ) : mode === "original" ? (
            img ? <Pane title="Original frame (faces boxed)" src={img} alt={`Evidence frame at ${evidence.timestamp_s.toFixed(1)} seconds`} /> : null
          ) : (
            cam ? <Pane title="Heatmap (Grad-CAM)" src={cam} alt={`Grad-CAM heat-map at ${evidence.timestamp_s.toFixed(1)} seconds`} caption="Face-level heat-map as generated; it is blended onto the face crop, not onto the whole frame." /> : null
          )}
          <p className="flex gap-2 rounded-xl border border-paper-line bg-paper/60 p-3 text-xs text-ink-soft"><IconEye className="mt-0.5 h-4 w-4 shrink-0 text-accent" />{GRADCAM_NOTE}</p>
        </>
      )}
    </div>
  );
}
