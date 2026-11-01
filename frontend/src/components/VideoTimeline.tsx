import type { FrameResult } from "../types";
import { fmtTime } from "../lib/format";

interface Props {
  frames: FrameResult[];
  duration: number;
  threshold?: number;
}

const GRID = "#e2e8f0", MUTE = "#64748b", LINE = "#0f766e", FAKE = "#b91c1c";

/** Frame-level fake-score over time. Red dashed line = 0.5 decision boundary. */
export function VideoTimeline({ frames, duration, threshold = 0.5 }: Props) {
  const W = 720, H = 200, L = 40, R = 12, T = 14, B = 28;
  const iw = W - L - R, ih = H - T - B;
  const tmax = Math.max(duration, (frames.length ? frames[frames.length - 1].timestamp_s : 0), 0.001);
  const x = (t: number) => L + (iw * t) / tmax;
  const y = (p: number) => T + ih * (1 - p);
  const pts = frames.map((f) => [x(f.timestamp_s), y(f.frame_fake_probability ?? 0)] as const);
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Frame-level fake-score timeline over ${tmax.toFixed(1)} seconds`}>
        <rect x={L} y={T} width={iw} height={ih} fill="#f8fafc" stroke={GRID} />
        <rect x={L} y={T} width={iw} height={ih * (1 - threshold)} fill={FAKE} opacity="0.10" />
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={L + iw} y1={y(v)} y2={y(v)} stroke={GRID} strokeWidth={v === threshold ? 0 : 0.7} />
            <text x={L - 6} y={y(v) + 3} fontSize="10" textAnchor="end" fill={MUTE}>{v.toFixed(2)}</text>
          </g>
        ))}
        <line x1={L} x2={L + iw} y1={y(threshold)} y2={y(threshold)} stroke={FAKE} strokeDasharray="5 4" strokeWidth="1.2" />
        <text x={L + iw} y={y(threshold) - 4} fontSize="10" textAnchor="end" fill={FAKE}>decision boundary 0.5</text>
        {pts.length > 1 && <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={LINE} strokeWidth="1.8" />}
        {frames.map((f, i) => (
          <circle key={f.frame_index} cx={pts[i][0]} cy={pts[i][1]} r="3.8" fill={f.suspicious ? FAKE : LINE} stroke="#ffffff" strokeWidth="1">
            <title>{`t=${fmtTime(f.timestamp_s)}  frame ${f.frame_index}  fake-score ${(f.frame_fake_probability ?? 0).toFixed(3)}`}</title>
          </circle>
        ))}
        <text x={L} y={H - 8} fontSize="10" fill={MUTE}>0 s</text>
        <text x={L + iw} y={H - 8} fontSize="10" textAnchor="end" fill={MUTE}>{fmtTime(tmax)}</text>
      </svg>
      <figcaption className="mt-1 text-xs text-ink-mute">Each dot is one analysed frame (highest face score in that frame). Red dots exceed 0.5.</figcaption>
    </figure>
  );
}
