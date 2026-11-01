import type { FrameResult } from "../types";
import { fmtTime } from "../lib/format";

interface Props {
  frames: FrameResult[];
  duration: number;
  threshold?: number;
}

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
        <rect x={L} y={T} width={iw} height={ih} fill="#f6f7f9" stroke="#dfe3e9" />
        <rect x={L} y={T} width={iw} height={ih * (1 - threshold)} fill="#b4301f" opacity="0.06" />
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={L + iw} y1={y(v)} y2={y(v)} stroke="#dfe3e9" strokeWidth={v === threshold ? 0 : 0.5} />
            <text x={L - 6} y={y(v) + 3} fontSize="10" textAnchor="end" fill="#5b6472">{v.toFixed(2)}</text>
          </g>
        ))}
        <line x1={L} x2={L + iw} y1={y(threshold)} y2={y(threshold)} stroke="#b4301f" strokeDasharray="5 4" strokeWidth="1.2" />
        <text x={L + iw} y={y(threshold) - 4} fontSize="10" textAnchor="end" fill="#b4301f">decision boundary 0.5</text>
        {pts.length > 1 && <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="#14213d" strokeWidth="1.6" />}
        {frames.map((f, i) => (
          <circle key={f.frame_index} cx={pts[i][0]} cy={pts[i][1]} r="3.4" fill={f.suspicious ? "#b4301f" : "#14213d"} stroke="#fff" strokeWidth="1">
            <title>{`t=${fmtTime(f.timestamp_s)}  frame ${f.frame_index}  fake-score ${(f.frame_fake_probability ?? 0).toFixed(3)}`}</title>
          </circle>
        ))}
        <text x={L} y={H - 8} fontSize="10" fill="#5b6472">0 s</text>
        <text x={L + iw} y={H - 8} fontSize="10" textAnchor="end" fill="#5b6472">{fmtTime(tmax)}</text>
      </svg>
      <figcaption className="mt-1 text-xs text-ink-mute">Each dot is one analysed frame (highest face score in that frame). Red dots exceed 0.5.</figcaption>
    </figure>
  );
}
