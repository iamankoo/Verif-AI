import type { FrameResult } from "../types";
import { fmtTime } from "../lib/format";

interface Props {
  frames: FrameResult[];
  duration: number;
  threshold?: number;
  /** frame_index of the selected frame (highlighted); selecting a dot calls onSelect. */
  selected?: number | null;
  onSelect?: (frameIndex: number) => void;
  /** frame_index values that have a stored evidence image/heat-map (drawn with a ring). */
  evidence?: ReadonlySet<number>;
}

const GRID = "#14474d", MUTE = "#82a8ac", LINE = "#1fe3c4", FAKE = "#ff6262";

/** Frame-level fake-score over time. Red dashed line = 0.5 decision boundary. Dots are clickable when onSelect is given. */
export function VideoTimeline({ frames, duration, threshold = 0.5, selected = null, onSelect, evidence }: Props) {
  const W = 720, H = 200, L = 40, R = 12, T = 14, B = 28;
  const iw = W - L - R, ih = H - T - B;
  const tmax = Math.max(duration, (frames.length ? frames[frames.length - 1].timestamp_s : 0), 0.001);
  const x = (t: number) => L + (iw * t) / tmax;
  const y = (p: number) => T + ih * (1 - p);
  const pts = frames.map((f) => [x(f.timestamp_s), y(f.frame_fake_probability ?? 0)] as const);
  return (
    <figure>
      <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[34rem]" role="img" aria-label={`Frame-level fake-score timeline over ${tmax.toFixed(1)} seconds`}>
        <rect x={L} y={T} width={iw} height={ih} fill="#051820" stroke={GRID} />
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
        {frames.map((f, i) => {
          const sel = selected === f.frame_index;
          const label = `t=${fmtTime(f.timestamp_s)}  frame ${f.frame_index}  fake-score ${(f.frame_fake_probability ?? 0).toFixed(3)}`;
          return (
            <g
              key={f.frame_index}
              role={onSelect ? "button" : undefined}
              tabIndex={onSelect ? 0 : undefined}
              aria-label={onSelect ? `Select ${label}` : undefined}
              aria-pressed={onSelect ? sel : undefined}
              onClick={onSelect ? () => onSelect(f.frame_index) : undefined}
              onKeyDown={onSelect ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(f.frame_index); } } : undefined}
              className={onSelect ? "cursor-pointer outline-none focus-visible:[&>circle:first-of-type]:stroke-white" : undefined}
              data-testid={`tl-dot-${f.frame_index}`}
            >
              {(sel || evidence?.has(f.frame_index)) && <circle cx={pts[i][0]} cy={pts[i][1]} r={sel ? 8 : 6.5} fill="none" stroke={sel ? "#ffffff" : LINE} strokeWidth={sel ? 1.8 : 1} opacity={sel ? 1 : 0.7} />}
              <circle cx={pts[i][0]} cy={pts[i][1]} r="3.8" fill={f.suspicious ? FAKE : LINE} stroke="#04121b" strokeWidth="1" />
              <title>{label}</title>
            </g>
          );
        })}
        <text x={L} y={H - 8} fontSize="10" fill={MUTE}>0 s</text>
        <text x={L + iw} y={H - 8} fontSize="10" textAnchor="end" fill={MUTE}>{fmtTime(tmax)}</text>
      </svg>
      </div>
      <figcaption className="mt-1 text-xs text-ink-mute">
        Each dot is one analysed frame (highest face score in that frame). Red dots exceed 0.5.{evidence && evidence.size > 0 ? " Ringed dots have a stored frame and heat-map." : ""}{onSelect ? " Select a dot to inspect that frame." : ""}
      </figcaption>
    </figure>
  );
}
