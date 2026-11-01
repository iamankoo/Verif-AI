import type { Classification } from "../types";
import { VERDICT_TEXT, pct } from "../lib/format";

const STYLE: Record<Classification, string> = {
  REAL: "bg-real-soft text-real border-real/30",
  FAKE: "bg-fake-soft text-fake border-fake/30",
  INCONCLUSIVE: "bg-warn-soft text-warn border-warn/30",
};

export function VerdictBadge({ value, size = "md" }: { value: Classification; size?: "sm" | "md" | "lg" }) {
  const cls = size === "lg" ? "px-5 py-2 text-2xl" : size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm";
  return (
    <span data-testid="verdict" className={`inline-block rounded-full border font-semibold tracking-wide ${STYLE[value]} ${cls}`}>
      {value}
    </span>
  );
}

export function VerdictBlurb({ value }: { value: Classification }) {
  return <p className="text-sm text-ink-soft">{VERDICT_TEXT[value].blurb}</p>;
}

export function ScoreBars({ fake, real, decisiveness, label }: { fake: number; real: number; decisiveness: number | null; label: string | null }) {
  return (
    <div className="space-y-3" aria-label="Model scores">
      {[
        { name: "Real score", v: real, color: "bg-real" },
        { name: "Fake score", v: fake, color: "bg-fake" },
      ].map((b) => (
        <div key={b.name}>
          <div className="mb-1 flex justify-between text-sm">
            <span>{b.name}</span>
            <span className="tabular-nums font-medium" data-testid={b.name === "Fake score" ? "fake-score" : "real-score"}>
              {b.v.toFixed(3)} <span className="text-ink-mute">({pct(b.v)})</span>
            </span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-paper" role="meter" aria-label={b.name} aria-valuemin={0} aria-valuemax={1} aria-valuenow={b.v}>
            <div className={`h-full ${b.color}`} style={{ width: `${Math.max(0, Math.min(1, b.v)) * 100}%` }} />
          </div>
        </div>
      ))}
      {decisiveness != null && (
        <p className="text-xs text-ink-mute">
          Distance from the 0.5 decision boundary: <strong>{decisiveness.toFixed(2)}</strong> ({label}). This is not a probability that the verdict is correct.
        </p>
      )}
    </div>
  );
}
