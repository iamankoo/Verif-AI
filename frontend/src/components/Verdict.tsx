import type { Classification } from "../types";
import { VERDICT_TEXT } from "../lib/format";
import { IconCheckCircle, IconHelp, IconWarning } from "./Icons";

const STYLE: Record<Classification, string> = {
  REAL: "bg-real-soft text-real border-real/50",
  FAKE: "bg-fake-soft text-fake border-fake/60",
  INCONCLUSIVE: "bg-warn-soft text-warn border-warn/50",
};
const ICON = { REAL: IconCheckCircle, FAKE: IconWarning, INCONCLUSIVE: IconHelp } as const;

export function VerdictBadge({ value, size = "md" }: { value: Classification; size?: "sm" | "md" | "lg" }) {
  const cls = size === "lg" ? "gap-2 px-6 py-2.5 text-3xl" : size === "sm" ? "gap-1 px-2 py-0.5 text-xs" : "gap-1.5 px-3 py-1 text-sm";
  const Icon = ICON[value];
  return (
    <span data-testid="verdict" className={`inline-flex items-center rounded-full border font-bold tracking-wide ${STYLE[value]} ${cls}`}>
      <Icon className={size === "lg" ? "h-7 w-7" : size === "sm" ? "h-3 w-3" : "h-4 w-4"} />
      {value}
    </span>
  );
}

export function VerdictBlurb({ value }: { value: Classification }) {
  return <p className="text-sm text-ink-soft">{VERDICT_TEXT[value].blurb}</p>;
}

export function ScoreBars({ fake, real, decisiveness, label }: { fake: number; real: number; decisiveness: number | null; label: string | null }) {
  return (
    <div className="space-y-4" aria-label="Model scores">
      {[
        { name: "Real score", v: real, color: "bg-real" },
        { name: "Fake score", v: fake, color: "bg-fake" },
      ].map((b) => (
        <div key={b.name}>
          <div className="mb-1.5 flex justify-between text-sm">
            <span>{b.name}</span>
            <span className="tabular-nums font-semibold" data-testid={b.name === "Fake score" ? "fake-score" : "real-score"}>
              {b.v.toFixed(3)} <span className="font-normal text-ink-mute">model score, 0-1</span>
            </span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-paper" role="meter" aria-label={b.name} aria-valuemin={0} aria-valuemax={1} aria-valuenow={b.v}>
            <div className={`h-full rounded-full ${b.color}`} style={{ width: `${Math.max(0, Math.min(1, b.v)) * 100}%` }} />
          </div>
        </div>
      ))}
      {decisiveness != null && (
        <p className="rounded-xl border border-paper-line bg-paper/60 p-3 text-xs text-ink-soft">
          Distance from the 0.5 decision boundary: <strong>{decisiveness.toFixed(2)}</strong> ({label}). This is not a probability that the verdict is correct.
        </p>
      )}
    </div>
  );
}
