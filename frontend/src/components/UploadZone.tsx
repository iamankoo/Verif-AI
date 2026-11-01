import { useRef, useState } from "react";
import type { Limits } from "../types";
import { DEFAULT_LIMITS, validateFile, type Kind } from "../lib/format";
import { IconFolder, IconUpload } from "./Icons";

interface Props {
  limits?: Pick<Limits, "max_image_mb" | "max_video_mb" | "image_formats" | "video_formats">;
  /** Which kind the visible tab is about; the zone still accepts both (a file of the other kind switches the tab in the parent). */
  mode?: Kind | "any";
  disabled?: boolean;
  onSelect: (file: File, kind: Kind) => void;
  onError: (message: string) => void;
}

const names = (exts: string[]) => exts.map((e) => e.replace(".", "").toUpperCase()).join(", ");

export function UploadZone({ limits = DEFAULT_LIMITS, mode = "any", disabled, onSelect, onError }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handle = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    const v = validateFile(f, limits);
    if (v.error || !v.kind) onError(v.error ?? "Unsupported file.");
    else onSelect(f, v.kind);
  };

  const accept = mode === "image" ? limits.image_formats : mode === "video" ? limits.video_formats : [...limits.image_formats, ...limits.video_formats];
  const headline = mode === "image" ? "Drag and drop an image here" : mode === "video" ? "Drag and drop a video here" : "Drag and drop an image or video here";
  const maxMb = mode === "video" ? limits.max_video_mb : limits.max_image_mb;

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) handle(e.dataTransfer.files);
      }}
      className={`rounded-2xl border-2 border-dashed p-6 text-center transition sm:p-8 ${over ? "border-accent bg-accent-soft" : "border-paper-line/90 bg-paper/40"} ${disabled ? "opacity-60" : ""}`}
      data-testid="dropzone"
    >
      <input
        ref={input}
        type="file"
        className="sr-only"
        aria-label="Choose an image or video file"
        accept={accept.join(",")}
        disabled={disabled}
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = "";
        }}
      />
      <IconUpload className="mx-auto mb-3 h-11 w-11 text-ink" />
      <p className="text-lg font-semibold">{headline}</p>
      <p className="mt-1 text-sm text-ink-soft">or click to browse from your device</p>
      <button type="button" className="btn-primary mt-4" disabled={disabled} onClick={() => input.current?.click()}>
        <IconFolder className="h-4 w-4" /> Browse files
      </button>
      <p className="mt-4 text-xs text-ink-mute">
        {mode === "any" ? (
          <>Images up to {limits.max_image_mb} MB, videos up to {limits.max_video_mb} MB &middot; {accept.join(" ")}</>
        ) : (
          <>Supported: {names(accept)} &nbsp;|&nbsp; Max size: {maxMb} MB</>
        )}
      </p>
    </div>
  );
}
