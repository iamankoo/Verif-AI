import { useRef, useState } from "react";
import type { Limits } from "../types";
import { DEFAULT_LIMITS, validateFile, type Kind } from "../lib/format";

interface Props {
  limits?: Pick<Limits, "max_image_mb" | "max_video_mb" | "image_formats" | "video_formats">;
  disabled?: boolean;
  onSelect: (file: File, kind: Kind) => void;
  onError: (message: string) => void;
}

export function UploadZone({ limits = DEFAULT_LIMITS, disabled, onSelect, onError }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const handle = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    const v = validateFile(f, limits);
    if (v.error || !v.kind) onError(v.error ?? "Unsupported file.");
    else onSelect(f, v.kind);
  };

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
      className={`rounded-xl border-2 border-dashed p-8 text-center transition ${over ? "border-accent bg-accent-soft" : "border-paper-line bg-white"} ${disabled ? "opacity-60" : ""}`}
      data-testid="dropzone"
    >
      <input
        ref={input}
        type="file"
        className="sr-only"
        aria-label="Choose an image or video file"
        accept={[...limits.image_formats, ...limits.video_formats].join(",")}
        disabled={disabled}
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = "";
        }}
      />
      <svg aria-hidden="true" viewBox="0 0 24 24" className="mx-auto mb-3 h-10 w-10 text-accent" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L8 8m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" />
      </svg>
      <p className="font-medium">Drag and drop an image or video here</p>
      <p className="mt-1 text-sm text-ink-mute">
        Images up to {limits.max_image_mb} MB, videos up to {limits.max_video_mb} MB
      </p>
      <button type="button" className="btn-secondary mt-4" disabled={disabled} onClick={() => input.current?.click()}>
        Browse files
      </button>
      <p className="mt-3 text-xs text-ink-mute">
        {limits.image_formats.join(" ")} &middot; {limits.video_formats.join(" ")}
      </p>
    </div>
  );
}
