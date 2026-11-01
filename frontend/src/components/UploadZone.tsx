import { useRef, useState } from "react";
import type { Limits } from "../types";
import { DEFAULT_LIMITS, validateFile, type Kind } from "../lib/format";
import { IconDatabase, IconImage, IconUpload, IconVideo } from "./Icons";

interface Props {
  limits?: Pick<Limits, "max_image_mb" | "max_video_mb" | "image_formats" | "video_formats">;
  disabled?: boolean;
  onSelect: (file: File, kind: Kind) => void;
  onError: (message: string) => void;
}

const names = (exts: string[]) => exts.map((e) => e.replace(".", "").toUpperCase()).join(", ");

/** One drop zone for images and videos. The kind is decided from the file extension (the server re-validates the content). */
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
      className={`rounded-2xl border-2 border-dashed px-4 py-8 text-center transition sm:px-8 ${over ? "border-accent bg-accent-soft" : "border-accent/50 bg-paper/40"} ${disabled ? "opacity-60" : ""}`}
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
      <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl border border-accent/40 bg-accent-soft text-accent"><IconUpload className="h-8 w-8" /></span>
      <h2 className="mt-3 text-2xl font-bold">Upload Media</h2>
      <p className="mt-1 text-ink-soft">Drag and drop an image or video, or browse your device.</p>
      <button type="button" className="btn-primary mt-5 px-8 py-3 text-base" disabled={disabled} onClick={() => input.current?.click()}>
        <IconUpload className="h-5 w-5" /> Choose File
      </button>
      <ul className="mx-auto mt-6 flex max-w-4xl flex-wrap items-center justify-center gap-x-5 gap-y-2 rounded-xl border border-paper-line bg-paper-raised/60 px-4 py-2.5 text-xs text-ink-soft sm:text-sm">
        <li className="flex items-center gap-2"><IconImage className="h-4 w-4 text-accent" /> Images: {names(limits.image_formats)}</li>
        <li className="flex items-center gap-2"><IconVideo className="h-4 w-4 text-accent" /> Videos: {names(limits.video_formats)}</li>
        <li className="flex items-center gap-2"><IconDatabase className="h-4 w-4 text-accent" /> Max size: {limits.max_image_mb} MB (images), {limits.max_video_mb} MB (videos)</li>
      </ul>
    </div>
  );
}
