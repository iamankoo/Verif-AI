import { useEffect, useState } from "react";
import { api } from "../api";
import { IconCheckCircle, IconChip, IconFile, IconLayers, IconScan, IconShield, IconUpload } from "../components/Icons";
import { ErrorBanner, HistoryList, ModelCard } from "../components/Panels";
import type { AnalysisSummary, Health, ModelResponse } from "../types";

const REPO = "https://github.com/iamankoo/Verif-AI";

function PageTitle({ title, lead }: { title: string; lead: string }) {
  return (
    <header className="mb-6">
      <h1 className="text-3xl font-bold sm:text-4xl">{title}</h1>
      <p className="mt-2 max-w-3xl text-ink-soft">{lead}</p>
    </header>
  );
}

export function HowItWorksPage() {
  const [info, setInfo] = useState<ModelResponse | null>(null);
  useEffect(() => {
    let alive = true;
    api.model().then((m) => alive && setInfo(m)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);
  const L = info?.limits;
  const steps = [
    { Icon: IconUpload, title: "1. Upload and validate", body: `The file type and size are checked in the browser and again on the server (images up to ${L?.max_image_mb ?? 15} MB, videos up to ${L?.max_video_mb ?? 200} MB). The upload is decoded to confirm it really is an image or video, analysed, and then deleted.` },
    { Icon: IconScan, title: "2. Find faces", body: "An MTCNN face detector looks for faces. Only faces that are detected with high confidence and are large enough are analysed (up to 10 per image); an image or video with no usable face is reported as INCONCLUSIVE, not as real or fake." },
    { Icon: IconLayers, title: "3. Prepare each face", body: "Each face is cropped to 224 x 224 pixels with a margin, passed through the same JPEG round-trip used when the training crops were stored, and normalised exactly as during training." },
    { Icon: IconChip, title: "4. Two-stream model", body: "One EfficientNet-B0 looks at the normal RGB face; a second looks at a noise residual produced by fixed high-pass (SRM-style) filters. Their features are combined into one fake-score between 0 and 1." },
    { Icon: IconCheckCircle, title: "5. Decide, with uncertainty", body: `A score above ${L?.decision_threshold ?? 0.5} is FAKE, otherwise REAL; a score within +/- ${L?.uncertainty_margin ?? 0.15} of that boundary is reported as INCONCLUSIVE. An image's score is its highest face score. For video, about ${L?.video_sample_fps ?? 2} frames per second (at most ${L?.video_max_frames ?? 64}) are scored, faces are tracked over time, each track's median score is taken and the highest track decides - a method that has not been validated on labelled videos.` },
    { Icon: IconFile, title: "6. Evidence and report", body: "Grad-CAM heat-maps show which regions of the face influenced the RGB stream, a timeline and suspicious time ranges are shown for videos, and a PDF forensic report records the result, the file and model hashes, the evidence and the limitations." },
  ];
  return (
    <div>
      <PageTitle title="How it works" lead="What happens to a file between upload and report - the steps below are the ones the software actually performs." />
      <ol className="grid gap-4 md:grid-cols-2">
        {steps.map(({ Icon, title, body }) => (
          <li key={title} className="card p-5">
            <div className="mb-2 flex items-center gap-3">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-accent-soft text-accent"><Icon className="h-5 w-5" /></span>
              <h2 className="text-lg">{title}</h2>
            </div>
            <p className="text-sm text-ink-soft">{body}</p>
          </li>
        ))}
      </ol>
      <p className="mt-6 text-xs text-ink-mute">Scores are uncalibrated model outputs, not probabilities, and a REAL result is not proof of authenticity.</p>
    </div>
  );
}

export function ModelPage() {
  const [info, setInfo] = useState<ModelResponse | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    api.health().then((h) => alive && setHealth(h)).catch(() => alive && setError("Cannot reach the Verif-AI server. Is the backend running?"));
    api.model().then((m) => alive && setInfo(m)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);
  return (
    <div>
      <PageTitle title="Model" lead="The exact model this server is running, read live from the API. Evaluation numbers are kept in the repository reports and are deliberately not repeated here, so this page can never show outdated statistics." />
      {error && <ErrorBanner message={error} />}
      <div className="grid gap-5 lg:grid-cols-2">
        <ModelCard info={info} health={health} />
        <section className="card p-5">
          <h2 className="mb-3 text-lg">What it can and cannot do</h2>
          <ul className="list-disc space-y-2 pl-5 text-sm text-ink-soft">
            <li>It analyses faces in images and videos; it is not a general detector of AI-generated or edited images.</li>
            <li>It was trained and evaluated on a limited set of data and generators; results on other generators, heavy compression or low-resolution footage can be wrong.</li>
            <li>Video-level accuracy has not been validated on labelled videos, and soft or compressed footage can produce false positives.</li>
            <li>Scores are uncalibrated; the threshold is fixed at {info?.limits.decision_threshold ?? 0.5} and an INCONCLUSIVE band is used for uncertain cases.</li>
          </ul>
          <p className="mt-4 text-sm">
            Evaluation reports: <a className="text-accent underline" href={`${REPO}/blob/main/docs/MODEL_EVALUATION_REPORT.md`} target="_blank" rel="noopener noreferrer">model evaluation report</a> and{" "}
            <a className="text-accent underline" href={`${REPO}/blob/main/docs/PROJECT_COMPLETION_REPORT.md`} target="_blank" rel="noopener noreferrer">completion report</a>.
          </p>
        </section>
      </div>
    </div>
  );
}

export function ReportsPage({ onOpen }: { onOpen: (id: string) => void }) {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    api.list().then((l) => alive && setItems(l)).catch(() => alive && setError("Cannot reach the Verif-AI server. Is the backend running?"));
    return () => {
      alive = false;
    };
  }, []);
  return (
    <div>
      <PageTitle title="Reports" lead="Analyses stored on this server. Open one to view its evidence and download its forensic PDF; results can be deleted at any time and expire automatically." />
      {error && <ErrorBanner message={error} />}
      <div className="max-w-3xl"><HistoryList items={items ?? []} onOpen={onOpen} /></div>
    </div>
  );
}

export function AboutPage() {
  return (
    <div>
      <PageTitle title="About Verif-AI" lead="A face-based synthetic-media detector with explainable evidence and forensic reports, running entirely on your own machine." />
      <div className="grid gap-5 md:grid-cols-2">
        <section className="card p-5">
          <h2 className="mb-2 flex items-center gap-2 text-lg"><IconShield className="h-5 w-5 text-accent" /> Privacy</h2>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-soft">
            <li>Everything runs on this server; nothing is sent to third-party AI services.</li>
            <li>Uploaded files are deleted right after analysis (also when it fails).</li>
            <li>Derived evidence images and results are kept for a limited time and can be deleted from the result page.</li>
          </ul>
        </section>
        <section className="card p-5">
          <h2 className="mb-2 flex items-center gap-2 text-lg"><IconScan className="h-5 w-5 text-accent" /> Scope and honesty</h2>
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-soft">
            <li>Face analysis only - not a universal AI-image or video detector.</li>
            <li>Outputs are model-based evidence, never proof; INCONCLUSIVE is a valid outcome.</li>
            <li>Known limitations are listed on every result page and in the repository documentation.</li>
          </ul>
        </section>
        <section className="card p-5 md:col-span-2">
          <h2 className="mb-2 text-lg">Project</h2>
          <p className="text-sm text-ink-soft">
            Source code, documentation and evaluation reports: <a className="text-accent underline" href={REPO} target="_blank" rel="noopener noreferrer">github.com/iamankoo/Verif-AI</a>. The original two-stream prototype was written by Lakshya Singh; this application, its training and evaluation tooling and its documentation build on it.
          </p>
        </section>
      </div>
    </div>
  );
}
