import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { FAKE_IMAGE, FAKE_VIDEO, ID } from "../test/fixtures";
import type { AnalysisResult } from "../types";
import { FaceHeatmapViewer, FrameEvidenceViewer, GRADCAM_NOTE, pickFrameForRange } from "./HeatmapViewer";

const art = (n?: string | null) => (n ? `/api/analyses/${ID}/artifacts/${n}` : undefined);
const A = (n: string) => `/api/analyses/${ID}/artifacts/${n}`;

function FaceHarness({ result = FAKE_IMAGE }: { result?: AnalysisResult }) {
  const [i, setI] = useState(0);
  return <FaceHeatmapViewer result={result} art={art} faceIndex={i} onFace={setI} />;
}
function FrameHarness({ start = 100 }: { start?: number }) {
  const [sel, setSel] = useState<number | null>(start);
  return <FrameEvidenceViewer video={FAKE_VIDEO.video!} art={art} selected={sel} onSelect={setSel} />;
}

describe("FaceHeatmapViewer (image)", () => {
  it("shows the original crop and the Grad-CAM side by side, from the real artifacts", () => {
    render(<FaceHarness />);
    expect(screen.getByAltText("Face #1 original face crop")).toHaveAttribute("src", A("face_0_crop.jpg"));
    expect(screen.getByAltText("Face #1 Grad-CAM heat-map")).toHaveAttribute("src", A("face_0_heatmap.jpg"));
    expect(screen.getByText(GRADCAM_NOTE)).toBeInTheDocument();                 // integrity note: not proof
  });

  it("switches between Original, Heatmap and Overlay", async () => {
    render(<FaceHarness />);
    await userEvent.click(screen.getByRole("button", { name: "Original" }));
    expect(screen.getByAltText("Face #1 original face crop")).toBeInTheDocument();
    expect(screen.queryByAltText("Face #1 Grad-CAM heat-map")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Heatmap" }));
    expect(screen.getByAltText("Face #1 Grad-CAM heat-map")).toBeInTheDocument();
    expect(screen.queryByAltText("Face #1 original face crop")).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: "Overlay" }));
    const top = screen.getByTestId("overlay-top");
    expect(top).toHaveAttribute("src", A("face_0_heatmap.jpg"));
    expect(top).toHaveStyle({ opacity: "0.6" });
    fireEvent.change(screen.getByRole("slider", { name: /heat-map strength/i }), { target: { value: "20" } });
    expect(screen.getByTestId("overlay-top")).toHaveStyle({ opacity: "0.2" });
    expect(screen.getByRole("button", { name: "Overlay" })).toHaveAttribute("aria-pressed", "true");
  });

  it("makes clear which face is shown and switches faces when several are detected", async () => {
    render(<FaceHarness />);
    expect(screen.getByText("Face #1", { selector: "strong" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Face #2/ }));
    expect(screen.getByText("Face #2", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByAltText("Face #2 Grad-CAM heat-map")).toHaveAttribute("src", A("face_1_heatmap.jpg"));
    expect(screen.getByAltText("Face #2 original face crop")).toHaveAttribute("src", A("face_1_crop.jpg"));
  });

  it("says so, and invents nothing, when a face has no heat-map", () => {
    const r: AnalysisResult = { ...FAKE_IMAGE, faces: [{ ...FAKE_IMAGE.faces[0], heatmap_artifact: null }] };
    render(<FaceHarness result={r} />);
    expect(screen.getByTestId("heatmap-unavailable")).toHaveTextContent(/not available/i);
    expect(screen.queryByAltText(/Grad-CAM heat-map/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Overlay" })).toBeNull();
    expect(screen.getByAltText("Face #1 original face crop")).toBeInTheDocument();
  });

  it("explains when no face crop was stored at all", () => {
    render(<FaceHarness result={{ ...FAKE_IMAGE, faces: [] }} />);
    expect(screen.getByTestId("heatmap-unavailable")).toBeInTheDocument();
  });
});

describe("FrameEvidenceViewer (video)", () => {
  it("shows the stored frame next to its own heat-map, for the highest-scoring face", () => {
    render(<FrameHarness />);
    expect(screen.getByAltText("Evidence frame at 4.0 seconds")).toHaveAttribute("src", A("evidence_0.jpg"));
    expect(screen.getByAltText("Grad-CAM heat-map at 4.0 seconds")).toHaveAttribute("src", A("evidence_0_heatmap.jpg"));
    expect(screen.getByText(/highest-scoring face in this frame \(track #1/)).toBeInTheDocument();
    expect(screen.getByTestId("frame-position")).toHaveTextContent("Frame 100");
  });

  it("is honest for frames without stored evidence and offers the nearest stored frame", async () => {
    render(<FrameHarness start={50} />);
    expect(screen.getByTestId("heatmap-unavailable")).toHaveTextContent(/No frame image or heat-map was stored/);
    expect(screen.queryByAltText(/Grad-CAM heat-map/)).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /Jump to the nearest stored frame/ }));
    expect(screen.getByAltText("Grad-CAM heat-map at 4.0 seconds")).toBeInTheDocument();
  });

  it("keeps frame and heat-map in step while navigating previous / next", async () => {
    render(<FrameHarness start={100} />);
    await userEvent.click(screen.getByRole("button", { name: /next analysed frame/i }));
    expect(screen.getByTestId("frame-position")).toHaveTextContent("Frame 150");
    expect(screen.queryByAltText(/Grad-CAM heat-map/)).toBeNull();               // frame 150 has none stored: heat-map must disappear
    await userEvent.click(screen.getByRole("button", { name: /previous analysed frame/i }));
    await userEvent.click(screen.getByRole("button", { name: /previous analysed frame/i }));
    expect(screen.getByTestId("frame-position")).toHaveTextContent("Frame 50");
    await userEvent.click(screen.getByRole("button", { name: /next analysed frame/i }));
    expect(screen.getByAltText("Evidence frame at 4.0 seconds")).toBeInTheDocument();
    expect(screen.getByAltText("Grad-CAM heat-map at 4.0 seconds")).toBeInTheDocument();
  });

  it("disables the overlay mode for video frames and explains why", () => {
    render(<FrameHarness />);
    const o = screen.getByRole("button", { name: "Overlay" });
    expect(o).toBeDisabled();
    expect(o).toHaveAttribute("title", expect.stringMatching(/approximation/));
  });

  it("supports Original and Heatmap views for a stored frame", async () => {
    render(<FrameHarness />);
    await userEvent.click(screen.getByRole("button", { name: "Heatmap" }));
    expect(screen.queryByAltText("Evidence frame at 4.0 seconds")).toBeNull();
    expect(screen.getByAltText("Grad-CAM heat-map at 4.0 seconds")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Original" }));
    expect(screen.getByAltText("Evidence frame at 4.0 seconds")).toBeInTheDocument();
    expect(screen.queryByAltText(/Grad-CAM heat-map/)).toBeNull();
  });

  it("handles a frame whose evidence has no heat-map", () => {
    const v = { ...FAKE_VIDEO.video!, evidence_frames: [{ ...FAKE_VIDEO.video!.evidence_frames[0], heatmap_artifact: null }] };
    render(<FrameEvidenceViewer video={v} art={art} selected={100} onSelect={() => undefined} />);
    expect(screen.getByTestId("heatmap-unavailable")).toHaveTextContent(/heat-map is not available for this frame/i);
    expect(screen.getByAltText("Evidence frame at 4.0 seconds")).toBeInTheDocument();
  });
});

describe("pickFrameForRange", () => {
  const v = FAKE_VIDEO.video!;
  it("prefers the stored evidence frame inside the range", () => {
    expect(pickFrameForRange(v, { start_s: 2, end_s: 6 })).toBe(100);
  });
  it("falls back to the highest-scoring analysed frame, and to null when the range is empty", () => {
    expect(pickFrameForRange(v, { start_s: 0, end_s: 1 })).toBe(0);
    expect(pickFrameForRange(v, { start_s: 7, end_s: 9 })).toBeNull();
  });
});

describe("timeline selection", () => {
  it("syncs the viewer when a timeline dot or a suspicious timestamp is chosen (via ResultView)", async () => {
    const { ResultView } = await import("./ResultView");
    render(<ResultView result={FAKE_VIDEO} onNew={() => undefined} />);
    await userEvent.click(screen.getByRole("tab", { name: /frame analysis/i }));
    const viewer = screen.getByTestId("frame-evidence-viewer");
    expect(within(viewer).getByTestId("frame-position")).toHaveTextContent("Frame 100");
    await userEvent.click(screen.getByTestId("tl-dot-50"));
    expect(within(viewer).getByTestId("frame-position")).toHaveTextContent("Frame 50");
    expect(within(viewer).getByTestId("heatmap-unavailable")).toBeInTheDocument();
    await userEvent.click(within(screen.getByTestId("segments")).getByRole("button"));
    expect(within(viewer).getByTestId("frame-position")).toHaveTextContent("Frame 100");
    expect(within(viewer).getByAltText("Grad-CAM heat-map at 4.0 seconds")).toBeInTheDocument();
  });
});
