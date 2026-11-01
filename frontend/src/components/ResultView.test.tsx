import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FAKE_IMAGE, FAKE_VIDEO, ID, INCONCLUSIVE_IMAGE, REAL_IMAGE } from "../test/fixtures";
import { ResultView } from "./ResultView";

const show = (r = FAKE_IMAGE, extra: Partial<Parameters<typeof ResultView>[0]> = {}) => render(<ResultView result={r} onNew={() => undefined} {...extra} />);

describe("ResultView - image", () => {
  it("renders a FAKE verdict with both scores and the uncertainty note", () => {
    show();
    expect(screen.getByTestId("result-view").querySelector("[data-testid=verdict]")).toHaveTextContent("FAKE");
    expect(screen.getByTestId("fake-score")).toHaveTextContent("0.930");
    expect(screen.getByTestId("real-score")).toHaveTextContent("0.070");
    expect(screen.getByText(/not a probability that the verdict is correct/i)).toBeInTheDocument();
    expect(screen.getByText(/NOT calibrated/)).toBeInTheDocument();
  });

  it("lists each face with its own verdict and shows Grad-CAM pairs", () => {
    show();
    const rows = within(screen.getByRole("table")).getAllByRole("row");
    expect(rows).toHaveLength(3);                                  // header + 2 faces
    expect(within(rows[1]).getByText("FAKE")).toBeInTheDocument();
    expect(within(rows[2]).getByText("REAL")).toBeInTheDocument();
    expect(screen.getByAltText("Face 1 Grad-CAM heat-map")).toHaveAttribute("src", `/api/analyses/${ID}/artifacts/face_0_heatmap.jpg`);
    expect(screen.getByAltText(/image with face bounding boxes/i)).toBeInTheDocument();
  });

  it("shows warnings in an alert and always shows scope/limitations", () => {
    show();
    expect(screen.getByRole("alert")).toHaveTextContent("2 faces detected");
    expect(screen.getByText(/not definitive proof/i)).toBeInTheDocument();
    expect(screen.getByText(/A REAL result does not prove authenticity/i)).toBeInTheDocument();
  });

  it("links the PDF forensic report", () => {
    show();
    const link = screen.getByRole("link", { name: /download forensic report/i });
    expect(link).toHaveAttribute("href", `/api/analyses/${ID}/report.pdf`);
    expect(link).toHaveAttribute("download");
  });

  it("renders REAL honestly (no claim of authenticity)", () => {
    show(REAL_IMAGE);
    expect(screen.getAllByTestId("verdict")[0]).toHaveTextContent("REAL");        // header badge (face rows repeat the badge)
    expect(screen.getAllByText(/does not prove authenticity/i).length).toBeGreaterThanOrEqual(2);   // verdict blurb + limitations
  });

  it("renders INCONCLUSIVE with no scores when no face was found", () => {
    show(INCONCLUSIVE_IMAGE);
    expect(screen.getAllByTestId("verdict")[0]).toHaveTextContent("INCONCLUSIVE");
    expect(screen.getByText(/No score could be produced/i)).toBeInTheDocument();
    expect(screen.queryByTestId("fake-score")).toBeNull();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("wires New analysis and Delete callbacks", async () => {
    const onNew = vi.fn(), onDelete = vi.fn();
    render(<ResultView result={FAKE_IMAGE} onNew={onNew} onDelete={onDelete} />);
    await userEvent.click(screen.getByRole("button", { name: /new analysis/i }));
    await userEvent.click(screen.getByRole("button", { name: /delete results/i }));
    expect(onNew).toHaveBeenCalled();
    expect(onDelete).toHaveBeenCalled();
  });
});

describe("ResultView - video", () => {
  it("renders summary, timeline, suspicious timestamps, tracks and evidence", () => {
    show(FAKE_VIDEO);
    expect(screen.getByTestId("verdict")).toHaveTextContent("FAKE");
    expect(screen.getByRole("img", { name: /timeline over 6.0 seconds/i })).toBeInTheDocument();
    const seg = within(screen.getByTestId("segments")).getAllByRole("listitem");
    expect(seg).toHaveLength(1);
    expect(seg[0]).toHaveTextContent("2.0s - 6.0s");
    expect(screen.getByText("1280x720")).toBeInTheDocument();
    expect(screen.getByText(/not been validated on labelled videos/i)).toBeInTheDocument();
    expect(screen.getByAltText(/Evidence frame at 4.0 seconds/)).toHaveAttribute("src", `/api/analyses/${ID}/artifacts/evidence_0.jpg`);
    expect(screen.getByText(/75% of frames above 0.5/)).toBeInTheDocument();
  });

  it("says so when no frame is suspicious", () => {
    const calm = { ...FAKE_VIDEO, classification: "REAL" as const, video: { ...FAKE_VIDEO.video!, suspicious_segments: [] } };
    show(calm);
    expect(screen.getByText(/No analysed frame scored above/i)).toBeInTheDocument();
  });
});
