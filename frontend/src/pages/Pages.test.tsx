import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api";
import { FAKE_VIDEO, ID, MODEL, REAL_IMAGE } from "../test/fixtures";
import { AnalysisPage } from "./AnalysisPage";
import { Home } from "./Home";

vi.mock("../api", async (orig) => {
  const real = await orig<typeof import("../api")>();
  return {
    ...real,
    api: {
      health: vi.fn(), model: vi.fn(), models: vi.fn(), list: vi.fn(), analyzeImage: vi.fn(), analyzeVideo: vi.fn(), status: vi.fn(), result: vi.fn(), remove: vi.fn(),
      reportUrl: (id: string) => `/api/analyses/${id}/report.pdf`, artifactUrl: (id: string, n: string) => `/api/analyses/${id}/artifacts/${n}`,
    },
  };
});
import { api } from "../api";
const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

beforeEach(() => {
  vi.clearAllMocks();
  m.health.mockResolvedValue({ status: "ok", model_loaded: true, device: "cpu", version: "1" });
  m.model.mockResolvedValue(MODEL);
  m.list.mockResolvedValue([]);
  m.models.mockResolvedValue(undefined);          // servers without /models: no selector
});

const upload = async (name: string, type: string) => {
  const input = await screen.findByLabelText(/choose an image or video/i);
  await userEvent.upload(input, new File(["bytes"], name, { type }), { applyAccept: false });
};

describe("Home", () => {
  it("shows model information from the API (no placeholder statistics)", async () => {
    render(<Home onOpen={() => undefined} />);
    expect(await screen.findByTestId("model-card")).toHaveTextContent("verifai-two-stream-test v9.9");
    expect(screen.getByText(/No analyses yet/)).toBeInTheDocument();
    expect(screen.getByText("Scope note for tests.")).toBeInTheDocument();
  });

  it("uploads an image, then opens the stored analysis", async () => {
    m.analyzeImage.mockResolvedValue(REAL_IMAGE);
    const onOpen = vi.fn();
    render(<Home onOpen={onOpen} />);
    await upload("face.png", "image/png");
    expect(await screen.findByTestId("preview")).toHaveTextContent("face.png");
    await userEvent.click(screen.getByRole("button", { name: /analyse image/i }));
    await waitFor(() => expect(onOpen).toHaveBeenCalledWith(ID));
    expect(m.analyzeImage).toHaveBeenCalledTimes(1);
  });

  describe("model selection", () => {
    const LIST = {
      default: "model4", selection_enabled: true,
      models: [
        { id: "model4", label: "Model 4", role: "production", role_label: "Production default", description: "", checkpoint_file: "m4.pth", available: true, selectable: true, is_default: true, loaded: true },
        { id: "v2", label: "v2", role: "rollback", role_label: "Rollback model", description: "", checkpoint_file: "v2.pth", available: true, selectable: true, is_default: false, loaded: false },
        { id: "e4", label: "E4", role: "candidate", role_label: "Candidate - not approved for production", description: "", checkpoint_file: "e4.pth", available: true, selectable: true, is_default: false, loaded: false },
      ],
    };

    it("shows no selector when the server offers only one model", async () => {
      m.models.mockResolvedValue({ ...LIST, selection_enabled: false });
      render(<Home onOpen={() => undefined} />);
      await upload("face.png", "image/png");
      expect(await screen.findByTestId("preview")).toBeInTheDocument();
      expect(screen.queryByTestId("model-select")).not.toBeInTheDocument();
    });

    it("keeps the production default selected and sends no model id for it", async () => {
      m.models.mockResolvedValue(LIST);
      m.analyzeImage.mockResolvedValue(REAL_IMAGE);
      render(<Home onOpen={() => undefined} />);
      await upload("face.png", "image/png");
      const sel = (await screen.findByTestId("model-select")) as HTMLSelectElement;
      expect(sel.value).toBe("model4");
      expect(screen.queryByTestId("candidate-note")).not.toBeInTheDocument();
      await userEvent.click(screen.getByRole("button", { name: /analyse image/i }));
      await waitFor(() => expect(m.analyzeImage).toHaveBeenCalledTimes(1));
      expect(m.analyzeImage.mock.calls[0][2]).toBeUndefined();
    });

    it("labels the candidate, warns that it is not approved and sends its id", async () => {
      m.models.mockResolvedValue(LIST);
      m.analyzeImage.mockResolvedValue(REAL_IMAGE);
      render(<Home onOpen={() => undefined} />);
      await upload("face.png", "image/png");
      const sel = await screen.findByTestId("model-select");
      expect(sel).toHaveTextContent(/E4 - Candidate - not approved for production/);
      await userEvent.selectOptions(sel, "e4");
      expect(screen.getByTestId("candidate-note")).toHaveTextContent(/not approved for production/i);
      await userEvent.click(screen.getByRole("button", { name: /analyse image/i }));
      await waitFor(() => expect(m.analyzeImage).toHaveBeenCalledTimes(1));
      expect(m.analyzeImage.mock.calls[0][2]).toBe("e4");
    });
  });

  it("submits a video as a background job", async () => {
    m.analyzeVideo.mockResolvedValue({ analysis_id: ID, status: "queued", media_type: "video", filename: "c.mp4", progress: 0, stage: "queued", created_at: "" });
    const onOpen = vi.fn();
    render(<Home onOpen={onOpen} />);
    await upload("c.mp4", "video/mp4");
    await userEvent.click(await screen.findByRole("button", { name: /analyse video/i }));
    await waitFor(() => expect(onOpen).toHaveBeenCalledWith(ID));
  });

  it("shows the server's error message when analysis is rejected", async () => {
    m.analyzeImage.mockRejectedValue(new ApiError("File is not a readable image (UnidentifiedImageError).", 422, "unreadable_image"));
    render(<Home onOpen={() => undefined} />);
    await upload("bad.png", "image/png");
    await userEvent.click(await screen.findByRole("button", { name: /analyse image/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("not a readable image");
    expect(screen.getByRole("button", { name: /analyse image/i })).toBeEnabled();          // user can retry
  });

  it("rejects an unsupported file client-side without calling the API", async () => {
    render(<Home onOpen={() => undefined} />);
    await upload("notes.txt", "text/plain");
    expect(await screen.findByRole("alert")).toHaveTextContent(/Unsupported file type/);
    expect(m.analyzeImage).not.toHaveBeenCalled();
  });

  it("disables upload and explains when the model is not loaded", async () => {
    m.health.mockResolvedValue({ status: "degraded", model_loaded: false, device: "n/a", version: "1", detail: "FileNotFoundError: no checkpoint" });
    render(<Home onOpen={() => undefined} />);
    expect(await screen.findByText(/model is not loaded/i)).toBeInTheDocument();
    expect(screen.getByText(/no checkpoint/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /choose file/i })).toBeDisabled();
  });

  it("tells the user when the backend is unreachable", async () => {
    m.health.mockRejectedValue(new Error("network"));
    render(<Home onOpen={() => undefined} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the Verif-AI server/);
  });

  it("lists history and opens an entry", async () => {
    m.list.mockResolvedValue([{ analysis_id: ID, media_type: "image", filename: "old.png", status: "completed", classification: "REAL", fake_probability: 0.1, created_at: "2026-11-01T10:00:00Z" }]);
    const onOpen = vi.fn();
    render(<Home onOpen={onOpen} />);
    await userEvent.click(await screen.findByRole("button", { name: /old\.png/ }));
    expect(onOpen).toHaveBeenCalledWith(ID);
  });
});

describe("AnalysisPage", () => {
  it("polls a running video job with progress, then renders the result", async () => {
    m.status
      .mockResolvedValueOnce({ analysis_id: ID, status: "running", media_type: "video", filename: "clip.mp4", progress: 0.4, stage: "analysing", created_at: "" })
      .mockResolvedValue({ analysis_id: ID, status: "completed", media_type: "video", filename: "clip.mp4", progress: 1, stage: "done", created_at: "" });
    m.result.mockResolvedValue(FAKE_VIDEO);
    render(<AnalysisPage id={ID} onHome={() => undefined} pollMs={10} />);
    expect(await screen.findByRole("progressbar")).toHaveAttribute("aria-valuenow", "40");
    expect(screen.getByText(/Stage: analysing/)).toBeInTheDocument();
    expect(await screen.findByTestId("result-view")).toBeInTheDocument();
    expect(screen.getByTestId("verdict")).toHaveTextContent("FAKE");
  });

  it("shows the failure reason for a failed job", async () => {
    m.status.mockResolvedValue({ analysis_id: ID, status: "failed", media_type: "video", filename: "x.mp4", progress: 0, stage: "failed", created_at: "", error: "The file could not be opened as a video." });
    render(<AnalysisPage id={ID} onHome={() => undefined} pollMs={10} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("could not be opened as a video");
  });

  it("explains an expired / deleted analysis", async () => {
    m.status.mockRejectedValue(new ApiError("Unknown analysis id.", 404));
    render(<AnalysisPage id={ID} onHome={() => undefined} pollMs={10} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/no longer exists/);
  });

  it("deletes results and returns home", async () => {
    m.status.mockResolvedValue({ analysis_id: ID, status: "completed", media_type: "image", filename: "a.png", progress: 1, stage: "done", created_at: "" });
    m.result.mockResolvedValue(REAL_IMAGE);
    m.remove.mockResolvedValue(undefined);
    const onHome = vi.fn();
    render(<AnalysisPage id={ID} onHome={onHome} pollMs={10} />);
    await userEvent.click(await screen.findByRole("button", { name: /delete results/i }));
    await waitFor(() => expect(onHome).toHaveBeenCalled());
    expect(m.remove).toHaveBeenCalledWith(ID);
  });
});
