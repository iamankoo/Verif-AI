import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UploadZone } from "./UploadZone";

const setup = (disabled = false) => {
  const onSelect = vi.fn(), onError = vi.fn();
  render(<UploadZone onSelect={onSelect} onError={onError} disabled={disabled} />);
  return { onSelect, onError, input: screen.getByLabelText(/choose an image or video/i) as HTMLInputElement };
};

describe("UploadZone", () => {
  it("selects a valid image via the file input", async () => {
    const { onSelect, onError, input } = setup();
    const file = new File(["x"], "face.png", { type: "image/png" });
    await userEvent.upload(input, file);
    expect(onSelect).toHaveBeenCalledWith(file, "image");
    expect(onError).not.toHaveBeenCalled();
  });

  it("selects a video", async () => {
    const { onSelect, input } = setup();
    const file = new File(["x"], "clip.mp4", { type: "video/mp4" });
    await userEvent.upload(input, file, { applyAccept: false });
    expect(onSelect).toHaveBeenCalledWith(file, "video");
  });

  it("reports unsupported files through onError", async () => {
    const { onSelect, onError, input } = setup();
    await userEvent.upload(input, new File(["x"], "virus.exe"), { applyAccept: false });
    expect(onSelect).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(expect.stringMatching(/Unsupported file type/));
  });

  it("accepts drag-and-drop", () => {
    const { onSelect } = setup();
    const file = new File(["x"], "face.jpg", { type: "image/jpeg" });
    const zone = screen.getByTestId("dropzone");
    fireEvent.dragOver(zone);
    fireEvent.drop(zone, { dataTransfer: { files: [file] } });
    expect(onSelect).toHaveBeenCalledWith(file, "image");
  });

  it("ignores drops when disabled", () => {
    const { onSelect } = setup(true);
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [new File(["x"], "a.png")] } });
    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /browse files/i })).toBeDisabled();
  });
});
