import type { AnalysisResult, ModelResponse } from "../types";

export const ID = "a".repeat(32);

export const MODEL: ModelResponse = {
  model: {
    name: "verifai-two-stream-test", version: "9.9", architecture: "Two-stream", checkpoint_file: "test.pth",
    checkpoint_sha256: "f".repeat(64), parameters: 11972954, device: "cpu", input_size: 224, decision_threshold: 0.5,
    trained_on: "unit-test data", metrics_note: "unit test",
  },
  scope_note: "Scope note for tests.",
  limits: {
    max_image_mb: 15, max_video_mb: 200, image_formats: [".jpg", ".jpeg", ".png"], video_formats: [".mp4", ".avi", ".mov"],
    video_sample_fps: 2, video_max_frames: 64, video_max_duration_s: 900, uncertainty_margin: 0.15, decision_threshold: 0.5,
    video_aggregation: "median", retention_hours: 72,
  },
  explainability: { grad_cam_available: true },
};

const base = {
  analysis_id: ID, status: "completed" as const, file_size_bytes: 123456, file_sha256: "b".repeat(64),
  created_at: "2026-11-01T10:00:00Z", completed_at: "2026-11-01T10:00:02Z", score_semantics: "Scores are NOT calibrated probabilities.",
  scope_note: "Scope: faces only.", model: MODEL.model, processing_time_s: 1.25, explanation: ["Explained."], warnings: [] as string[],
  image_width: 640, image_height: 480, aggregation_note: null, decisiveness: null, decisiveness_label: null,
};

export const FAKE_IMAGE: AnalysisResult = {
  ...base, media_type: "image", filename: "portrait.jpg", classification: "FAKE", fake_probability: 0.93, real_probability: 0.07,
  decisiveness: 0.86, decisiveness_label: "high", faces_detected: 2, annotated_artifact: "annotated.jpg",
  faces: [
    { index: 0, box: [10, 10, 100, 100], detection_confidence: 0.999, fake_probability: 0.93, real_probability: 0.07, label: "FAKE", crop_artifact: "face_0_crop.jpg", heatmap_artifact: "face_0_heatmap.jpg" },
    { index: 1, box: [200, 10, 300, 100], detection_confidence: 0.98, fake_probability: 0.04, real_probability: 0.96, label: "REAL", crop_artifact: "face_1_crop.jpg", heatmap_artifact: "face_1_heatmap.jpg" },
  ],
  warnings: ["2 faces detected; each was scored independently."],
};

export const REAL_IMAGE: AnalysisResult = {
  ...FAKE_IMAGE, classification: "REAL", fake_probability: 0.02, real_probability: 0.98, faces_detected: 1, warnings: [],
  faces: [FAKE_IMAGE.faces[1]],
};

export const INCONCLUSIVE_IMAGE: AnalysisResult = {
  ...base, media_type: "image", filename: "landscape.png", classification: "INCONCLUSIVE", fake_probability: null, real_probability: null,
  faces_detected: 0, faces: [], annotated_artifact: null, warnings: ["No face detected."],
};

export const FAKE_VIDEO: AnalysisResult = {
  ...base, media_type: "video", filename: "clip.mp4", classification: "FAKE", fake_probability: 0.88, real_probability: 0.12,
  decisiveness: 0.76, decisiveness_label: "high", faces_detected: 4, faces: [], aggregation_note: "Video score = max over tracks of the median.",
  video: {
    duration_s: 6, fps: 25, width: 1280, height: 720, total_frames_reported: 150, frames_sampled: 4, frames_with_faces: 4, corrupt_frames_skipped: 0,
    sampling_fps: 2, aggregation_method: "median", aggregation_validated: false,
    frames: [0, 1, 2, 3].map((i) => ({ frame_index: i * 50, timestamp_s: i * 2, faces: [{ track_id: 0, box: [1, 1, 2, 2], detection_confidence: 0.99, fake_probability: i === 0 ? 0.1 : 0.9 }], frame_fake_probability: i === 0 ? 0.1 : 0.9, suspicious: i !== 0 })),
    tracks: [{ track_id: 0, frames: 4, first_timestamp_s: 0, last_timestamp_s: 6, median_fake_probability: 0.9, mean_fake_probability: 0.7, suspicious_fraction: 0.75 }],
    suspicious_segments: [{ start_s: 2, end_s: 6, frames: 3, max_fake_probability: 0.9 }],
    evidence_frames: [{ frame_index: 100, timestamp_s: 4, fake_probability: 0.9, frame_artifact: "evidence_0.jpg", heatmap_artifact: "evidence_0_heatmap.jpg" }],
    temporal: { score_std: 0.34, mean_abs_frame_delta: 0.27, longest_suspicious_run: 3, suspicious_fraction: 0.75, sign_flips: 1 },
  },
};
