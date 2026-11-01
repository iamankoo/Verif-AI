export type Classification = "REAL" | "FAKE" | "INCONCLUSIVE";
export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface ModelInfo {
  name: string;
  version: string;
  architecture: string;
  checkpoint_file: string;
  checkpoint_sha256: string;
  parameters: number;
  device: string;
  input_size: number;
  decision_threshold: number;
  trained_on?: string | null;
  training_notes?: string | null;
  metrics_note?: string | null;
}

export interface Limits {
  max_image_mb: number;
  max_video_mb: number;
  image_formats: string[];
  video_formats: string[];
  video_sample_fps: number;
  video_max_frames: number;
  video_max_duration_s: number;
  uncertainty_margin: number;
  decision_threshold: number;
  video_aggregation: string;
  retention_hours: number;
}

export interface ModelResponse {
  model: ModelInfo;
  scope_note: string;
  limits: Limits;
  explainability: { grad_cam_available: boolean };
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  device: string;
  version: string;
  detail?: string | null;
}

export interface FaceResult {
  index: number;
  box: number[];
  detection_confidence: number;
  fake_probability: number;
  real_probability: number;
  label: Classification;
  crop_artifact?: string | null;
  heatmap_artifact?: string | null;
}

export interface FrameFace {
  track_id: number;
  box: number[];
  detection_confidence: number;
  fake_probability: number;
}
export interface FrameResult {
  frame_index: number;
  timestamp_s: number;
  faces: FrameFace[];
  frame_fake_probability: number | null;
  suspicious: boolean;
}
export interface TrackSummary {
  track_id: number;
  frames: number;
  first_timestamp_s: number;
  last_timestamp_s: number;
  median_fake_probability: number;
  mean_fake_probability: number;
  suspicious_fraction: number;
}
export interface SuspiciousSegment {
  start_s: number;
  end_s: number;
  frames: number;
  max_fake_probability: number;
}
export interface EvidenceFrame {
  frame_index: number;
  timestamp_s: number;
  fake_probability: number;
  frame_artifact: string;
  heatmap_artifact?: string | null;
}
export interface TemporalStats {
  score_std: number;
  mean_abs_frame_delta: number;
  longest_suspicious_run: number;
  suspicious_fraction: number;
  sign_flips: number;
}
export interface VideoSummary {
  duration_s: number;
  fps: number;
  width: number;
  height: number;
  total_frames_reported: number;
  frames_sampled: number;
  frames_with_faces: number;
  corrupt_frames_skipped: number;
  sampling_fps: number;
  aggregation_method: string;
  aggregation_validated: boolean;
  frames: FrameResult[];
  tracks: TrackSummary[];
  suspicious_segments: SuspiciousSegment[];
  evidence_frames: EvidenceFrame[];
  temporal?: TemporalStats | null;
}

export interface AnalysisResult {
  analysis_id: string;
  status: JobStatus;
  media_type: "image" | "video";
  filename: string;
  file_size_bytes: number;
  file_sha256?: string | null;
  created_at: string;
  completed_at?: string | null;
  classification: Classification | null;
  real_probability: number | null;
  fake_probability: number | null;
  score_semantics: string;
  decisiveness: number | null;
  decisiveness_label: "low" | "moderate" | "high" | null;
  faces_detected: number;
  faces: FaceResult[];
  annotated_artifact?: string | null;
  video?: VideoSummary | null;
  explanation: string[];
  warnings: string[];
  scope_note: string;
  model?: ModelInfo | null;
  processing_time_s?: number | null;
  image_width?: number | null;
  image_height?: number | null;
  aggregation_note?: string | null;
}

export interface JobState {
  analysis_id: string;
  status: JobStatus;
  media_type: "image" | "video";
  filename: string;
  progress: number;
  stage: string;
  created_at: string;
  error?: string | null;
}

export interface AnalysisSummary {
  analysis_id: string;
  media_type: string;
  filename: string;
  status: JobStatus;
  classification: Classification | null;
  fake_probability: number | null;
  created_at: string;
}
