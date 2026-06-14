// Shapes mirroring the backend domain models (app/models.py).

export type ProjectStatus = "created" | "queued" | "processing" | "ready" | "failed";
export type ClipStatus = "pending" | "rendering" | "ready" | "failed";
export type Platform = "tiktok" | "reels" | "shorts" | "generic";

export interface ScoreFactor {
  label: string;
  weight: number;
  detail: string;
}

export interface CaptionWord {
  t: number;
  d: number;
  text: string;
  speaker: number | null;
}

export interface DetectedEvent {
  t: number;
  source: "cue" | "ocr";
  label: string;
  detail: string;
  confidence: number;
}

export interface CaptionSet {
  id: string;
  words: CaptionWord[];
  style_id: string;
  max_words_per_line: number;
}

export interface ReframeKeyframe {
  t: number;
  cx: number;
}

// Region of the source frame, all values fractions [0,1].
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type Layout = "fill" | "center" | "split" | "framed";

export interface Reframe {
  layout: Layout;
  keyframes: ReframeKeyframe[];
  tracked: boolean;
  overridden: boolean;
  cx_overridden: boolean;
  facecam: Rect | null;
}

export interface Clip {
  id: string;
  start: number;
  end: number;
  title: string;
  kind: string;
  score: number;
  factors: ScoreFactor[];
  hashtags: string[];
  feedback: "up" | "down" | null;
  transcript_excerpt: string;
  captions: CaptionSet;
  reframe: Reframe;
  aspect: string | null;
  status: ClipStatus;
  // Diarized speakers present in this clip, and which are kept in captions
  // (null = all). Drives the editor's per-speaker caption toggles.
  speakers: number[];
  caption_speakers: number[] | null;
  tightened_duration: number | null;
  export_url: string | null;
  thumb_url: string | null;
  error: string | null;
}

export interface StageView {
  name: string;
  label: string;
  status: "pending" | "active" | "done";
  pct: number;
}

export interface JobProgress {
  stage: string;
  stage_index: number;
  total_stages: number;
  message: string;
  pct: number;
  stages: StageView[];
  updated_at: number;
}

export interface SourceMedia {
  filename: string;
  path: string;
  url: string | null;
  duration: number;
  width: number;
  height: number;
  fps: number;
  size_bytes: number;
}

export interface ImportSettings {
  platform: Platform;
  min_len: number;
  max_len: number;
  target_clips: number;
  default_style_id: string;
  language: string;
  content_type: string;
  aspect: string;
  burn_captions: boolean;
  game_profile: string;
  tighten: boolean;
  motion: string;
  facecam_layout: string;
}

export interface Montage {
  id: string;
  title: string;
  clip_ids: string[];
  score: number;
  factors: ScoreFactor[];
  duration: number;
  status: ClipStatus;
  export_url: string | null;
  thumb_url: string | null;
  error: string | null;
}

export interface Project {
  id: string;
  name: string;
  status: ProjectStatus;
  settings: ImportSettings;
  source: SourceMedia | null;
  transcript: { provider: string; language: string; words: unknown[]; speakers: number } | null;
  clips: Clip[];
  montages: Montage[];
  progress: JobProgress;
  content_type: string | null;
  facecam: Rect | null;
  events: DetectedEvent[];
  warnings: string[];
  error: string | null;
  created_at: number;
  updated_at: number;
}

export interface ProjectSummary {
  id: string;
  name: string;
  status: ProjectStatus;
  clip_count: number;
  ready_clips: number;
  duration: number;
  progress: JobProgress;
  created_at: number;
  updated_at: number;
}

export interface StatusPayload {
  id: string;
  status: ProjectStatus;
  error: string | null;
  warnings: string[];
  content_type: string | null;
  progress: JobProgress;
  clips: Array<
    Pick<Clip, "id" | "title" | "score" | "kind" | "status" | "thumb_url" | "export_url"> & {
      duration: number;
    }
  >;
}

export interface StyleTemplate {
  id: string;
  name: string;
  font: string;
  font_size: number;
  primary: string;
  highlight: string;
  outline: string;
  y_frac: number;
  uppercase: boolean;
}

export interface Health {
  ok: boolean;
  version: string;
  capabilities: {
    ffmpeg: boolean;
    ffprobe: boolean;
    transcription: string;
    diarization: boolean;
    ocr: string | false;
    face_tracking: boolean;
    url_import: boolean;
    gpu: boolean;
    gpu_encode: boolean;
    device: string;
    llm: boolean;
    llm_model: string | null;
    whisper_model: string;
    auto_model: boolean;
    vram_gb: number;
    cpu: number;
  };
  output: { width: number; height: number };
}
