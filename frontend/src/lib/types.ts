// Shapes mirroring the backend domain models (app/models.py).

export type ProjectStatus = "created" | "queued" | "processing" | "paused" | "ready" | "failed";
export type ClipStatus = "pending" | "rendering" | "ready" | "failed";
export type Platform = "tiktok" | "reels" | "shorts" | "generic";
export type PowerMode = "balanced" | "max_gpu" | "quality";

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
  // Marked by the backend's keyword-emphasis pass (caption_fx.annotate). Honoured
  // at render time; the editor doesn't author these directly.
  emphasis?: boolean;
  emoji?: string | null;
}

export interface DetectedEvent {
  t: number;
  source: "cue" | "ocr" | "audio";
  label: string;
  detail: string;
  confidence: number;
}

export interface OcrRead {
  t: number;
  roi: string;
  text: string;
  confidence: number;
  matched: string[];
}

export interface OcrReport {
  enabled: boolean;
  engine: string;
  status: "skipped" | "unavailable" | "running" | "ran" | "failed" | string;
  frames_sampled: number;
  crops_read: number;
  cache_hits: number;
  texts_found: number;
  matches: number;
  warnings: string[];
  reads: OcrRead[];
}

export interface Notice {
  message: string;
  severity: "info" | "warn" | "error" | string;
  code?: string | null;
}

export interface CaptionSet {
  id: string;
  words: CaptionWord[];
  speech: number[][];
  style_id: string;
  max_words_per_line: number;
  // Spoken language ("en"/"de"); drives the lexicon for keyword emphasis + emoji.
  lang?: string;
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

export interface GameProfileConfig {
  detection_mode: "zero_shot" | "manual" | "hybrid" | string;
  visual_rois?: Rect[];
  visual_text_cues?: string[];
  reference_audio_files?: string[];
  vlm_visual_prompts?: string[];
  audio_prompts?: string[];
  audio_negative_prompts?: string[];
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
  status: "pending" | "active" | "paused" | "done";
  pct: number;
  /** Seconds the active stage has run / a completed stage took (optional). */
  elapsed_seconds?: number | null;
}

export interface JobProgress {
  stage: string;
  stage_index: number;
  total_stages: number;
  message: string;
  pct: number;
  stages: StageView[];
  started_at?: number | null;
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

export interface AiBoostSettings {
  /** Keyword emphasis: colour + enlarge power words for the whole line. */
  emphasis: boolean;
  /** Auto-emoji: 1-2 fitting emojis per line next to power words. */
  emoji: boolean;
  /** Speaker-aware caption colours (podcast look). */
  speakerColors: boolean;
  /** Auto zoom/punch-in on emphasis words and scene cuts. */
  autoZoom: boolean;
  /** B-roll smart cutaways during static voiceover spans. */
  broll: boolean;
  /** Hook/first-3s analysis with a warning + suggestion. */
  hookCheck: boolean;
}

export interface ImportSettings {
  platform: Platform;
  power_mode: PowerMode;
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
  denoise: boolean;
  motion: string;
  /** AI Boost — the viral-effect toggles grouped in the upload panel. */
  ai_boost: AiBoostSettings;
  facecam_layout: string;
  use_ocr: boolean;
  use_vlm: boolean;
  use_cues: boolean;
  use_audio_events: boolean;
  cue_learning: boolean;
  auto_length: boolean;
  lead_seconds: number | null;
  tail_seconds: number | null;
  game_config: GameProfileConfig;
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
  transcript: { provider: string; language: string; words: unknown[]; speech?: number[][]; speakers: number } | null;
  clips: Clip[];
  montages: Montage[];
  progress: JobProgress;
  content_type: string | null;
  facecam: Rect | null;
  events: DetectedEvent[];
  ocr_report: OcrReport;
  warnings: Notice[];
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
  warnings: Notice[];
  content_type: string | null;
  settings: Pick<ImportSettings, "power_mode" | "aspect">;
  system?: {
    cpu_pct: number | null;
    gpu_pct: number | null;
    gpu_mem_mb: number | null;
    gpu_mem_total_mb: number | null;
  };
  timing?: {
    elapsed_seconds: number | null;
    eta_seconds: number | null;
    source_duration: number | null;
  };
  target_clips?: number;
  rendered_count?: number;
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
  // Caption production-value flags — whether this preset enables keyword
  // emphasis (power-word colour/scale across the line) and tasteful auto-emoji.
  emphasis?: boolean;
  emoji?: boolean;
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
    vad: boolean;
    scene_detect: boolean;
    emotion: boolean;
    denoise: boolean;
    audio_events: boolean;
    panns_audio: boolean;
    clap_audio: boolean;
    reframe_engine: string;
    active_speaker: boolean;
    face_tracking: boolean;
    url_import: boolean;
    gpu: boolean;
    gpu_encode: boolean;
    device: string;
    llm: boolean;
    llm_model: string | null;
    vlm: boolean;
    vlm_model: string | null;
    whisper_model: string;
    diarization_model: string | null;
    auto_model: boolean;
    vram_gb: number;
    cpu: number;
    recommended_power_mode: PowerMode;
    /** New diagnostics-panel fields (from /api/capabilities). */
    deno: boolean;
    ollama: boolean;
    torchaudio: boolean;
    paddleocr: boolean;
    easyocr: boolean;
    tesseract: boolean;
  };
  output: { width: number; height: number };
}

/** Grouped detail view returned by /api/capabilities.detail. */
export interface CapabilityItem {
  key: string;
  available: boolean;
  label: string;
  impact: string;
}
export interface CapabilityCategory {
  name: string;
  items: CapabilityItem[];
}
export interface CapabilityDetail {
  categories: CapabilityCategory[];
}

export interface PublishContent {
  titles: string[];
  description: string;
  hashtags: string[];
  excerpt: string;
}
