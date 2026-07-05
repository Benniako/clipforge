// Tiny typed API client. All paths are same-origin (dev proxy / prod static).
import type {
  AiBoostSettings,
  Clip,
  CapabilityDetail,
  GameProfileConfig,
  Health,
  ImportSettings,
  Project,
  ProjectSummary,
  PublishContent,
  StatusPayload,
  StyleTemplate,
} from "./types";

export interface CueEvent {
  name: string;
  desc: string;
  hint: string;
  configured: boolean;
}
export type CuesStatus = Record<
  string,
  { label: string; configured: number; total: number; events: CueEvent[]; visual?: Record<string, string[]> }
>;
export type VisualCuesStatus = Record<string, Record<string, string[]>>;
export interface VisualCueRegion {
  name: string;
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface VisualCueProfile {
  phrases: Record<string, string[]>;
  regions: Record<string, VisualCueRegion[]>;
  false: Record<string, string[]>;
}
export type VisualCueMeta = Record<string, VisualCueProfile>;

/** Default timeout (ms) for non-upload requests. */
const REQUEST_TIMEOUT_MS = 30_000;

function fetchWithTimeout(url: string | URL, options: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const pathPart = (value: string) => encodeURIComponent(value);

export interface CreateProjectInput {
  name?: string;
  file?: File;
  url?: string;
  platform: string;
  power_mode: string;
  min_len: number;
  max_len: number;
  target_clips: number;
  style_id: string;
  language: string;
  content_type: string;
  aspect: string;
  burn_captions: boolean;
  game_profile: string;
  tighten: boolean;
  denoise: boolean;
  motion: string;
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
  game_config?: GameProfileConfig;
  onProgress?: (pct: number) => void;
}

function projectSettings(input: CreateProjectInput): Record<string, unknown> {
  return {
    name: input.name ?? "",
    platform: input.platform,
    power_mode: input.power_mode,
    min_len: input.min_len,
    max_len: input.max_len,
    target_clips: input.target_clips,
    style_id: input.style_id,
    language: input.language,
    content_type: input.content_type,
    aspect: input.aspect,
    burn_captions: input.burn_captions,
    game_profile: input.game_profile,
    tighten: input.tighten,
    denoise: input.denoise,
    motion: input.motion,
    ai_boost_emphasis: input.ai_boost?.emphasis ?? true,
    ai_boost_emoji: input.ai_boost?.emoji ?? true,
    ai_boost_speaker_colors: input.ai_boost?.speakerColors ?? true,
    ai_boost_auto_zoom: input.ai_boost?.autoZoom ?? true,
    ai_boost_broll: input.ai_boost?.broll ?? false,
    ai_boost_hook_check: input.ai_boost?.hookCheck ?? true,
    facecam_layout: input.facecam_layout,
    use_ocr: input.use_ocr,
    use_vlm: input.use_vlm,
    use_cues: input.use_cues,
    use_audio_events: input.use_audio_events,
    cue_learning: input.cue_learning,
    auto_length: input.auto_length,
    lead_seconds: input.lead_seconds,
    tail_seconds: input.tail_seconds,
    game_config: input.game_config,
  };
}

function asciiJson(value: unknown): string {
  return JSON.stringify(value).replace(/[^\x20-\x7E]/g, (ch) =>
    `\\u${ch.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
}

function appendProjectFormFields(fd: FormData, input: CreateProjectInput) {
  const settings = projectSettings(input);
  Object.entries(settings).forEach(([key, value]) => {
    if (value === null || value === undefined || key === "game_config") return;
    fd.set(key, String(value));
  });
  if (input.lead_seconds !== null) fd.set("lead_seconds", String(input.lead_seconds));
  if (input.tail_seconds !== null) fd.set("tail_seconds", String(input.tail_seconds));
  if (input.game_config) {
    fd.set("detection_mode", input.game_config.detection_mode);
    fd.set("visual_rois_json", JSON.stringify(input.game_config.visual_rois ?? []));
    fd.set("visual_text_cues", (input.game_config.visual_text_cues ?? []).join("\n"));
    fd.set("reference_audio_files", (input.game_config.reference_audio_files ?? []).join("\n"));
    fd.set("vlm_visual_prompts", (input.game_config.vlm_visual_prompts ?? []).join("\n"));
    fd.set("audio_prompts", (input.game_config.audio_prompts ?? []).join("\n"));
    fd.set("audio_negative_prompts", (input.game_config.audio_negative_prompts ?? []).join("\n"));
  }
}

function parseXhrError(xhr: XMLHttpRequest): Error {
  let msg = xhr.statusText;
  try {
    msg = JSON.parse(xhr.responseText).detail ?? msg;
  } catch {
    /* ignore */
  }
  return new Error(msg);
}

export const api = {
  health: () => fetchWithTimeout("/api/health").then((r) => json<Health>(r)),

  /** AI-generated publish-ready content for a clip (titles, description, hashtags). */
  publishContent: (projectId: string, clipId: string, platform?: string) =>
    fetchWithTimeout(`/api/projects/${projectId}/clips/${clipId}/publish-content?platform=${encodeURIComponent(platform ?? "generic")}`)
      .then((r) => json<PublishContent>(r)),

  /** Grouped capability inventory for the diagnostics panel. */
  capabilities: () =>
    fetchWithTimeout("/api/capabilities")
      .then((r) => json<{ flat: Health["capabilities"]; detail: CapabilityDetail }>(r)),

  styles: () => fetchWithTimeout("/api/styles").then((r) => json<StyleTemplate[]>(r)),

  cues: () => fetchWithTimeout("/api/cues").then((r) => json<CuesStatus>(r)),

  addCue: (game: string, event: string, opts: { url?: string; file?: File }) => {
    const fd = new FormData();
    if (opts.url) fd.set("url", opts.url);
    if (opts.file) fd.set("file", opts.file);
    return fetchWithTimeout(`/api/cues/${pathPart(game)}/${pathPart(event)}`, { method: "POST", body: fd }).then(
      (r) => json<CuesStatus>(r),
    );
  },

  removeCue: (game: string, event: string) =>
    fetchWithTimeout(`/api/cues/${pathPart(game)}/${pathPart(event)}`, { method: "DELETE" }).then((r) => json<CuesStatus>(r)),

  visualCues: () => fetchWithTimeout("/api/cues/visual").then((r) => json<VisualCuesStatus>(r)),

  visualCueMeta: () => fetchWithTimeout("/api/cues/visual-meta").then((r) => json<VisualCueMeta>(r)),

  addVisualCue: (game: string, label: string, phrase: string) => {
    const fd = new FormData();
    fd.set("phrase", phrase);
    return fetchWithTimeout(`/api/cues/visual/${pathPart(game)}/${pathPart(label)}`, { method: "POST", body: fd }).then((r) =>
      json<VisualCuesStatus>(r),
    );
  },

  addVisualCueRegion: (
    game: string,
    label: string,
    box: { x: number; y: number; w: number; h: number },
    opts: { name?: string; phrase?: string } = {},
  ) => {
    const fd = new FormData();
    fd.set("x", String(box.x));
    fd.set("y", String(box.y));
    fd.set("w", String(box.w));
    fd.set("h", String(box.h));
    if (opts.name) fd.set("name", opts.name);
    if (opts.phrase) fd.set("phrase", opts.phrase);
    return fetchWithTimeout(`/api/cues/visual/${pathPart(game)}/${pathPart(label)}/region`, { method: "POST", body: fd }).then((r) =>
      json<VisualCueMeta>(r),
    );
  },

  markVisualCueFalse: (game: string, label: string, phrase: string) => {
    const fd = new FormData();
    fd.set("phrase", phrase);
    return fetchWithTimeout(`/api/cues/visual/${pathPart(game)}/${pathPart(label)}/false`, { method: "POST", body: fd }).then((r) =>
      json<VisualCuesStatus>(r),
    );
  },

  removeVisualCue: (game: string, label: string, phrase?: string) => {
    const qs = phrase ? `?phrase=${encodeURIComponent(phrase)}` : "";
    return fetchWithTimeout(`/api/cues/visual/${pathPart(game)}/${pathPart(label)}${qs}`, { method: "DELETE" }).then((r) =>
      json<VisualCuesStatus>(r),
    );
  },

  testOcrCue: (
    game: string,
    file: File,
    box: { x: number; y: number; w: number; h: number },
    opts: { label?: string; save?: boolean } = {},
  ) => {
    const fd = new FormData();
    fd.set("game", game);
    fd.set("file", file);
    fd.set("x", String(box.x));
    fd.set("y", String(box.y));
    fd.set("w", String(box.w));
    fd.set("h", String(box.h));
    if (opts.label) fd.set("label", opts.label);
    fd.set("save", String(!!opts.save));
    return fetchWithTimeout("/api/cues/lab/ocr", { method: "POST", body: fd }).then((r) =>
      json<{ text: string; matches: { label: string; phrase: string }[]; saved: boolean; visual: VisualCuesStatus }>(r),
    );
  },

  testAudioCues: (game: string, file: File) => {
    const fd = new FormData();
    fd.set("game", game);
    fd.set("file", file);
    return fetchWithTimeout("/api/cues/lab/audio", { method: "POST", body: fd }).then((r) =>
      json<{ count: number; events: { t: number; label: string; similarity: number; source: string }[] }>(r),
    );
  },

  testAudioWindow: (
    game: string,
    file: File,
    start: number,
    duration: number,
    opts: { label?: string; save?: boolean } = {},
  ) => {
    const fd = new FormData();
    fd.set("game", game);
    fd.set("file", file);
    fd.set("start", String(start));
    fd.set("duration", String(duration));
    fd.set("save", String(!!opts.save));
    if (opts.label) fd.set("label", opts.label);
    return fetchWithTimeout("/api/cues/lab/audio-window", { method: "POST", body: fd }).then((r) =>
      json<{ count: number; saved: boolean; start: number; duration: number; events: { t: number; label: string; similarity: number; source: string }[] }>(r),
    );
  },

  listProjects: () => fetchWithTimeout("/api/projects").then((r) => json<ProjectSummary[]>(r)),

  getProject: (id: string) =>
    fetchWithTimeout(`/api/projects/${id}`).then((r) => json<Project>(r)),

  status: (id: string) =>
    fetchWithTimeout(`/api/projects/${id}/status`).then((r) => json<StatusPayload>(r)),

  pauseProject: (id: string) =>
    fetchWithTimeout(`/api/projects/${id}/pause`, { method: "POST" }).then((r) => json<StatusPayload>(r)),

  resumeProject: (id: string) =>
    fetchWithTimeout(`/api/projects/${id}/resume`, { method: "POST" }).then((r) => json<StatusPayload>(r)),

  deleteProject: (id: string) =>
    fetchWithTimeout(`/api/projects/${id}`, { method: "DELETE" }).then((r) => json(r)),

  purgeProject: (id: string) =>
    fetch(`/api/projects/${id}/purge`, { method: "DELETE" }).then((r) => json(r)),

  reprocess: (id: string, overrides: Partial<ImportSettings> = {}) =>
    fetchWithTimeout(`/api/projects/${id}/reprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(overrides),
    }).then((r) => json<Project>(r)),

  // Re-render all clips in a new output format (no re-detection).
  setAspect: (id: string, aspect: string) =>
    fetchWithTimeout(`/api/projects/${id}/aspect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aspect }),
    }).then((r) => json<Project>(r)),

  downloadSrtUrl: (projectId: string, clipId: string) =>
    `/api/projects/${projectId}/clips/${clipId}/captions?format=srt`,

  downloadVttUrl: (projectId: string, clipId: string) =>
    `/api/projects/${projectId}/clips/${clipId}/captions?format=vtt`,

	  trimClip: (projectId: string, clipId: string, bounds: { start?: number; end?: number; title?: string }) =>
	    fetchWithTimeout(`/api/projects/${projectId}/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bounds),
    }).then((r) => { if (!r.ok) throw new Error(`trim failed (${r.status})`); }),

  // Uses XHR so we can report real upload progress for large files.
  createProject: (input: CreateProjectInput) =>
    new Promise<Project>((resolve, reject) => {
      const sendMultipart = () => {
        const fd = new FormData();
        appendProjectFormFields(fd, input);
        if (input.url) fd.set("url", input.url);
        if (input.file) fd.set("file", input.file);

        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/projects");
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && input.onProgress)
            input.onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else {
            reject(parseXhrError(xhr));
          }
        };
        xhr.onerror = () => reject(new Error("network error during upload"));
        xhr.send(fd);
      };

      if (input.file && !input.url) {
        const xhr = new XMLHttpRequest();
        const filename = encodeURIComponent(input.file.name || "upload.mp4");
        xhr.open("POST", `/api/projects/raw-upload?filename=${filename}`);
        xhr.setRequestHeader("Content-Type", input.file.type || "application/octet-stream");
        xhr.setRequestHeader("X-ClipForge-Settings", asciiJson(projectSettings(input)));
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && input.onProgress)
            input.onProgress(Math.round((e.loaded / e.total) * 100));
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(JSON.parse(xhr.responseText));
          } else if (xhr.status === 404 || xhr.status === 405) {
            sendMultipart();
          } else {
            reject(parseXhrError(xhr));
          }
        };
        xhr.onerror = () => reject(new Error("network error during upload"));
        xhr.send(input.file);
        return;
      }

      sendMultipart();
    }),

  editClip: (
    projectId: string,
    clipId: string,
    edit: Partial<{
      title: string;
      start: number;
      end: number;
      style_id: string;
      reframe_cx: number | null;
      caption_words: { t: number; d: number; text: string; speaker: number | null }[];
      caption_speakers: number[] | null;
      layout: string;
      facecam: { x: number; y: number; w: number; h: number };
      aspect: string;
    }>,
  ) =>
    fetchWithTimeout(`/api/projects/${projectId}/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(edit),
    }).then((r) => json<Clip>(r)),

  rerenderClip: (projectId: string, clipId: string) =>
    fetchWithTimeout(`/api/projects/${projectId}/clips/${clipId}/rerender`, {
      method: "POST",
    }).then((r) => json<Clip>(r)),

  rerenderClips: (projectId: string, clipIds: string[]) =>
    fetchWithTimeout(`/api/projects/${projectId}/clips/rerender`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: clipIds }),
    }).then((r) => json<Project>(r)),

  createMontage: (projectId: string, clipIds: string[], title?: string) =>
    fetchWithTimeout(`/api/projects/${projectId}/montage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: clipIds, title }),
    }).then((r) => json<import("./types").Montage>(r)),

  downloadMontageUrl: (projectId: string, montageId: string) =>
    `/api/projects/${projectId}/montages/${montageId}/download`,

  rateClip: (projectId: string, clipId: string, rating: "up" | "down" | "none") =>
    fetchWithTimeout(`/api/projects/${projectId}/clips/${clipId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    }).then((r) => json<Clip>(r)),

  learning: () =>
    fetchWithTimeout("/api/learning").then((r) =>
      json<{
        total_ratings: number;
        likes: number;
        dislikes: number;
        trims: number;
        personalized: boolean;
        learned_top_features: Record<string, Record<string, number>>;
      }>(r),
    ),

  resetLearning: (scope?: string) =>
    fetchWithTimeout("/api/learning/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scope ? { scope } : {}),
    }).then((r) => json(r)),

  downloadClipUrl: (projectId: string, clipId: string) =>
    `/api/projects/${projectId}/clips/${clipId}/download`,

  exportBatchUrl: (projectId: string) => `/api/projects/${projectId}/export`,

  exportPremiereUrl: (projectId: string) => `/api/projects/${projectId}/export/premiere`,
};
