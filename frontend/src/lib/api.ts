// Tiny typed API client. All paths are same-origin (dev proxy / prod static).
import type {
  Clip,
  Health,
  ImportSettings,
  Project,
  ProjectSummary,
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
  facecam_layout: string;
  use_ocr: boolean;
  use_vlm: boolean;
  use_cues: boolean;
  use_audio_events: boolean;
  cue_learning: boolean;
  auto_length: boolean;
  lead_seconds: number | null;
  tail_seconds: number | null;
  onProgress?: (pct: number) => void;
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),

  styles: () => fetch("/api/styles").then((r) => json<StyleTemplate[]>(r)),

  cues: () => fetch("/api/cues").then((r) => json<CuesStatus>(r)),

  addCue: (game: string, event: string, opts: { url?: string; file?: File }) => {
    const fd = new FormData();
    if (opts.url) fd.set("url", opts.url);
    if (opts.file) fd.set("file", opts.file);
    return fetch(`/api/cues/${game}/${event}`, { method: "POST", body: fd }).then(
      (r) => json<CuesStatus>(r),
    );
  },

  removeCue: (game: string, event: string) =>
    fetch(`/api/cues/${game}/${event}`, { method: "DELETE" }).then((r) => json<CuesStatus>(r)),

  visualCues: () => fetch("/api/cues/visual").then((r) => json<VisualCuesStatus>(r)),

  addVisualCue: (game: string, label: string, phrase: string) => {
    const fd = new FormData();
    fd.set("phrase", phrase);
    return fetch(`/api/cues/visual/${game}/${label}`, { method: "POST", body: fd }).then((r) =>
      json<VisualCuesStatus>(r),
    );
  },

  removeVisualCue: (game: string, label: string, phrase?: string) => {
    const qs = phrase ? `?phrase=${encodeURIComponent(phrase)}` : "";
    return fetch(`/api/cues/visual/${game}/${label}${qs}`, { method: "DELETE" }).then((r) =>
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
    return fetch("/api/cues/lab/ocr", { method: "POST", body: fd }).then((r) =>
      json<{ text: string; matches: { label: string; phrase: string }[]; saved: boolean; visual: VisualCuesStatus }>(r),
    );
  },

  testAudioCues: (game: string, file: File) => {
    const fd = new FormData();
    fd.set("game", game);
    fd.set("file", file);
    return fetch("/api/cues/lab/audio", { method: "POST", body: fd }).then((r) =>
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
    return fetch("/api/cues/lab/audio-window", { method: "POST", body: fd }).then((r) =>
      json<{ count: number; saved: boolean; start: number; duration: number; events: { t: number; label: string; similarity: number; source: string }[] }>(r),
    );
  },

  listProjects: () => fetch("/api/projects").then((r) => json<ProjectSummary[]>(r)),

  getProject: (id: string) =>
    fetch(`/api/projects/${id}`).then((r) => json<Project>(r)),

  status: (id: string) =>
    fetch(`/api/projects/${id}/status`).then((r) => json<StatusPayload>(r)),

  deleteProject: (id: string) =>
    fetch(`/api/projects/${id}`, { method: "DELETE" }).then((r) => json(r)),

  reprocess: (id: string, overrides: Partial<ImportSettings> = {}) =>
    fetch(`/api/projects/${id}/reprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(overrides),
    }).then((r) => json<Project>(r)),

  // Re-render all clips in a new output format (no re-detection).
  setAspect: (id: string, aspect: string) =>
    fetch(`/api/projects/${id}/aspect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ aspect }),
    }).then((r) => json<Project>(r)),

  downloadSrtUrl: (projectId: string, clipId: string) =>
    `/api/projects/${projectId}/clips/${clipId}/captions.srt`,

  // Uses XHR so we can report real upload progress for large files.
  createProject: (input: CreateProjectInput) =>
    new Promise<Project>((resolve, reject) => {
      const fd = new FormData();
      fd.set("name", input.name ?? "");
      fd.set("platform", input.platform);
      fd.set("power_mode", input.power_mode);
      fd.set("min_len", String(input.min_len));
      fd.set("max_len", String(input.max_len));
      fd.set("target_clips", String(input.target_clips));
      fd.set("style_id", input.style_id);
      fd.set("language", input.language);
      fd.set("content_type", input.content_type);
      fd.set("aspect", input.aspect);
      fd.set("burn_captions", String(input.burn_captions));
      fd.set("game_profile", input.game_profile);
      fd.set("tighten", String(input.tighten));
      fd.set("denoise", String(input.denoise));
      fd.set("motion", input.motion);
      fd.set("facecam_layout", input.facecam_layout);
      fd.set("use_ocr", String(input.use_ocr));
      fd.set("use_vlm", String(input.use_vlm));
      fd.set("use_cues", String(input.use_cues));
      fd.set("use_audio_events", String(input.use_audio_events));
      fd.set("cue_learning", String(input.cue_learning));
      fd.set("auto_length", String(input.auto_length));
      if (input.lead_seconds !== null) fd.set("lead_seconds", String(input.lead_seconds));
      if (input.tail_seconds !== null) fd.set("tail_seconds", String(input.tail_seconds));
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
          let msg = xhr.statusText;
          try {
            msg = JSON.parse(xhr.responseText).detail ?? msg;
          } catch {
            /* ignore */
          }
          reject(new Error(msg));
        }
      };
      xhr.onerror = () => reject(new Error("network error during upload"));
      xhr.send(fd);
    }),

  editClip: (
    projectId: string,
    clipId: string,
    edit: Partial<{
      title: string;
      start: number;
      end: number;
      style_id: string;
      reframe_cx: number;
      caption_words: { t: number; d: number; text: string }[];
      caption_speakers: number[] | null;
      layout: string;
      facecam: { x: number; y: number; w: number; h: number };
      aspect: string;
    }>,
  ) =>
    fetch(`/api/projects/${projectId}/clips/${clipId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(edit),
    }).then((r) => json<Clip>(r)),

  rerenderClip: (projectId: string, clipId: string) =>
    fetch(`/api/projects/${projectId}/clips/${clipId}/rerender`, {
      method: "POST",
    }).then((r) => json<Clip>(r)),

  rerenderClips: (projectId: string, clipIds: string[]) =>
    fetch(`/api/projects/${projectId}/clips/rerender`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: clipIds }),
    }).then((r) => json<Project>(r)),

  createMontage: (projectId: string, clipIds: string[], title?: string) =>
    fetch(`/api/projects/${projectId}/montage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clip_ids: clipIds, title }),
    }).then((r) => json<import("./types").Montage>(r)),

  downloadMontageUrl: (projectId: string, montageId: string) =>
    `/api/projects/${projectId}/montages/${montageId}/download`,

  rateClip: (projectId: string, clipId: string, rating: "up" | "down" | "none") =>
    fetch(`/api/projects/${projectId}/clips/${clipId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    }).then((r) => json<Clip>(r)),

  learning: () =>
    fetch("/api/learning").then((r) =>
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
    fetch("/api/learning/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scope ? { scope } : {}),
    }).then((r) => json(r)),

  downloadClipUrl: (projectId: string, clipId: string) =>
    `/api/projects/${projectId}/clips/${clipId}/download`,

  exportBatchUrl: (projectId: string) => `/api/projects/${projectId}/export`,
};
