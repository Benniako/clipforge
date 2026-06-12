// Tiny typed API client. All paths are same-origin (dev proxy / prod static).
import type {
  Clip,
  Health,
  Project,
  ProjectSummary,
  StatusPayload,
  StyleTemplate,
} from "./types";

export interface CueEvent {
  name: string;
  desc: string;
  hint: string;
  kind: "audio" | "visual";
  configured: boolean;
}
export type CuesStatus = Record<
  string,
  { label: string; configured: number; total: number; events: CueEvent[] }
>;

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
  motion: string;
  facecam_layout: string;
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

  listProjects: () => fetch("/api/projects").then((r) => json<ProjectSummary[]>(r)),

  getProject: (id: string) =>
    fetch(`/api/projects/${id}`).then((r) => json<Project>(r)),

  status: (id: string) =>
    fetch(`/api/projects/${id}/status`).then((r) => json<StatusPayload>(r)),

  deleteProject: (id: string) =>
    fetch(`/api/projects/${id}`, { method: "DELETE" }).then((r) => json(r)),

  reprocess: (id: string) =>
    fetch(`/api/projects/${id}/reprocess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
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
      fd.set("motion", input.motion);
      fd.set("facecam_layout", input.facecam_layout);
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
