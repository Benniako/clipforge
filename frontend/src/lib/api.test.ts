import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

// Mock global fetch so we can assert URL/method/body without hitting the network.
type FetchMock = ReturnType<typeof vi.fn>;
let fetchMock: FetchMock;

beforeEach(() => {
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof globalThis.fetch;
});

afterEach(() => {
  vi.restoreAllMocks();
});

function mockResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? "OK" : "Error",
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
    headers: new Headers(),
  } as unknown as Response;
}

const projectInput = {
  platform: "tiktok", power_mode: "balanced", min_len: 15, max_len: 60, target_clips: 10,
  style_id: "bold-pop", language: "de", content_type: "auto", aspect: "9:16",
  burn_captions: true, game_profile: "auto", tighten: false, denoise: false, motion: "none",
  ai_boost: { emphasis: true, emoji: true, speakerColors: true, autoZoom: true, broll: false, hookCheck: true },
  facecam_layout: "auto", use_ocr: false, use_vlm: false, use_cues: true, use_audio_events: false,
  cue_learning: true, auto_length: false, lead_seconds: null, tail_seconds: null,
};

describe("importing recordings by path", () => {
  it("lists recordings from /api/recordings", async () => {
    const rows = [{ path: "C:\\Videos\\a.mp4", name: "a.mp4", folder: "C:\\Videos",
                    size_bytes: 5, modified: 1, in_progress: false }];
    fetchMock.mockResolvedValue(mockResponse(rows));
    expect(await api.recordings()).toEqual(rows);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/recordings");
  });

  it("sends the path in a form and uploads no file", async () => {
    const sent: { method?: string; url?: string; body?: FormData } = {};
    class FakeXhr {
      status = 201;
      statusText = "Created";
      responseText = JSON.stringify({ id: "p9" });
      upload: { onprogress: unknown } = { onprogress: null };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      open(method: string, url: string) { sent.method = method; sent.url = url; }
      setRequestHeader() {}
      send(body: FormData) { sent.body = body; this.onload?.(); }
    }
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    try {
      const project = await api.createProject({ ...projectInput, local_path: "C:\\Videos\\match.mp4" });
      expect(project).toEqual({ id: "p9" });
      expect(sent.method).toBe("POST");
      expect(sent.url).toBe("/api/projects");
      expect(sent.body?.get("local_path")).toBe("C:\\Videos\\match.mp4");
      expect(sent.body?.get("file")).toBeNull();
      expect(sent.body?.get("platform")).toBe("tiktok");
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("api client", () => {
  it("parses the error detail from a non-ok JSON response", async () => {
    fetchMock.mockResolvedValue(mockResponse({ detail: "clip not found" }, false, 404));
    await expect(api.getProject("p1")).rejects.toThrow("clip not found");
  });

  it("falls back to statusText when the error body is not JSON", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: () => Promise.reject(new SyntaxError("not JSON")),
      text: () => Promise.resolve("plain text"),
      headers: new Headers(),
    } as unknown as Response);
    await expect(api.getProject("p1")).rejects.toThrow("Internal Server Error");
  });

  it("constructs the correct URL, method, and body for createStyle", async () => {
    const style = {
      id: "custom_my", name: "My Template", font: "DejaVu Sans",
      font_size: 92, primary: "FFFFFF", highlight: "F5C518",
      outline: "000000", outline_w: 6, y_frac: 0.78, uppercase: true,
    };
    fetchMock.mockResolvedValue(mockResponse(style, true, 201));
    await api.createStyle(style);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/styles");
    expect(opts.method).toBe("POST");
    expect(opts.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(opts.body)).toEqual(style);
  });

  it("purgeProject uses fetchWithTimeout (not raw fetch) and DELETE method", async () => {
    fetchMock.mockResolvedValue(mockResponse({ ok: true }));
    await api.purgeProject("p1");
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/projects/p1/purge");
    expect(opts.method).toBe("DELETE");
  });

  it("trimClip parses the error body for detail on failure", async () => {
    fetchMock.mockResolvedValue(mockResponse({ detail: "start must be before end" }, false, 400));
    await expect(api.trimClip("p1", "c1", { start: 10, end: 5 }))
      .rejects.toThrow("start must be before end");
  });

  it("rateClip sends the rating as JSON to the feedback endpoint", async () => {
    fetchMock.mockResolvedValue(mockResponse({ id: "c1", rating: "up" }));
    await api.rateClip("p1", "c1", "up");
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/projects/p1/clips/c1/feedback");
    expect(opts.method).toBe("POST");
    expect(JSON.parse(opts.body)).toEqual({ rating: "up" });
  });

  it("encodes path segments in cue endpoints", async () => {
    fetchMock.mockResolvedValue(mockResponse({}));
    // Space and # are encoded by encodeURIComponent; ! is not (JS quirk).
    await api.removeCue("Valorant", "Ace #1");
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/cues/Valorant/Ace%20%231");
  });

  it("downloadSrtUrl and downloadVttUrl return same-origin paths", () => {
    expect(api.downloadSrtUrl("p1", "c1"))
      .toBe("/api/projects/p1/clips/c1/captions?format=srt");
    expect(api.downloadVttUrl("p1", "c1"))
      .toBe("/api/projects/p1/clips/c1/captions?format=vtt");
  });
});
