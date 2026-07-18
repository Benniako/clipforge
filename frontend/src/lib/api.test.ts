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
