import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { api, type VisualCueMeta } from "../lib/api";
import type { DetectedEvent, Project } from "../lib/types";
import { fmtClock } from "../lib/format";

type Box = { x: number; y: number; w: number; h: number };
type OcrResult = {
  text: string;
  matches: { label: string; phrase: string }[];
  saved: boolean;
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const cleanLabel = (s: string) =>
  s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9äöüß_-]+/g, "_")
    .replace(/^_+|_+$/g, "");

const detailPhrase = (event?: DetectedEvent | null) => {
  const detail = (event?.detail || event?.label || "").trim();
  if (!detail) return "";
  const parts = detail.split("|").map((p) => p.trim()).filter(Boolean);
  return parts[parts.length - 1] || detail;
};

const defaultBox = (label: string): Box => {
  const key = label.toLowerCase();
  if (key.includes("kill")) return { x: 0.55, y: 0.02, w: 0.42, h: 0.28 };
  if (key.includes("spike") || key.includes("round") || key.includes("victory") || key.includes("win")) {
    return { x: 0.18, y: 0.0, w: 0.64, h: 0.32 };
  }
  return { x: 0.16, y: 0.22, w: 0.68, h: 0.5 };
};

const waitForSeek = (video: HTMLVideoElement, t: number) =>
  new Promise<void>((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.removeEventListener("seeked", finish);
      resolve();
    };
    video.addEventListener("seeked", finish);
    video.currentTime = Math.max(0, t);
    window.setTimeout(finish, 1200);
  });

export default function VisualCueCalibration({ project }: { project: Project }) {
  const game = project.settings.game_profile || "generic";
  const videoSrc = project.source ? `/media/${project.source.path}` : "";
  const ocrEvents = useMemo(
    () => (project.events || []).filter((e) => e.source === "ocr").sort((a, b) => a.t - b.t),
    [project.events],
  );
  const [selectedIdx, setSelectedIdx] = useState(0);
  const selectedEvent = ocrEvents[selectedIdx] ?? null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const [frameFile, setFrameFile] = useState<File | null>(null);
  const [frameUrl, setFrameUrl] = useState("");
  const [box, setBox] = useState<Box>(defaultBox(selectedEvent?.label || "killfeed"));
  const [label, setLabel] = useState(cleanLabel(selectedEvent?.label || "killfeed"));
  const [phrase, setPhrase] = useState(detailPhrase(selectedEvent));
  const [ocrResult, setOcrResult] = useState<OcrResult | null>(null);
  const [meta, setMeta] = useState<VisualCueMeta | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);

  useEffect(() => {
    api.visualCueMeta().then(setMeta).catch(() => {});
  }, []);

  useEffect(() => {
    if (!frameFile) {
      setFrameUrl("");
      return;
    }
    const url = URL.createObjectURL(frameFile);
    setFrameUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [frameFile]);

  useEffect(() => {
    if (!selectedEvent) return;
    setLabel(cleanLabel(selectedEvent.label));
    setPhrase(detailPhrase(selectedEvent));
    setBox(defaultBox(selectedEvent.label));
    setOcrResult(null);
    setOk(null);
    setErr(null);
    const video = videoRef.current;
    if (video) {
      video.currentTime = Math.max(0, selectedEvent.t);
    }
  }, [project.id, selectedEvent?.t, selectedEvent?.label]);

  const pointFromEvent = (ev: PointerEvent<HTMLDivElement>) => {
    const rect = previewRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: clamp01((ev.clientX - rect.left) / rect.width),
      y: clamp01((ev.clientY - rect.top) / rect.height),
    };
  };

  const startBox = (ev: PointerEvent<HTMLDivElement>) => {
    if (!frameUrl) return;
    const p = pointFromEvent(ev);
    if (!p) return;
    dragStart.current = p;
    setBox({ x: p.x, y: p.y, w: 0.01, h: 0.01 });
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };

  const moveBox = (ev: PointerEvent<HTMLDivElement>) => {
    if (!dragStart.current) return;
    const p = pointFromEvent(ev);
    if (!p) return;
    const sx = dragStart.current.x;
    const sy = dragStart.current.y;
    const x = Math.min(sx, p.x);
    const y = Math.min(sy, p.y);
    setBox({
      x,
      y,
      w: Math.max(0.01, Math.abs(p.x - sx)),
      h: Math.max(0.01, Math.abs(p.y - sy)),
    });
  };

  const stopBox = () => {
    dragStart.current = null;
  };

  const captureFrame = async (time?: number) => {
    const video = videoRef.current;
    if (!video || !videoSrc) {
      setErr("Kein Quellvideo für die Kalibrierung gefunden.");
      return;
    }
    setBusy("frame");
    setErr(null);
    setOk(null);
    try {
      if (typeof time === "number") {
        await waitForSeek(video, time);
      }
      if (!video.videoWidth || !video.videoHeight) {
        throw new Error("Videoframe ist noch nicht bereit. Kurz abspielen oder erneut versuchen.");
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("Frame konnte nicht gelesen werden.");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) throw new Error("Frame konnte nicht gespeichert werden.");
      const ms = Math.round((video.currentTime || 0) * 1000);
      setFrameFile(new File([blob], `visual-cue-${ms}.png`, { type: "image/png" }));
      setOcrResult(null);
      setOk(`Frame bei ${fmtClock(video.currentTime || 0)} geladen.`);
    } catch (e: any) {
      setErr(e?.message ?? "Frame konnte nicht geladen werden.");
    } finally {
      setBusy(null);
    }
  };

  const testBox = async () => {
    if (!frameFile) {
      setErr("Lade zuerst einen Frame und ziehe die Box auf den genauen HUD-Bereich.");
      return;
    }
    setBusy("ocr");
    setErr(null);
    setOk(null);
    try {
      const result = await api.testOcrCue(game, frameFile, box, { label: cleanLabel(label), save: false });
      setOcrResult(result);
      if (result.text.trim()) setPhrase((p) => p || result.text.trim());
    } catch (e: any) {
      setErr(e?.message ?? "OCR-Test fehlgeschlagen.");
    } finally {
      setBusy(null);
    }
  };

  const saveRegion = async () => {
    if (!frameFile) {
      setErr("Lade zuerst einen Frame, damit die Box wirklich kalibriert ist.");
      return;
    }
    const cueLabel = cleanLabel(label);
    if (!cueLabel) {
      setErr("Bitte gib dem visuellen Cue einen Namen.");
      return;
    }
    setBusy("save");
    setErr(null);
    setOk(null);
    try {
      const next = await api.addVisualCueRegion(game, cueLabel, box, {
        name: cueLabel,
        phrase: phrase.trim() || undefined,
      });
      setMeta(next);
      setOk("Visueller Cue und Pixelbereich gespeichert. Beim nächsten Scan wird diese Box mitgelesen.");
    } catch (e: any) {
      setErr(e?.message ?? "Visueller Cue konnte nicht gespeichert werden.");
    } finally {
      setBusy(null);
    }
  };

  const markFalse = async () => {
    const cueLabel = cleanLabel(label || selectedEvent?.label || "ocr");
    const falsePhrase = phrase.trim() || detailPhrase(selectedEvent) || selectedEvent?.label || "";
    if (!cueLabel || !falsePhrase.trim()) {
      setErr("Zum Markieren brauche ich Cue-Name und den falschen OCR-Text.");
      return;
    }
    setBusy("false");
    setErr(null);
    setOk(null);
    try {
      const next = await api.markVisualCueFalse(game, cueLabel, falsePhrase);
      setMeta(next);
      setOk("False Recognition gespeichert. Dieser Text wird künftig für diesen Cue ignoriert.");
    } catch (e: any) {
      setErr(e?.message ?? "False Recognition konnte nicht gespeichert werden.");
    } finally {
      setBusy(null);
    }
  };

  const savedRegions = meta?.[game]?.regions?.[cleanLabel(label)]?.length ?? 0;
  const savedFalse = meta?.[game]?.false?.[cleanLabel(label)]?.length ?? 0;

  return (
    <div className="visual-calibration">
      <div className="visual-calibration-grid">
        <div className="calibration-events">
          <h4>OCR-Treffer aus diesem Scan</h4>
          {ocrEvents.length === 0 ? (
            <p className="muted tiny">
              Noch keine OCR-Treffer im Projekt. Du kannst trotzdem im Video zu einer Stelle springen und manuell einen Frame laden.
            </p>
          ) : (
            <div className="calibration-event-list">
              {ocrEvents.slice(0, 40).map((event, idx) => (
                <button
                  key={`${event.t}-${event.label}-${idx}`}
                  className={"calibration-event" + (selectedIdx === idx ? " on" : "")}
                  onClick={() => setSelectedIdx(idx)}
                  title={`${event.detail || event.label} - ${Math.round(event.confidence * 100)}%`}
                >
                  <b>{event.label}</b>
                  <span>{fmtClock(event.t)}</span>
                  <small>{detailPhrase(event) || "OCR"}</small>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="calibration-workspace">
          <div className="calibration-video">
            {videoSrc ? (
              <video ref={videoRef} src={videoSrc} controls preload="metadata" playsInline />
            ) : (
              <div className="empty">Kein Quellvideo gefunden.</div>
            )}
            <div className="row" style={{ flexWrap: "wrap" }}>
              <button
                className="btn primary sm"
                disabled={!selectedEvent || busy === "frame"}
                onClick={() => selectedEvent && captureFrame(selectedEvent.t)}
              >
                Frame vom OCR-Treffer laden
              </button>
              <button className="btn ghost sm" disabled={!videoSrc || busy === "frame"} onClick={() => captureFrame()}>
                Aktuellen Videoframe laden
              </button>
            </div>
          </div>

          <div
            ref={previewRef}
            className={"cue-image-preview calibration-frame" + (!frameUrl ? " empty-preview" : "")}
            onPointerDown={startBox}
            onPointerMove={moveBox}
            onPointerUp={stopBox}
            onPointerCancel={stopBox}
          >
            {frameUrl ? (
              <img src={frameUrl} alt="" draggable={false} />
            ) : (
              <span>Frame laden, dann die Box exakt um Killfeed, Banner oder HUD-Text ziehen.</span>
            )}
            {frameUrl && (
              <i
                className="cue-ocr-box"
                style={{
                  left: `${box.x * 100}%`,
                  top: `${box.y * 100}%`,
                  width: `${box.w * 100}%`,
                  height: `${box.h * 100}%`,
                }}
              />
            )}
          </div>

          <div className="calibration-form">
            <input
              className="input"
              value={label}
              onChange={(ev) => setLabel(ev.target.value)}
              placeholder="Cue-Name, z. B. killfeed"
            />
            <textarea
              className="input cue-textarea"
              value={phrase}
              onChange={(ev) => setPhrase(ev.target.value)}
              placeholder="OCR-Text oder Korrektur, z. B. Spike entschärft"
            />
            <div className="range-label">
              <span>
                Box: x {(box.x * 100).toFixed(1)}%, y {(box.y * 100).toFixed(1)}%, w{" "}
                {(box.w * 100).toFixed(1)}%, h {(box.h * 100).toFixed(1)}%
              </span>
              <b>{savedRegions} Regionen / {savedFalse} False</b>
            </div>
            <div className="row" style={{ flexWrap: "wrap" }}>
              <button className="btn sm" disabled={!frameFile || busy === "ocr"} onClick={testBox}>
                OCR in Box testen
              </button>
              <button className="btn primary sm" disabled={!frameFile || busy === "save"} onClick={saveRegion}>
                Cue + Region speichern
              </button>
              <button className="btn danger sm" disabled={busy === "false"} onClick={markFalse}>
                Als False Recognition markieren
              </button>
            </div>
          </div>

          {ocrResult && (
            <div className="cue-result">
              <b>OCR gelesen</b>
              <span>{ocrResult.text || "Kein Text in dieser Box erkannt."}</span>
              {ocrResult.matches.length > 0 && (
                <small>
                  Treffer: {ocrResult.matches.map((m) => `${m.label} (${m.phrase})`).join(", ")}
                </small>
              )}
            </div>
          )}
          {err && <p className="tiny calibration-message bad">{err}</p>}
          {ok && <p className="tiny calibration-message good">{ok}</p>}
          {busy && <span className="pill">Arbeitet</span>}
        </div>
      </div>
    </div>
  );
}
