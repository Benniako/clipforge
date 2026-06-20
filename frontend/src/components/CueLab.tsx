import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import type { VisualCuesStatus } from "../lib/api";

type Box = { x: number; y: number; w: number; h: number };
type OcrResult = {
  text: string;
  matches: { label: string; phrase: string }[];
  saved: boolean;
  visual: VisualCuesStatus;
};
type AudioResult = {
  count: number;
  events: { t: number; label: string; similarity: number; source: string }[];
};

const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
const cleanLabel = (s: string) =>
  s
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9äöüß_-]+/g, "_")
    .replace(/^_+|_+$/g, "");

export default function CueLab({
  game,
  visual,
  sourceFile,
  onVisualChange,
  onAudioChange,
}: {
  game: string;
  visual?: Record<string, string[]>;
  sourceFile?: File | null;
  onVisualChange: (visual: VisualCuesStatus) => void;
  onAudioChange: () => void;
}) {
  const previewRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string>("");
  const [box, setBox] = useState<Box>({ x: 0.56, y: 0.02, w: 0.38, h: 0.28 });
  const [visualLabel, setVisualLabel] = useState("killfeed");
  const [manualPhrase, setManualPhrase] = useState("");
  const [ocrResult, setOcrResult] = useState<OcrResult | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioLabel, setAudioLabel] = useState("kill");
  const [audioWindowSeconds, setAudioWindowSeconds] = useState(2.5);
  const [audioResult, setAudioResult] = useState<AudioResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!imageFile) {
      setImageUrl("");
      return;
    }
    const url = URL.createObjectURL(imageFile);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  useEffect(() => {
    const looksLikeVideo = !!sourceFile && (
      sourceFile.type.startsWith("video/") || /\.(mp4|mov|mkv|webm|avi)$/i.test(sourceFile.name)
    );
    if (!looksLikeVideo || !sourceFile) {
      setSourceVideoUrl("");
      return;
    }
    const url = URL.createObjectURL(sourceFile);
    setSourceVideoUrl(url);
    setAudioFile((current) => current ?? sourceFile);
    return () => URL.revokeObjectURL(url);
  }, [sourceFile]);

  const currentVisual = useMemo(() => visual ?? {}, [visual]);
  const phraseToSave = manualPhrase.trim() || ocrResult?.text.trim() || "";

  const pointFromEvent = (ev: React.PointerEvent<HTMLDivElement>) => {
    const rect = previewRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: clamp01((ev.clientX - rect.left) / rect.width),
      y: clamp01((ev.clientY - rect.top) / rect.height),
    };
  };

  const startBox = (ev: React.PointerEvent<HTMLDivElement>) => {
    if (!imageFile) return;
    const p = pointFromEvent(ev);
    if (!p) return;
    dragStart.current = p;
    setBox({ x: p.x, y: p.y, w: 0.01, h: 0.01 });
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };

  const moveBox = (ev: React.PointerEvent<HTMLDivElement>) => {
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

  const captureVideoFrame = async () => {
    const video = videoRef.current;
    if (!video || !sourceFile) {
      setErr("Wähle zuerst ein Video aus.");
      return;
    }
    if (!video.videoWidth || !video.videoHeight) {
      setErr("Der Videoframe ist noch nicht bereit. Spiele kurz ab und stoppe dann.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setErr("Dieser Frame konnte nicht erfasst werden.");
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) {
      setErr("Dieser Frame konnte nicht erfasst werden.");
      return;
    }
    const ms = Math.round((video.currentTime || 0) * 1000);
    setImageFile(new File([blob], `clipforge-frame-${ms}.png`, { type: "image/png" }));
    setOcrResult(null);
    setErr(null);
  };

  const testOcr = async (save: boolean) => {
    if (!imageFile) {
      setErr("Erfasse erst einen Frame aus dem Video oder lade ein Bild hoch.");
      return;
    }
    const label = cleanLabel(visualLabel);
    if (save && !label) {
      setErr("Benenne den visuellen Cue vor dem Speichern.");
      return;
    }
    setBusy(save ? "save-ocr" : "test-ocr");
    setErr(null);
    try {
      if (save && manualPhrase.trim()) {
        await api.addVisualCueRegion(game, label, box, { name: label, phrase: manualPhrase.trim() });
        const next = await api.visualCues();
        onVisualChange(next);
        setOcrResult((r) => r && { ...r, saved: true, visual: next });
      } else {
        const result = await api.testOcrCue(game, imageFile, box, { label, save });
        setOcrResult(result);
        if (save) onVisualChange(result.visual);
      }
    } catch (e: any) {
      setErr(e?.message ?? "Cue-Lab fehlgeschlagen.");
    } finally {
      setBusy(null);
    }
  };

  const testAudio = async () => {
    if (!audioFile) {
      setErr("Füge zuerst ein Audio- oder Video-Beispiel hinzu.");
      return;
    }
    setBusy("test-audio");
    setErr(null);
    try {
      setAudioResult(await api.testAudioCues(game, audioFile));
    } catch (e: any) {
      setErr(e?.message ?? "Audio-Test fehlgeschlagen.");
    } finally {
      setBusy(null);
    }
  };

  const useAudioWindow = async (save: boolean) => {
    if (!sourceFile || !videoRef.current) {
      setErr("Wähle ein Video und gehe zuerst zur Cue-Stelle.");
      return;
    }
    const label = cleanLabel(audioLabel);
    if (save && !label) {
      setErr("Benenne den Audio-Cue vor dem Speichern.");
      return;
    }
    const start = Math.max(0, (videoRef.current.currentTime || 0) - audioWindowSeconds / 2);
    setBusy(save ? "save-audio-window" : "test-audio-window");
    setErr(null);
    try {
      const result = await api.testAudioWindow(game, sourceFile, start, audioWindowSeconds, {
        label,
        save,
      });
      setAudioResult(result);
      if (save) onAudioChange();
    } catch (e: any) {
      setErr(e?.message ?? "Audio-Fenster fehlgeschlagen.");
    } finally {
      setBusy(null);
    }
  };

  const saveAudio = async () => {
    if (!audioFile) {
      setErr("Füge zuerst einen sauberen Referenzsound hinzu.");
      return;
    }
    const label = cleanLabel(audioLabel);
    if (!label) {
      setErr("Benenne den Audio-Cue vor dem Speichern.");
      return;
    }
    setBusy("save-audio");
    setErr(null);
    try {
      await api.addCue(game, label, { file: audioFile });
      onAudioChange();
    } catch (e: any) {
      setErr(e?.message ?? "Audio-Cue konnte nicht gespeichert werden.");
    } finally {
      setBusy(null);
    }
  };

  const removeVisual = async (label: string, phrase: string) => {
    setBusy(`visual-${label}`);
    setErr(null);
    try {
      onVisualChange(await api.removeVisualCue(game, label, phrase));
    } catch (e: any) {
      setErr(e?.message ?? "Visueller Cue konnte nicht entfernt werden.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel section cue-lab">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h3>Cue-Testlabor</h3>
          <p className="muted tiny" style={{ margin: "6px 0 0" }}>
            Gehe durch das importierte Video, erfasse einen Frame, teste OCR/Audio und speichere dann nützliche Cues.
          </p>
        </div>
        {busy && <span className="pill">Arbeitet</span>}
      </div>
      {err && (
        <p className="tiny" style={{ color: "var(--bad)", marginBottom: 0 }}>
          {err}
        </p>
      )}

      <div className="cue-lab-grid">
        <div className="cue-lab-card">
          <h4>Visuell / OCR</h4>
          {sourceVideoUrl && (
            <div className="cue-video-source">
              <video ref={videoRef} src={sourceVideoUrl} controls preload="metadata" />
              <button className="btn primary sm" onClick={captureVideoFrame}>
                Aktuellen Frame erfassen
              </button>
            </div>
          )}
          <div className="row" style={{ flexWrap: "wrap" }}>
            <label className="btn sm ghost cue-file-btn">
              Bild-Alternative
              <input
                type="file"
                accept="image/*"
                hidden
                onChange={(ev) => {
                  setImageFile(ev.target.files?.[0] ?? null);
                  setOcrResult(null);
                }}
              />
            </label>
            {imageFile && <span className="muted tiny">Frame: {imageFile.name}</span>}
          </div>
          <div
            ref={previewRef}
            className={"cue-image-preview" + (!imageUrl ? " empty-preview" : "")}
            onPointerDown={startBox}
            onPointerMove={moveBox}
            onPointerUp={stopBox}
            onPointerCancel={stopBox}
          >
            {imageUrl ? <img src={imageUrl} alt="" draggable={false} /> : <span>Erfasse einen Videoframe und ziehe dann die OCR-Box auf.</span>}
            {imageUrl && (
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
          <div className="cue-form-row">
            <input
              className="input"
              value={visualLabel}
              onChange={(ev) => setVisualLabel(ev.target.value)}
              placeholder="Cue-Name, z. B. killfeed"
            />
            <button className="btn sm" disabled={busy === "test-ocr"} onClick={() => testOcr(false)}>
              OCR testen
            </button>
          </div>
          <textarea
            className="input cue-textarea"
            value={manualPhrase}
            onChange={(ev) => setManualPhrase(ev.target.value)}
            placeholder="Optional: korrigierten OCR-Text vor dem Speichern hier eintragen"
          />
          <button className="btn primary sm" disabled={!phraseToSave || busy === "save-ocr"} onClick={() => testOcr(true)}>
            Visuellen Cue speichern
          </button>
          {ocrResult && (
            <div className="cue-result">
              <b>OCR gelesen:</b>
              <span>{ocrResult.text || "In dieser Box wurde kein Text gefunden."}</span>
              {ocrResult.matches.length > 0 && (
                <small>
                  Treffer: {ocrResult.matches.map((m) => `${m.label} (${m.phrase})`).join(", ")}
                </small>
              )}
            </div>
          )}
        </div>

        <div className="cue-lab-card">
          <h4>Audio</h4>
          {sourceFile && (
            <button
              className="btn sm ghost cue-file-btn"
              onClick={() => {
                setAudioFile(sourceFile);
                setAudioResult(null);
              }}
            >
              Importiertes Video nutzen
            </button>
          )}
          <label className="btn sm ghost cue-file-btn">
            Anderes Audio-/Video-Beispiel
            <input
              type="file"
              accept="audio/*,video/*"
              hidden
              onChange={(ev) => {
                setAudioFile(ev.target.files?.[0] ?? null);
                setAudioResult(null);
              }}
            />
          </label>
          <p className="muted tiny cue-file-name">{audioFile?.name || "Noch kein Beispiel ausgewählt"}</p>
          <input
            className="input"
            value={audioLabel}
            onChange={(ev) => setAudioLabel(ev.target.value)}
            placeholder="Cue-Name, z. B. spike_planted"
          />
          {sourceVideoUrl && (
            <div className="cue-window-tools">
              <label className="muted tiny">Audiofenster an der aktuellen Stelle</label>
              <div className="range-label">
                <span>Länge des Ausschnitts</span>
                <b>{audioWindowSeconds.toFixed(1)}s</b>
              </div>
              <input
                type="range"
                min={0.5}
                max={8}
                step={0.5}
                value={audioWindowSeconds}
                onChange={(ev) => setAudioWindowSeconds(Number(ev.target.value))}
              />
              <div className="row" style={{ flexWrap: "wrap" }}>
                <button className="btn sm" disabled={busy === "test-audio-window"} onClick={() => useAudioWindow(false)}>
                  Aktuelles Fenster testen
                </button>
                <button className="btn primary sm" disabled={busy === "save-audio-window"} onClick={() => useAudioWindow(true)}>
                  Aktuelles Fenster speichern
                </button>
              </div>
            </div>
          )}
          <div className="row" style={{ flexWrap: "wrap" }}>
            <button className="btn sm" disabled={busy === "test-audio"} onClick={testAudio}>
              Installierte Cues testen
            </button>
            <button
              className="btn primary sm"
              disabled={!audioFile || (!!sourceVideoUrl && audioFile === sourceFile) || busy === "save-audio"}
              onClick={saveAudio}
              title={!!sourceVideoUrl && audioFile === sourceFile ? "Nutze Aktuelles Fenster speichern statt die ganze Gameplay-Datei zu speichern" : undefined}
            >
              Als Audio-Cue speichern
            </button>
          </div>
          {audioResult && (
            <div className="cue-result">
              <b>{audioResult.count} Cue-Treffer</b>
              {audioResult.events.slice(0, 8).map((e, idx) => (
                <span key={`${e.source}-${e.label}-${e.t}-${idx}`}>
                  {e.t.toFixed(1)}s - {e.label} - {(e.similarity * 100).toFixed(0)}%
                </span>
              ))}
              {audioResult.events.length > 8 && <small>Es werden die ersten 8 Treffer gezeigt.</small>}
            </div>
          )}
        </div>
      </div>

      {Object.keys(currentVisual).length > 0 && (
        <div className="cue-saved-list">
          <h4>Gespeicherte visuelle Cues</h4>
          {Object.entries(currentVisual).map(([label, phrases]) => (
            <div key={label} className="cue-saved-group">
              <b>{label}</b>
              {phrases.map((phrase) => (
                <button
                  key={`${label}-${phrase}`}
                  className="cue-chip"
                  disabled={busy === `visual-${label}`}
                  onClick={() => removeVisual(label, phrase)}
                  title="Diesen gespeicherten OCR-Begriff entfernen"
                >
                  {phrase} <span>x</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
