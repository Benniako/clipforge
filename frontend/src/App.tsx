import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { api } from "./lib/api";
import type { Health } from "./lib/types";
import CueModal from "./components/CueModal";
import ClipEditor from "./screens/ClipEditor";
import ProjectView from "./screens/ProjectView";
import Upload from "./screens/Upload";

function Caps({ health }: { health: Health | null }) {
  if (!health) return null;
  const c = health.capabilities;
  const engine =
    c.transcription === "whisperx" ? "WhisperX" : c.transcription === "whisper" ? "Whisper" : "Synthetisch";
  const asr =
    c.transcription === "synthetic"
      ? "Synthetische Transkription"
      : `${engine} ${c.whisper_model}` + (c.diarization ? " + Sprecher" : "");
  const hw = `Geraet: ${c.device}${c.vram_gb ? ` - ${c.vram_gb} GB VRAM` : ""} - ${c.cpu} CPU` +
    (c.auto_model ? " - Modell automatisch gewaehlt" : "");
  const ocrName = c.ocr
    ? { paddleocr: "PaddleOCR", easyocr: "EasyOCR", tesseract: "Tesseract" }[c.ocr] ?? "OCR"
    : "OCR";
  const items: [string, boolean, string][] = [
    [asr, c.transcription !== "synthetic", hw],
    ["Gesichts-Tracking", c.face_tracking, ""],
    [
      c.ocr ? `${ocrName} Cues` : "Bildschirm-OCR",
      Boolean(c.ocr),
      c.ocr
        ? "Liest Spieltext auf dem Bildschirm und lernt wiederverwendbare Audio-Cues"
        : "Installiere easyocr oder paddleocr, um Spielereignisse auf dem Bildschirm zu erkennen",
    ],
    [c.gpu_encode ? "GPU-Encode" : c.gpu ? "GPU-Rendering" : "CPU-Rendering", c.gpu || c.gpu_encode, hw],
  ];
  if (c.vad) items.push(["VAD-Untertitel", true, "Untertitel werden exakt an Sprache ausgerichtet"]);
  if (c.emotion) items.push(["Emotions-Score", true, "Erkennt Aufregung als Virality-Signal"]);
  if (c.audio_events) {
    items.push([
      c.panns_audio ? "PANNs Audio" : c.clap_audio ? "CLAP Audio" : "Audio-Ereignisse",
      true,
      c.panns_audio
        ? "Erkennt Jubel, Lachen und Explosionen fuer die Virality-Wertung"
        : "Zero-Shot-Audio-Cues fuer Jubel, Lachen und Action",
    ]);
  }
  if (c.denoise) items.push(["Saubere Stimme", true, "Trennt Sprache von Musik und Spielsound"]);
  if (c.reframe_engine && c.reframe_engine !== "haar") {
    items.push([
      `${c.reframe_engine === "yolo" ? "YOLO" : "MediaPipe"} Reframe`,
      true,
      "Motiv-Tracking fuer 9:16",
    ]);
  }
  if (c.active_speaker) items.push(["Aktiver Sprecher", true, "LR-ASD folgt der tatsaechlich sprechenden Person"]);
  if (c.llm) items.push(["KI-Titel + Viral", true, c.llm_model ?? ""]);
  if (c.vlm) items.push(["KI-Bildanalyse", true, `Bildbewertung auf Keyframes${c.vlm_model ? ` (${c.vlm_model})` : ""}`]);
  return (
    <div className="caps" title="In dieser Umgebung erkannte ClipForge-Funktionen">
      {items.map(([label, on, title]) => (
        <span key={label} title={title || undefined}>
          <span className={"cap-dot" + (on ? "" : " off")} />
          {label}
        </span>
      ))}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [showCues, setShowCues] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <nav className="nav">
        <Link to="/" className="brand">
          <span className="mark">◆</span> ClipForge
        </Link>
        <div className="spacer" />
        <Caps health={health} />
        <button
          className="btn ghost sm"
          onClick={() => setShowCues(true)}
          title="Referenzsounds oder OCR-Begriffe fuer Spielereignisse hinzufuegen und testen"
        >
          Spiel-Cues
        </button>
        <Link to="/" className="btn primary sm">
          + Neues Projekt
        </Link>
      </nav>
      {showCues && <CueModal onClose={() => setShowCues(false)} />}
      <Routes>
        <Route path="/" element={<Upload health={health} />} />
        <Route path="/p/:projectId" element={<ProjectView />} />
        <Route path="/p/:projectId/clip/:clipId" element={<ClipEditor />} />
      </Routes>
    </div>
  );
}
