import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { api } from "./lib/api";
import type { Health } from "./lib/types";
import Upload from "./screens/Upload";
import ProjectView from "./screens/ProjectView";
import ClipEditor from "./screens/ClipEditor";

function Caps({ health }: { health: Health | null }) {
  if (!health) return null;
  const c = health.capabilities;
  const engine =
    c.transcription === "whisperx" ? "WhisperX" : c.transcription === "whisper" ? "Whisper" : "Synthetic";
  const asr =
    c.transcription === "synthetic"
      ? "Synthetic ASR"
      : `${engine} ${c.whisper_model}` + (c.diarization ? " +diariz." : "");
  const hw = `device: ${c.device}${c.vram_gb ? ` · ${c.vram_gb} GB VRAM` : ""} · ${c.cpu} cpu` +
    (c.auto_model ? " · model auto-selected" : "");
  const items: [string, boolean, string][] = [
    [asr, c.transcription !== "synthetic", hw],
    ["Face tracking", c.face_tracking, ""],
    [c.gpu_encode ? "GPU encode" : c.gpu ? "GPU" : "CPU render", c.gpu || c.gpu_encode, hw],
  ];
  if (c.llm) items.push(["AI titles", true, ""]);
  return (
    <div className="caps" title="Pipeline capabilities detected in this environment">
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
        <Link to="/" className="btn primary sm">
          + New project
        </Link>
      </nav>
      <Routes>
        <Route path="/" element={<Upload health={health} />} />
        <Route path="/p/:projectId" element={<ProjectView />} />
        <Route path="/p/:projectId/clip/:clipId" element={<ClipEditor />} />
      </Routes>
    </div>
  );
}
