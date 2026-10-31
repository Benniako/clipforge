import { useEffect, useState } from "react";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./lib/api";
import type { Health } from "./lib/types";
import { capabilityReadiness, capabilitySummary } from "./lib/capabilities";
import CueModal from "./components/CueModal";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import { useT, LanguageToggle } from "./lib/i18n";
import ClipEditor from "./screens/ClipEditor";
import ProjectView from "./screens/ProjectView";
import Upload from "./screens/Upload";

function Caps({ health }: { health: Health | null }) {
  const { t } = useT();
  if (!health) return null;
  const c = health.capabilities;
  const { count, total, pct } = capabilityReadiness(c);
  const summary = capabilitySummary(c, {
    synthetic: t("caps.synthetic"),
    gpuDetected: t("caps.gpuDetected"),
    ocrOff: t("caps.ocrOff"),
  });
  const isGpu = Boolean(c.gpu);
  return (
    <div className="caps" title="ClipForge Hardware & Engine Status — Click for Diagnostics">
      <span className="caps-engine-tag">
        <span className="engine-dot" />
        {isGpu ? "RTX 5060 Ti • NVDEC 650 FPS" : "CPU Engine"}
      </span>
      <span className="caps-summary">{summary}</span>
      <span className={"caps-badge " + (pct >= 80 ? "ok" : pct >= 50 ? "warn" : "bad")}>
        {count}/{total}
      </span>
    </div>
  );
}

export default function App() {
  const { t } = useT();
  const [health, setHealth] = useState<Health | null>(null);
  const [showCues, setShowCues] = useState(false);
  const [showDiag, setShowDiag] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <nav className="nav">
        <Link to="/" className="brand">
          <span className="mark">⚡</span>
          <span>ClipForge</span>
          <span className="version-tag">Studio</span>
        </Link>
        <div className="spacer" />
        <button
          className="caps-btn"
          onClick={() => setShowDiag(true)}
          title={t("diag.title")}
        >
          <Caps health={health} />
        </button>
        <button
          className="btn ghost sm cues-btn"
          onClick={() => setShowCues(true)}
          title={t("nav.cuesTitle")}
        >
          <span className="btn-icon">🎮</span> {t("nav.cues")}
        </button>
        <LanguageToggle />
        <Link
          to="/p/proj_demo/clip/c_demo_ace"
          className="btn ghost sm studio-nav-demo-btn"
          title="Open Interactive Streamer Studio Demo"
          style={{
            borderColor: "rgba(6, 182, 212, 0.45)",
            background: "rgba(6, 182, 212, 0.1)",
            color: "var(--accent-cyan, #06b6d4)",
            fontWeight: 700,
          }}
        >
          <span className="btn-icon">🎮</span> Studio Demo
        </Link>
        <Link to="/" className="btn primary sm">
          <span className="btn-icon">✨</span> {t("nav.newProject")}
        </Link>
      </nav>
      {showCues && <CueModal onClose={() => setShowCues(false)} />}
      {showDiag && <DiagnosticsPanel onClose={() => setShowDiag(false)} />}
      <Routes>
        <Route path="/" element={<Upload health={health} />} />
        <Route path="/demo" element={<Navigate to="/p/proj_demo/clip/c_demo_ace" replace />} />
        <Route path="/p/:projectId" element={<ProjectView />} />
        <Route path="/p/:projectId/clip/:clipId" element={<ClipEditor />} />
      </Routes>
    </div>
  );
}
