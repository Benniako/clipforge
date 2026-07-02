import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
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
  return (
    <div className="caps" title="ClipForge System-Erkennung — klicke für Details">
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
          <span className="mark">◆</span> ClipForge
        </Link>
        <div className="spacer" />
        <button
          className="caps-btn"
          onClick={() => setShowDiag(true)}
          title={t("diag.title")}
        >
          <Caps health={health} />
        </button>
        <LanguageToggle />
        <button
          className="btn ghost sm"
          onClick={() => setShowCues(true)}
          title={t("nav.cuesTitle")}
        >
          {t("nav.cues")}
        </button>
        <Link to="/" className="btn primary sm">
          {t("nav.newProject")}
        </Link>
      </nav>
      {showCues && <CueModal onClose={() => setShowCues(false)} />}
      {showDiag && <DiagnosticsPanel onClose={() => setShowDiag(false)} />}
      <Routes>
        <Route path="/" element={<Upload health={health} />} />
        <Route path="/p/:projectId" element={<ProjectView />} />
        <Route path="/p/:projectId/clip/:clipId" element={<ClipEditor />} />
      </Routes>
    </div>
  );
}
