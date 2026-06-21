import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import type { Health, ProjectSummary, StyleTemplate } from "../lib/types";
import { fmtDuration, timeAgo } from "../lib/format";
import { useT } from "../lib/i18n";
import CueLab from "../components/CueLab";
import CueManager from "../components/CueManager";

const PLATFORMS = [
  { id: "tiktok", label: "TikTok" },
  { id: "reels", label: "Reels" },
  { id: "shorts", label: "Shorts" },
  { id: "generic", label: "Beliebig" },
];

const POWER_MODES = [
  { id: "balanced", label: "Ausgewogen", hint: "Schneller Standardmodus und dein PC bleibt reaktionsfreudig" },
  { id: "max_gpu", label: "Max GPU", hint: "Nutzt größere Batches, mehr Render-Worker und mehr Budget für KI-Bildanalyse" },
  { id: "quality", label: "Qualität", hint: "Langsamer, aber mit mehr visuellem Kontext für die KI-Bewertung" },
];

const LENGTHS = [
  { label: "15-30s", min: 15, max: 30 },
  { label: "20-45s", min: 20, max: 45 },
  { label: "30-60s", min: 30, max: 60 },
  { label: "15-60s", min: 15, max: 60 },
];

export default function Upload({ health }: { health: Health | null }) {
  const { t } = useT();
  const nav = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [platform, setPlatform] = useState("tiktok");
  const [powerMode, setPowerMode] = useState("balanced");
  const [lenIdx, setLenIdx] = useState(3);
  const [target, setTarget] = useState(10);
  const [styleId, setStyleId] = useState("bold-pop");
  const [language, setLanguage] = useState("de");
  const [contentType, setContentType] = useState("auto");
  const [aspect, setAspect] = useState("9:16");
  const [burnCaptions, setBurnCaptions] = useState(true);
  const [gameProfile, setGameProfile] = useState("auto");
  const [detectionMode, setDetectionMode] = useState("zero_shot");
  const [audioCues, setAudioCues] = useState("");
  const [visualTextCues, setVisualTextCues] = useState("");
  const [vlmCues, setVlmCues] = useState("");
  const [tighten, setTighten] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [motion, setMotion] = useState("none");
  const [facecamLayout, setFacecamLayout] = useState("auto");
  const [useOcr, setUseOcr] = useState(true);
  const [useVlm, setUseVlm] = useState(true);
  const [useCues, setUseCues] = useState(() => localStorage.getItem("clipforge.useCues") !== "off");
  const [useAudioEvents, setUseAudioEvents] = useState(true);
  const [cueLearning, setCueLearning] = useState(true);
  const [autoLength, setAutoLength] = useState(false);
  const [manualContext, setManualContext] = useState(false);
  const [leadSeconds, setLeadSeconds] = useState(16);
  const [tailSeconds, setTailSeconds] = useState(20);
  const [styles, setStyles] = useState<StyleTemplate[]>([]);
  const [cues, setCues] = useState<CuesStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);

  // Auto-dismiss the error toast (no CSS animation drives it).
  useEffect(() => {
    if (!err) return;
    const id = setTimeout(() => setErr(null), 6000);
    return () => clearTimeout(id);
  }, [err]);

  useEffect(() => {
    api.styles().then(setStyles).catch(() => {});
    api.cues().then(setCues).catch(() => {});
    refreshProjects();
  }, []);

  useEffect(() => {
    const recommended = health?.capabilities.recommended_power_mode;
    if (recommended) setPowerMode(recommended);
  }, [health?.capabilities.recommended_power_mode]);

  useEffect(() => {
    localStorage.setItem("clipforge.useCues", useCues ? "on" : "off");
  }, [useCues]);

  const refreshProjects = () =>
    api.listProjects().then(setProjects).catch(() => {});

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) {
      setFile(f);
      setUrl("");
    }
  }, []);

  const submit = async () => {
    if (!file && !url.trim()) {
      setErr("Füge zuerst eine Videodatei hinzu oder füge einen Link ein.");
      return;
    }
    setErr(null);
    setBusy(true);
    setPct(0);
    try {
      const len = LENGTHS[lenIdx];
      const splitCues = (s: string) =>
        s.split(/[\n,]+/).map((x) => x.trim()).filter(Boolean);
      const audioPrompts = splitCues(audioCues);
      const visualCues = splitCues(visualTextCues);
      const vlmPrompts = splitCues(vlmCues);
      // Only send a game_config when the user changed something, so default
      // runs stay byte-identical to before this control existed.
      const gameConfig =
        detectionMode !== "zero_shot" ||
        audioPrompts.length ||
        visualCues.length ||
        vlmPrompts.length
          ? {
              detection_mode: detectionMode,
              audio_prompts: audioPrompts,
              visual_text_cues: visualCues,
              vlm_visual_prompts: vlmPrompts,
            }
          : undefined;
      const project = await api.createProject({
        file: file ?? undefined,
        url: url.trim() || undefined,
        platform,
        power_mode: powerMode,
        min_len: len.min,
        max_len: len.max,
        target_clips: target,
        style_id: styleId,
        language,
        content_type: contentType,
        aspect,
        burn_captions: burnCaptions,
        game_profile: gameProfile,
        tighten,
        denoise,
        motion,
        facecam_layout: facecamLayout,
        use_ocr: useOcr,
        use_vlm: useVlm,
        use_cues: useCues,
        use_audio_events: useAudioEvents,
        cue_learning: cueLearning,
        auto_length: autoLength,
        lead_seconds: manualContext ? leadSeconds : null,
        tail_seconds: manualContext ? tailSeconds : null,
        game_config: gameConfig,
        onProgress: setPct,
      });
      nav(`/p/${project.id}`);
    } catch (e: any) {
      setErr(e.message ?? "Etwas ist schiefgelaufen.");
      setBusy(false);
    }
  };

  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await api.deleteProject(id).catch(() => {});
    refreshProjects();
  };

  const cueLabGame = gameProfile !== "auto" ? gameProfile : "common";
  const updateVisualCues = (visual: Record<string, Record<string, string[]>>) =>
    setCues((prev) => {
      if (!prev) return prev;
      const next = { ...prev };
      const pack = next[cueLabGame] ?? {
          label: cueLabGame === "common" ? "Allgemein (alle Spiele)" : cueLabGame,
        configured: 0,
        total: 0,
        events: [],
      };
      next[cueLabGame] = { ...pack, visual: visual[cueLabGame] ?? {} };
      return next;
    });

  const urlDisabled = health ? !health.capabilities.url_import : false;
  const caps = health?.capabilities;
  const status = {
    captions: caps?.transcription && caps.transcription !== "synthetic" ? caps.transcription : "synthetic",
    cleanVoice: caps?.denoise ? "Bereit" : "Nicht verfügbar",
    ocr: caps?.ocr ? String(caps.ocr) : "Nicht verfügbar",
    vlm: caps?.vlm ? caps.vlm_model ?? "Bereit" : "Nicht verfügbar",
    audio: caps?.audio_events
      ? caps.clap_audio
        ? "CLAP"
        : caps.panns_audio
          ? "PANNs"
          : "Bereit"
      : "Nicht verfügbar",
    cues: "Bereit",
  };

  return (
    <div className="container">
      <div className="hero">
        <h1>Ein langes Video rein. Eine Woche Short-Clips raus.</h1>
        <p>
          Lade einen Podcast, ein Interview oder Gameplay hoch. ClipForge findet die
          stärksten Momente, setzt sie vertikal um, fügt Untertitel hinzu und sortiert
          sie nach erwartetem Potenzial.
        </p>
      </div>

      <div
        className={"dropzone" + (drag ? " drag" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
      >
        {file ? (
          <div className="col" style={{ alignItems: "center", gap: 8 }}>
            <div className="big">Video: {file.name}</div>
            <div className="muted tiny">
              {(file.size / 1024 / 1024).toFixed(1)} MB - bereit zur Verarbeitung
            </div>
            <button className="btn ghost sm" onClick={() => setFile(null)}>
              Andere Datei wählen
            </button>
          </div>
        ) : (
          <>
            <div className="big">Ziehe ein Video hierher</div>
            <div className="muted tiny" style={{ marginTop: 6 }}>
              MP4, MOV, MKV, WEBM - auch mehrere Stunden sind ok
            </div>
            <div style={{ marginTop: 16 }}>
              <button className="btn" onClick={() => fileRef.current?.click()}>
                Dateien auswählen
              </button>
            </div>
            <div className="or">- oder Link einfügen -</div>
            <div className="url-row">
              <input
                className="input"
                placeholder={
                  urlDisabled
                    ? "URL-Import ist in dieser Umgebung nicht verfügbar"
                    : "https://youtube.com/watch?v=..."
                }
                value={url}
                disabled={urlDisabled}
                onChange={(e) => setUrl(e.target.value)}
              />
            </div>
          </>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) {
              setFile(f);
              setUrl("");
            }
          }}
        />
      </div>

      <div className="settings-grid">
        <div className="field wide">
          <label>Leistungsmodus</label>
          <div className="seg power-seg">
            {POWER_MODES.map((m) => (
              <button
                key={m.id}
                className={powerMode === m.id ? "on" : ""}
                onClick={() => setPowerMode(m.id)}
                title={m.hint}
              >
                <span>{m.label}</span>
                {health?.capabilities.recommended_power_mode === m.id && (
                  <small>Empfohlen</small>
                )}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Inhaltstyp</label>
          <div className="seg">
            {[
              { id: "auto", label: "Auto" },
              { id: "talking", label: "Sprache" },
              { id: "gameplay", label: "Gameplay" },
            ].map((c) => (
              <button
                key={c.id}
                className={contentType === c.id ? "on" : ""}
                onClick={() => setContentType(c.id)}
                title={
                  c.id === "gameplay"
                    ? "Findet starke Momente wie Kills und Tore über Audio- und Spielszenen-Signale"
                    : c.id === "talking"
                      ? "Findet die besten gesprochenen Momente aus dem Transkript"
                      : "Erkennt automatisch, ob es um Sprache oder Gameplay geht"
                }
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Seitenverhältnis</label>
          <select className="input" value={aspect} onChange={(e) => setAspect(e.target.value)}>
            <option value="9:16">9:16 (Reels/Shorts/TikTok)</option>
            <option value="4:5">4:5 (Feed)</option>
            <option value="1:1">1:1 (Quadrat)</option>
            <option value="16:9">16:9 (YouTube / in Premiere bearbeiten)</option>
          </select>
        </div>
        <div className="field">
          <label>Spielprofil <span className="muted tiny">(nur Gameplay)</span></label>
          <select className="input" value={gameProfile} onChange={(e) => setGameProfile(e.target.value)}
            title="Passt die Highlight-Erkennung für Gameplay an und funktioniert auch für andere Spiele">
            <option value="auto">Auto / Beliebiges Spiel</option>
            <option value="valorant">Valorant</option>
            <option value="cs2">CS2</option>
            <option value="eafc">EA FC / FIFA</option>
            <option value="rocketleague">Rocket League</option>
            <option value="horror">Horror</option>
          </select>
          {cues && cues[gameProfile] && (
            <span className="muted tiny" title="Optional: Füge exakte Spielsounds hinzu. OCR und Audio-Erkennung funktionieren auch ohne sie.">
              {cues[gameProfile].configured}/{cues[gameProfile].total} Cues -{" "}
              {cues[gameProfile].configured === 0 ? "nur OCR/Audio" : "eigene Sounds aktiv"}
            </span>
          )}
        </div>
        {contentType !== "talking" && (
          <div className="field wide">
            <label>
              {t("gc.section")} <span className="muted tiny">{t("gc.sectionHint")}</span>
            </label>
            <select
              className="input"
              value={detectionMode}
              onChange={(e) => setDetectionMode(e.target.value)}
              title={t("gc.detectionMode")}
            >
              <option value="zero_shot">{t("gc.modeAuto")}</option>
              <option value="hybrid">{t("gc.modeHybrid")}</option>
              <option value="manual">{t("gc.modeManual")}</option>
            </select>
            <div className="row" style={{ gap: 8, marginTop: 8, flexWrap: "wrap" }}>
              <div className="col" style={{ flex: "1 1 220px", gap: 4 }}>
                <label className="muted tiny">{t("gc.audioCues")}</label>
                <input
                  className="input"
                  value={audioCues}
                  onChange={(e) => setAudioCues(e.target.value)}
                  placeholder={t("gc.placeholder")}
                  title={t("gc.audioCuesHint")}
                />
              </div>
              <div className="col" style={{ flex: "1 1 220px", gap: 4 }}>
                <label className="muted tiny">{t("gc.visualCues")}</label>
                <input
                  className="input"
                  value={visualTextCues}
                  onChange={(e) => setVisualTextCues(e.target.value)}
                  placeholder={t("gc.placeholder")}
                  title={t("gc.visualCuesHint")}
                />
              </div>
              <div className="col" style={{ flex: "1 1 220px", gap: 4 }}>
                <label className="muted tiny">{t("gc.vlmCues")}</label>
                <input
                  className="input"
                  value={vlmCues}
                  onChange={(e) => setVlmCues(e.target.value)}
                  placeholder={t("gc.placeholder")}
                  title={t("gc.vlmCuesHint")}
                />
              </div>
            </div>
          </div>
        )}
        {contentType !== "talking" && (
          <div className="field">
            <label>Facecam <span className="muted tiny">(nur Gameplay)</span></label>
            <select className="input" value={facecamLayout}
              onChange={(e) => setFacecamLayout(e.target.value)}
              title="Wenn eine Streamer-Cam erkannt wird: über dem Gameplay stapeln, als Overlay zeigen oder ignorieren">
              <option value="auto">Auto (bei Erkennung stapeln)</option>
              <option value="split">Gestapelt (Cam oben)</option>
              <option value="framed">Overlay (PiP-Cam)</option>
              <option value="off">Aus (nur Crop)</option>
            </select>
          </div>
        )}
        <div className="field wide">
          <label>Erkennungs-Schalter</label>
          <div className="toggle-stack compact capability-toggles">
            <button
              className={"toggle" + (useOcr ? " on" : "")}
              onClick={() => setUseOcr((v) => !v)}
              title="Liest Scoreboards, Killfeed, Siegesmeldungen und andere Bildschirmhinweise"
            >
              <span>OCR</span>
              <small>{status.ocr}</small>
              <i>{useOcr ? "An" : "Aus"}</i>
            </button>
            <button
              className={"toggle" + (useVlm ? " on" : "")}
              onClick={() => setUseVlm((v) => !v)}
              title="Nutzt das lokale Vision-Modell für Action, Ausdruck, Klarheit und langweilige Frames"
            >
              <span>KI-Bildanalyse</span>
              <small>{status.vlm}</small>
              <i>{useVlm ? "An" : "Aus"}</i>
            </button>
            <button
              className={"toggle" + (useAudioEvents ? " on" : "")}
              onClick={() => setUseAudioEvents((v) => !v)}
              title="Erkennt Jubel, Lachen, Impacts und CLAP-Zero-Shot-Audio-Cues"
            >
              <span>Audio-Ereignisse</span>
              <small>{status.audio}</small>
              <i>{useAudioEvents ? "An" : "Aus"}</i>
            </button>
            <button
              className={"toggle" + (cueLearning ? " on" : "")}
              onClick={() => setCueLearning((v) => !v)}
              title="Lernt neue wiederverwendbare Audio-Cues aus OCR-Treffern und behält deine Cue-Pakete"
            >
              <span>Cue-Lernen</span>
              <small>{status.cues}</small>
              <i>{cueLearning ? "An" : "Aus"}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>Untertitel</label>
          <button
            className={"toggle" + (burnCaptions ? " on" : "")}
            onClick={() => setBurnCaptions((v) => !v)}
            title="Brennt Untertitel direkt in die exportierten Clips ein"
          >
            <span>Untertitel einbrennen</span>
            <small>{status.captions}</small>
            <i>{burnCaptions ? "An" : "Aus"}</i>
          </button>
          <span className="muted tiny">Aus = saubere Clips für Premiere oder Resolve</span>
        </div>
        <div className="field">
          <label>Rhythmus & Stil</label>
          <div className="toggle-stack">
            <button
              className={"toggle" + (tighten ? " on" : "")}
              onClick={() => setTighten((v) => !v)}
              title="Schneidet Pausen und Leerlauf in Sprach-Clips heraus"
            >
              <span>Jump-Cuts</span>
              <i>{tighten ? "An" : "Aus"}</i>
            </button>
            <button
              className={"toggle" + (motion === "push" ? " on" : "")}
              onClick={() => setMotion((v) => (v === "push" ? "none" : "push"))}
              title="Langsamer Push-in über den ganzen Clip"
            >
              <span>Langsamer Push-in</span>
              <i>{motion === "push" ? "An" : "Aus"}</i>
            </button>
            <button
              className={"toggle" + (denoise ? " on" : "")}
              onClick={() => setDenoise((v) => !v)}
              title="Trennt die Stimme von Musik und Spielsound. Demucs muss installiert sein."
            >
              <span>Saubere Stimme</span>
              <small>{status.cleanVoice}</small>
              <i>{denoise ? "An" : "Aus"}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>Optimieren für</label>
          <div className="seg">
            {PLATFORMS.map((p) => (
              <button
                key={p.id}
                className={platform === p.id ? "on" : ""}
                onClick={() => setPlatform(p.id)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Gesprochene Sprache</label>
          <select
            className="input"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            title="Verbessert die Transkription und passt die Moment-Erkennung an die Sprache an"
          >
            <option value="de">Deutsch</option>
            <option value="en">Englisch</option>
            <option value="auto">Automatisch erkennen</option>
          </select>
        </div>
        <div className="field">
          <label>Clip-Länge</label>
          <button
            className={"toggle" + (autoLength ? " on" : "")}
            onClick={() => setAutoLength((v) => !v)}
            title="Lässt ClipForge die passende Clip-Länge für Plattform und Inhalt wählen"
            style={{ marginBottom: 8 }}
          >
            <span>Auto-Länge</span>
            <i>{autoLength ? "An" : "Aus"}</i>
          </button>
          <select
            className="input"
            value={lenIdx}
            disabled={autoLength}
            onChange={(e) => setLenIdx(Number(e.target.value))}
          >
            {LENGTHS.map((l, i) => (
              <option key={i} value={i}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        {contentType !== "talking" && (
          <div className="field wide timing-controls">
            <label>Clip-Kontext</label>
            <button
              className={"toggle" + (!manualContext ? " on" : "")}
              onClick={() => setManualContext((v) => !v)}
              title="Automatisch nutzt die beste Vorlauf- und Nachlaufzeit für den erkannten Moment"
              style={{ marginBottom: 8 }}
            >
              <span>Automatischer Kontext</span>
              <i>{manualContext ? "Aus" : "An"}</i>
            </button>
            {manualContext && (
              <>
            <div className="range-label">
              <span>Vor dem erkannten Moment</span>
              <b>{leadSeconds}s</b>
            </div>
            <input
              type="range"
              min={0}
              max={30}
              step={1}
              value={leadSeconds}
              onChange={(e) => setLeadSeconds(Number(e.target.value))}
            />
            <div className="range-label">
              <span>Nach dem erkannten Moment</span>
              <b>{tailSeconds}s</b>
            </div>
            <input
              type="range"
              min={2}
              max={30}
              step={1}
              value={tailSeconds}
              onChange={(e) => setTailSeconds(Number(e.target.value))}
            />
              </>
            )}
          </div>
        )}
        <div className="field">
          <label>Maximale Clips: {target}</label>
          <input
            type="range"
            min={3}
            max={20}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>Untertitel-Stil</label>
          <select
            className="input"
            value={styleId}
            onChange={(e) => setStyleId(e.target.value)}
          >
            {styles.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {contentType !== "talking" && (
        <>
          {file && (
            <CueLab
              game={cueLabGame}
              visual={cues?.[cueLabGame]?.visual}
              sourceFile={file}
              onVisualChange={updateVisualCues}
              onAudioChange={() => api.cues().then(setCues).catch(() => {})}
            />
          )}
          {gameProfile !== "auto" && (
            <CueManager
              game={gameProfile}
              cues={cues}
              onChange={setCues}
              enabled={useCues}
              onToggle={setUseCues}
            />
          )}
        </>
      )}

      <div style={{ marginTop: 22, display: "flex", justifyContent: "center" }}>
        <button className="btn primary" onClick={submit} disabled={busy} style={{ minWidth: 240, justifyContent: "center" }}>
          {busy ? (
            pct < 100 && file ? (
              <>Lade hoch... {pct}%</>
            ) : (
              <>
                <span className="spinner" /> Starte...
              </>
            )
          ) : (
            <>Clips erzeugen</>
          )}
        </button>
      </div>
      {err && (
        <div className="toast err" onClick={() => setErr(null)}>
          {err}
        </div>
      )}

      {projects.length > 0 && (
        <div style={{ marginTop: 44 }}>
          <h3 style={{ marginBottom: 14 }}>Letzte Projekte</h3>
          <div className="proj-list">
            {projects.map((p) => (
              <div
                key={p.id}
                className="proj-row"
                onClick={() => nav(`/p/${p.id}`)}
                role="button"
              >
                <div className="col" style={{ flex: 1 }}>
                  <span className="name">{p.name}</span>
                  <span className="muted tiny">
                    {fmtDuration(p.duration)} Quelle - {p.ready_clips}/{p.clip_count} Clips -{" "}
                    {timeAgo(p.created_at)}
                  </span>
                </div>
                <StatusPill status={p.status} pct={p.progress?.pct ?? 0} />
                <button className="btn sm danger" onClick={(e) => del(p.id, e)}>
                  Löschen
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusPill({ status, pct }: { status: string; pct: number }) {
  if (status === "ready") return <span className="pill" style={{ color: "var(--good)" }}>Bereit</span>;
  if (status === "failed") return <span className="pill" style={{ color: "var(--bad)" }}>Fehlgeschlagen</span>;
  if (status === "paused") return <span className="pill" style={{ color: "var(--warn)" }}>Pausiert</span>;
  if (status === "processing")
    return <span className="pill" style={{ color: "var(--warn)" }}>{Math.round(pct)}%</span>;
  return <span className="pill">Warteschlange</span>;
}





