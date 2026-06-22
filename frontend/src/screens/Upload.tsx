import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import type { Health, ProjectSummary, StyleTemplate } from "../lib/types";
import { fmtDuration, timeAgo } from "../lib/format";
import { useT } from "../lib/i18n";
import CueLab from "../components/CueLab";
import CueManager from "../components/CueManager";

const PLATFORMS: { id: string; label?: string; labelKey?: string }[] = [
  { id: "tiktok", label: "TikTok" },
  { id: "reels", label: "Reels" },
  { id: "shorts", label: "Shorts" },
  { id: "generic", labelKey: "up.platformGeneric" },
];

const POWER_MODES = [
  { id: "balanced", labelKey: "up.powerBalanced", hintKey: "up.powerBalancedHint" },
  { id: "max_gpu", labelKey: "up.powerMaxGpu", hintKey: "up.powerMaxGpuHint" },
  { id: "quality", labelKey: "up.powerQuality", hintKey: "up.powerQualityHint" },
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
      setErr(t("up.errNoSource"));
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
      setErr(e.message ?? t("up.errGeneric"));
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
          label: cueLabGame === "common" ? t("up.cueLabCommonLabel") : cueLabGame,
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
    cleanVoice: caps?.denoise ? t("up.statusReady") : t("up.statusUnavailable"),
    ocr: caps?.ocr ? String(caps.ocr) : t("up.statusUnavailable"),
    vlm: caps?.vlm ? caps.vlm_model ?? t("up.statusReady") : t("up.statusUnavailable"),
    audio: caps?.audio_events
      ? caps.clap_audio
        ? "CLAP"
        : caps.panns_audio
          ? "PANNs"
          : t("up.statusReady")
      : t("up.statusUnavailable"),
    cues: t("up.statusReady"),
  };

  return (
    <div className="container">
      <div className="hero">
        <h1>{t("up.heroTitle")}</h1>
        <p>{t("up.heroText")}</p>
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
            <div className="big">{t("up.videoLabel", { name: file.name })}</div>
            <div className="muted tiny">
              {t("up.fileReady", { size: (file.size / 1024 / 1024).toFixed(1) })}
            </div>
            <button className="btn ghost sm" onClick={() => setFile(null)}>
              {t("up.chooseOtherFile")}
            </button>
          </div>
        ) : (
          <>
            <div className="big">{t("up.dropHere")}</div>
            <div className="muted tiny" style={{ marginTop: 6 }}>
              {t("up.dropFormats")}
            </div>
            <div style={{ marginTop: 16 }}>
              <button className="btn" onClick={() => fileRef.current?.click()}>
                {t("up.chooseFiles")}
              </button>
            </div>
            <div className="or">{t("up.orPasteLink")}</div>
            <div className="url-row">
              <input
                className="input"
                placeholder={
                  urlDisabled ? t("up.urlDisabled") : t("up.urlPlaceholder")
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
          <label>{t("up.powerMode")}</label>
          <div className="seg power-seg">
            {POWER_MODES.map((m) => (
              <button
                key={m.id}
                className={powerMode === m.id ? "on" : ""}
                onClick={() => setPowerMode(m.id)}
                title={t(m.hintKey)}
              >
                <span>{t(m.labelKey)}</span>
                {health?.capabilities.recommended_power_mode === m.id && (
                  <small>{t("up.recommended")}</small>
                )}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>{t("up.contentType")}</label>
          <div className="seg">
            {[
              { id: "auto", labelKey: "up.contentAuto" },
              { id: "talking", labelKey: "up.contentTalking" },
              { id: "gameplay", labelKey: "up.contentGameplay" },
            ].map((c) => (
              <button
                key={c.id}
                className={contentType === c.id ? "on" : ""}
                onClick={() => setContentType(c.id)}
                title={
                  c.id === "gameplay"
                    ? t("up.contentGameplayHint")
                    : c.id === "talking"
                      ? t("up.contentTalkingHint")
                      : t("up.contentAutoHint")
                }
              >
                {t(c.labelKey)}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>{t("up.aspect")}</label>
          <select className="input" value={aspect} onChange={(e) => setAspect(e.target.value)}>
            <option value="9:16">{t("up.aspect916")}</option>
            <option value="4:5">{t("up.aspect45")}</option>
            <option value="1:1">{t("up.aspect11")}</option>
            <option value="16:9">{t("up.aspect169")}</option>
          </select>
        </div>
        <div className="field">
          <label>{t("up.gameProfile")} <span className="muted tiny">{t("up.gameplayOnly")}</span></label>
          <select className="input" value={gameProfile} onChange={(e) => setGameProfile(e.target.value)}
            title={t("up.gameProfileTitle")}>
            <option value="auto">{t("up.gameProfileAuto")}</option>
            <option value="valorant">Valorant</option>
            <option value="cs2">CS2</option>
            <option value="eafc">{t("up.gameProfileEafc")}</option>
            <option value="rocketleague">{t("up.gameProfileRocketLeague")}</option>
            <option value="horror">{t("up.gameProfileHorror")}</option>
          </select>
          {cues && cues[gameProfile] && (
            <span className="muted tiny" title={t("up.cuesHint")}>
              {t("up.cuesCount", {
                configured: cues[gameProfile].configured,
                total: cues[gameProfile].total,
                state:
                  cues[gameProfile].configured === 0
                    ? t("up.cuesOcrOnly")
                    : t("up.cuesCustomActive"),
              })}
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
            <label>{t("up.facecam")} <span className="muted tiny">{t("up.gameplayOnly")}</span></label>
            <select className="input" value={facecamLayout}
              onChange={(e) => setFacecamLayout(e.target.value)}
              title={t("up.facecamTitle")}>
              <option value="auto">{t("up.facecamAuto")}</option>
              <option value="split">{t("up.facecamSplit")}</option>
              <option value="framed">{t("up.facecamFramed")}</option>
              <option value="off">{t("up.facecamOff")}</option>
            </select>
          </div>
        )}
        <div className="field wide">
          <label>{t("up.detectionToggles")}</label>
          <div className="toggle-stack compact capability-toggles">
            <button
              className={"toggle" + (useOcr ? " on" : "")}
              onClick={() => setUseOcr((v) => !v)}
              title={t("up.ocrTitle")}
            >
              <span>OCR</span>
              <small>{status.ocr}</small>
              <i>{useOcr ? t("up.on") : t("up.off")}</i>
            </button>
            <button
              className={"toggle" + (useVlm ? " on" : "")}
              onClick={() => setUseVlm((v) => !v)}
              title={t("up.aiVisionTitle")}
            >
              <span>{t("up.aiVision")}</span>
              <small>{status.vlm}</small>
              <i>{useVlm ? t("up.on") : t("up.off")}</i>
            </button>
            <button
              className={"toggle" + (useAudioEvents ? " on" : "")}
              onClick={() => setUseAudioEvents((v) => !v)}
              title={t("up.audioEventsTitle")}
            >
              <span>{t("up.audioEvents")}</span>
              <small>{status.audio}</small>
              <i>{useAudioEvents ? t("up.on") : t("up.off")}</i>
            </button>
            <button
              className={"toggle" + (cueLearning ? " on" : "")}
              onClick={() => setCueLearning((v) => !v)}
              title={t("up.cueLearningTitle")}
            >
              <span>{t("up.cueLearning")}</span>
              <small>{status.cues}</small>
              <i>{cueLearning ? t("up.on") : t("up.off")}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>{t("up.captions")}</label>
          <button
            className={"toggle" + (burnCaptions ? " on" : "")}
            onClick={() => setBurnCaptions((v) => !v)}
            title={t("up.burnCaptionsTitle")}
          >
            <span>{t("up.burnCaptions")}</span>
            <small>{status.captions}</small>
            <i>{burnCaptions ? t("up.on") : t("up.off")}</i>
          </button>
          <span className="muted tiny">{t("up.captionsOffNote")}</span>
        </div>
        <div className="field">
          <label>{t("up.rhythmStyle")}</label>
          <div className="toggle-stack">
            <button
              className={"toggle" + (tighten ? " on" : "")}
              onClick={() => setTighten((v) => !v)}
              title={t("up.jumpCutsTitle")}
            >
              <span>{t("up.jumpCuts")}</span>
              <i>{tighten ? t("up.on") : t("up.off")}</i>
            </button>
            <button
              className={"toggle" + (motion === "push" ? " on" : "")}
              onClick={() => setMotion((v) => (v === "push" ? "none" : "push"))}
              title={t("up.slowPushTitle")}
            >
              <span>{t("up.slowPush")}</span>
              <i>{motion === "push" ? t("up.on") : t("up.off")}</i>
            </button>
            <button
              className={"toggle" + (denoise ? " on" : "")}
              onClick={() => setDenoise((v) => !v)}
              title={t("up.cleanVoiceTitle")}
            >
              <span>{t("up.cleanVoice")}</span>
              <small>{status.cleanVoice}</small>
              <i>{denoise ? t("up.on") : t("up.off")}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>{t("up.optimizeFor")}</label>
          <div className="seg">
            {PLATFORMS.map((p) => (
              <button
                key={p.id}
                className={platform === p.id ? "on" : ""}
                onClick={() => setPlatform(p.id)}
              >
                {p.labelKey ? t(p.labelKey) : p.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>{t("up.spokenLanguage")}</label>
          <select
            className="input"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            title={t("up.spokenLanguageTitle")}
          >
            <option value="de">{t("up.langGerman")}</option>
            <option value="en">{t("up.langEnglish")}</option>
            <option value="auto">{t("up.langAuto")}</option>
          </select>
        </div>
        <div className="field">
          <label>{t("up.clipLength")}</label>
          <button
            className={"toggle" + (autoLength ? " on" : "")}
            onClick={() => setAutoLength((v) => !v)}
            title={t("up.autoLengthTitle")}
            style={{ marginBottom: 8 }}
          >
            <span>{t("up.autoLength")}</span>
            <i>{autoLength ? t("up.on") : t("up.off")}</i>
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
            <label>{t("up.clipContext")}</label>
            <button
              className={"toggle" + (!manualContext ? " on" : "")}
              onClick={() => setManualContext((v) => !v)}
              title={t("up.autoContextTitle")}
              style={{ marginBottom: 8 }}
            >
              <span>{t("up.autoContext")}</span>
              <i>{manualContext ? t("up.off") : t("up.on")}</i>
            </button>
            {manualContext && (
              <>
            <div className="range-label">
              <span>{t("up.beforeMoment")}</span>
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
              <span>{t("up.afterMoment")}</span>
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
          <label>{t("up.maxClips", { target })}</label>
          <input
            type="range"
            min={3}
            max={20}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>{t("up.captionStyle")}</label>
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
              <>{t("up.uploading", { pct })}</>
            ) : (
              <>
                <span className="spinner" /> {t("up.starting")}
              </>
            )
          ) : (
            <>{t("up.generateClips")}</>
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
          <h3 style={{ marginBottom: 14 }}>{t("up.recentProjects")}</h3>
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
                    {t("up.projectMeta", {
                      duration: fmtDuration(p.duration),
                      ready: p.ready_clips,
                      total: p.clip_count,
                      ago: timeAgo(p.created_at),
                    })}
                  </span>
                </div>
                <StatusPill status={p.status} pct={p.progress?.pct ?? 0} />
                <button className="btn sm danger" onClick={(e) => del(p.id, e)}>
                  {t("up.delete")}
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
  const { t } = useT();
  if (status === "ready") return <span className="pill" style={{ color: "var(--good)" }}>{t("up.pillReady")}</span>;
  if (status === "failed") return <span className="pill" style={{ color: "var(--bad)" }}>{t("up.pillFailed")}</span>;
  if (status === "paused") return <span className="pill" style={{ color: "var(--warn)" }}>{t("up.pillPaused")}</span>;
  if (status === "processing")
    return <span className="pill" style={{ color: "var(--warn)" }}>{Math.round(pct)}%</span>;
  return <span className="pill">{t("up.pillQueued")}</span>;
}





