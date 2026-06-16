import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import type { Health, ProjectSummary, StyleTemplate } from "../lib/types";
import { fmtDuration, timeAgo } from "../lib/format";
import CueManager from "../components/CueManager";

const PLATFORMS = [
  { id: "tiktok", label: "TikTok" },
  { id: "reels", label: "Reels" },
  { id: "shorts", label: "Shorts" },
  { id: "generic", label: "Any" },
];

const POWER_MODES = [
  { id: "balanced", label: "Balanced", hint: "Fast default; keeps the PC responsive" },
  { id: "max_gpu", label: "Max GPU", hint: "Uses larger batches, more render workers, and bigger AI vision budget" },
  { id: "quality", label: "Quality", hint: "Slower, more visual context for AI scoring" },
];

const LENGTHS = [
  { label: "15–30s", min: 15, max: 30 },
  { label: "20–45s", min: 20, max: 45 },
  { label: "30–60s", min: 30, max: 60 },
  { label: "15–60s", min: 15, max: 60 },
];

export default function Upload({ health }: { health: Health | null }) {
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
  const [tighten, setTighten] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [motion, setMotion] = useState("none");
  const [facecamLayout, setFacecamLayout] = useState("auto");
  const [useOcr, setUseOcr] = useState(true);
  const [useVlm, setUseVlm] = useState(true);
  const [useAudioEvents, setUseAudioEvents] = useState(true);
  const [cueLearning, setCueLearning] = useState(true);
  const [autoLength, setAutoLength] = useState(false);
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
      setErr("Add a video file or paste a link first.");
      return;
    }
    setErr(null);
    setBusy(true);
    setPct(0);
    try {
      const len = LENGTHS[lenIdx];
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
        use_audio_events: useAudioEvents,
        cue_learning: cueLearning,
        auto_length: autoLength,
        lead_seconds: leadSeconds,
        tail_seconds: tailSeconds,
        onProgress: setPct,
      });
      nav(`/p/${project.id}`);
    } catch (e: any) {
      setErr(e.message ?? "Something went wrong.");
      setBusy(false);
    }
  };

  const del = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await api.deleteProject(id).catch(() => {});
    refreshProjects();
  };

  const urlDisabled = health ? !health.capabilities.url_import : false;
  const caps = health?.capabilities;
  const status = {
    captions: caps?.transcription && caps.transcription !== "synthetic" ? caps.transcription : "synthetic",
    cleanVoice: caps?.denoise ? "Ready" : "Unavailable",
    ocr: caps?.ocr ? String(caps.ocr) : "Unavailable",
    vlm: caps?.vlm ? caps.vlm_model ?? "Ready" : "Unavailable",
    audio: caps?.audio_events
      ? caps.clap_audio
        ? "CLAP"
        : caps.panns_audio
          ? "PANNs"
          : "Ready"
      : "Unavailable",
    cues: "Ready",
  };

  return (
    <div className="container">
      <div className="hero">
        <h1>One long video in. A week of short clips out.</h1>
        <p>
          Drop in a podcast, interview, or talk. ClipForge finds the best moments,
          reframes them vertical, captions them, and ranks each by predicted reach.
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
            <div className="big">🎬 {file.name}</div>
            <div className="muted tiny">
              {(file.size / 1024 / 1024).toFixed(1)} MB · ready to process
            </div>
            <button className="btn ghost sm" onClick={() => setFile(null)}>
              Choose a different file
            </button>
          </div>
        ) : (
          <>
            <div className="big">Drag &amp; drop a video here</div>
            <div className="muted tiny" style={{ marginTop: 6 }}>
              MP4, MOV, MKV, WEBM — up to a couple of hours
            </div>
            <div style={{ marginTop: 16 }}>
              <button className="btn" onClick={() => fileRef.current?.click()}>
                Browse files
              </button>
            </div>
            <div className="or">— or paste a link —</div>
            <div className="url-row">
              <input
                className="input"
                placeholder={
                  urlDisabled
                    ? "URL import unavailable in this environment"
                    : "https://youtube.com/watch?v=…"
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
          <label>Power mode</label>
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
                  <small>Recommended</small>
                )}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Content type</label>
          <div className="seg">
            {[
              { id: "auto", label: "Auto" },
              { id: "talking", label: "Talking" },
              { id: "gameplay", label: "Gameplay" },
            ].map((c) => (
              <button
                key={c.id}
                className={contentType === c.id ? "on" : ""}
                onClick={() => setContentType(c.id)}
                title={
                  c.id === "gameplay"
                    ? "Find epic moments (kills, goals) from audio spikes — no face tracking"
                    : c.id === "talking"
                      ? "Find the best spoken moments from the transcript"
                      : "Detect talking vs gameplay automatically"
                }
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Aspect ratio</label>
          <select className="input" value={aspect} onChange={(e) => setAspect(e.target.value)}>
            <option value="9:16">9:16 (Reels/Shorts/TikTok)</option>
            <option value="4:5">4:5 (Feed)</option>
            <option value="1:1">1:1 (Square)</option>
            <option value="16:9">16:9 (YouTube / edit in Premiere)</option>
          </select>
        </div>
        <div className="field">
          <label>Game profile <span className="muted tiny">(gameplay only)</span></label>
          <select className="input" value={gameProfile} onChange={(e) => setGameProfile(e.target.value)}
            title="Tunes gameplay highlight detection; works for any game">
            <option value="auto">Auto / Any game</option>
            <option value="valorant">Valorant</option>
            <option value="cs2">CS2</option>
            <option value="eafc">EA FC / FIFA</option>
            <option value="rocketleague">Rocket League</option>
            <option value="horror">Horror</option>
          </select>
          {cues && cues[gameProfile] && (
            <span className="muted tiny" title="Add reference sounds for pinpoint event detection (docs/GAME_CUES.md)">
              {cues[gameProfile].configured}/{cues[gameProfile].total} cues ·{" "}
              {cues[gameProfile].configured === 0 ? "audio-energy only" : "pinpoint events on"}
            </span>
          )}
        </div>
        {contentType !== "talking" && (
          <div className="field">
            <label>Facecam <span className="muted tiny">(gameplay only)</span></label>
            <select className="input" value={facecamLayout}
              onChange={(e) => setFacecamLayout(e.target.value)}
              title="When a streamer cam is detected: stack it above the gameplay, overlay it, or ignore it">
              <option value="auto">Auto (stack when detected)</option>
              <option value="split">Stacked (cam on top)</option>
              <option value="framed">Overlay (PiP cam)</option>
              <option value="off">Off (plain crop)</option>
            </select>
          </div>
        )}
        <div className="field wide">
          <label>Detection switches</label>
          <div className="toggle-stack compact capability-toggles">
            <button
              className={"toggle" + (useOcr ? " on" : "")}
              onClick={() => setUseOcr((v) => !v)}
              title="Read scoreboards, killfeed, victory text, and other on-screen cues"
            >
              <span>OCR</span>
              <small>{status.ocr}</small>
              <i>{useOcr ? "On" : "Off"}</i>
            </button>
            <button
              className={"toggle" + (useVlm ? " on" : "")}
              onClick={() => setUseVlm((v) => !v)}
              title="Use the local vision model to score action, expression, clarity, and boring frames"
            >
              <span>AI vision</span>
              <small>{status.vlm}</small>
              <i>{useVlm ? "On" : "Off"}</i>
            </button>
            <button
              className={"toggle" + (useAudioEvents ? " on" : "")}
              onClick={() => setUseAudioEvents((v) => !v)}
              title="Detect cheers, laughs, impact sounds, and CLAP zero-shot audio cues"
            >
              <span>Audio events</span>
              <small>{status.audio}</small>
              <i>{useAudioEvents ? "On" : "Off"}</i>
            </button>
            <button
              className={"toggle" + (cueLearning ? " on" : "")}
              onClick={() => setCueLearning((v) => !v)}
              title="Learn new reusable audio cues from OCR hits while keeping your cue packs"
            >
              <span>Cue learning</span>
              <small>{status.cues}</small>
              <i>{cueLearning ? "On" : "Off"}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>Captions</label>
          <button
            className={"toggle" + (burnCaptions ? " on" : "")}
            onClick={() => setBurnCaptions((v) => !v)}
            title="Burn captions into the exported clips"
          >
            <span>Burn captions</span>
            <small>{status.captions}</small>
            <i>{burnCaptions ? "On" : "Off"}</i>
          </button>
          <span className="muted tiny">Off = clean clips for Premiere/Resolve</span>
        </div>
        <div className="field">
          <label>Pacing & style</label>
          <div className="toggle-stack">
            <button
              className={"toggle" + (tighten ? " on" : "")}
              onClick={() => setTighten((v) => !v)}
              title="Cuts out pauses/dead air inside talking clips"
            >
              <span>Jump cuts</span>
              <i>{tighten ? "On" : "Off"}</i>
            </button>
            <button
              className={"toggle" + (motion === "push" ? " on" : "")}
              onClick={() => setMotion((v) => (v === "push" ? "none" : "push"))}
              title="Slow push-in across each clip"
            >
              <span>Slow push-in</span>
              <i>{motion === "push" ? "On" : "Off"}</i>
            </button>
            <button
              className={"toggle" + (denoise ? " on" : "")}
              onClick={() => setDenoise((v) => !v)}
              title="Isolate the voice from background music/game audio. Needs Demucs installed."
            >
              <span>Clean voice</span>
              <small>{status.cleanVoice}</small>
              <i>{denoise ? "On" : "Off"}</i>
            </button>
          </div>
        </div>
        <div className="field">
          <label>Optimize for</label>
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
          <label>Spoken language</label>
          <select
            className="input"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            title="Improves transcription and tunes moment detection to the language"
          >
            <option value="de">German (Deutsch)</option>
            <option value="en">English</option>
            <option value="auto">Auto-detect</option>
          </select>
        </div>
        <div className="field">
          <label>Clip length</label>
          <button
            className={"toggle" + (autoLength ? " on" : "")}
            onClick={() => setAutoLength((v) => !v)}
            title="Let ClipForge choose the clip length range for this platform and content"
            style={{ marginBottom: 8 }}
          >
            <span>Auto length</span>
            <i>{autoLength ? "On" : "Off"}</i>
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
            <label>Event padding</label>
            <div className="range-label">
              <span>Seconds before event</span>
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
              <span>Seconds after event</span>
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
          </div>
        )}
        <div className="field">
          <label>Max clips: {target}</label>
          <input
            type="range"
            min={3}
            max={20}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>Caption style</label>
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
          {gameProfile !== "auto" && (
            <CueManager game={gameProfile} cues={cues} onChange={setCues} />
          )}
          {/* Cross-game sounds (airhorn, hype, laugh…) — matched for any game. */}
          <CueManager game="common" cues={cues} onChange={setCues} />
        </>
      )}

      <div style={{ marginTop: 22, display: "flex", justifyContent: "center" }}>
        <button className="btn primary" onClick={submit} disabled={busy} style={{ minWidth: 240, justifyContent: "center" }}>
          {busy ? (
            pct < 100 && file ? (
              <>Uploading… {pct}%</>
            ) : (
              <>
                <span className="spinner" /> Starting…
              </>
            )
          ) : (
            <>✨ Generate clips</>
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
          <h3 style={{ marginBottom: 14 }}>Recent projects</h3>
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
                    {fmtDuration(p.duration)} source · {p.ready_clips}/{p.clip_count} clips ·{" "}
                    {timeAgo(p.created_at)}
                  </span>
                </div>
                <StatusPill status={p.status} pct={p.progress?.pct ?? 0} />
                <button className="btn sm danger" onClick={(e) => del(p.id, e)}>
                  Delete
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
  if (status === "ready") return <span className="pill" style={{ color: "var(--good)" }}>Ready</span>;
  if (status === "failed") return <span className="pill" style={{ color: "var(--bad)" }}>Failed</span>;
  if (status === "processing")
    return <span className="pill" style={{ color: "var(--warn)" }}>{Math.round(pct)}%</span>;
  return <span className="pill">Queued</span>;
}
