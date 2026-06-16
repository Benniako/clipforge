import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { Project } from "../lib/types";
import { fmtClock, fmtDuration, scoreColor } from "../lib/format";
import ClipCard from "./ClipCard";
import CueModal from "./CueModal";

interface Learning {
  total_ratings: number;
  likes: number;
  dislikes: number;
  trims: number;
  personalized: boolean;
  learned_top_features: Record<string, Record<string, number>>;
}

type Sort = "score" | "timeline" | "duration";

const POWER_LABELS: Record<string, string> = {
  balanced: "Balanced",
  max_gpu: "Max GPU",
  quality: "Quality",
};

export default function ClipGridView({
  project,
  onChange,
}: {
  project: Project;
  onChange: (p: Project) => void;
}) {
  const [sort, setSort] = useState<Sort>("score");
  const [minScore, setMinScore] = useState(0);
  const [selected, setSelected] = useState<string[]>([]); // selection order = montage order
  const [activeSelectedId, setActiveSelectedId] = useState<string | null>(null);
  const [montaging, setMontaging] = useState(false);
  const [rerenderingSelected, setRerenderingSelected] = useState(false);
  const [montageErr, setMontageErr] = useState<string | null>(null);
  const [showCues, setShowCues] = useState(false);
  const [selectedPreviewMode, setSelectedPreviewMode] = useState<"rendered" | "original">("rendered");
  const [renderDraft, setRenderDraft] = useState({
    power_mode: project.settings.power_mode,
    aspect: project.settings.aspect,
    burn_captions: project.settings.burn_captions,
    tighten: project.settings.tighten,
    denoise: project.settings.denoise,
    motion: project.settings.motion,
    facecam_layout: project.settings.facecam_layout,
    use_ocr: project.settings.use_ocr,
    use_vlm: project.settings.use_vlm,
    use_audio_events: project.settings.use_audio_events,
    cue_learning: project.settings.cue_learning,
    auto_length: project.settings.auto_length,
    lead_seconds: project.settings.lead_seconds ?? 16,
    tail_seconds: project.settings.tail_seconds ?? 20,
  });
  const alive = useRef(true); // stops the montage poll after unmount

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    if (!montageErr) return;
    const id = setTimeout(() => setMontageErr(null), 6000);
    return () => clearTimeout(id);
  }, [montageErr]);

  useEffect(() => {
    setRenderDraft({
      power_mode: project.settings.power_mode,
      aspect: project.settings.aspect,
      burn_captions: project.settings.burn_captions,
      tighten: project.settings.tighten,
      denoise: project.settings.denoise,
      motion: project.settings.motion,
      facecam_layout: project.settings.facecam_layout,
      use_ocr: project.settings.use_ocr,
      use_vlm: project.settings.use_vlm,
      use_audio_events: project.settings.use_audio_events,
      cue_learning: project.settings.cue_learning,
      auto_length: project.settings.auto_length,
      lead_seconds: project.settings.lead_seconds ?? 16,
      tail_seconds: project.settings.tail_seconds ?? 20,
    });
  }, [project.id, project.settings]);

  const toggleSelect = (id: string) =>
    setSelected((s) => {
      const exists = s.includes(id);
      const next = exists ? s.filter((x) => x !== id) : [...s, id];
      setActiveSelectedId(exists ? (activeSelectedId === id ? next[0] ?? null : activeSelectedId) : id);
      return next;
    });

  const makeMontage = async () => {
    if (selected.length < 2) return;
    setMontaging(true);
    setMontageErr(null);
    try {
      await api.createMontage(project.id, selected);
      // poll until the new montage finishes rendering
      for (let i = 0; i < 90; i++) {
        if (!alive.current) return; // user navigated away — stop polling
        const p = await api.getProject(project.id);
        if (!alive.current) return;
        onChange(p);
        const pending = p.montages.some((m) => m.status === "rendering");
        if (!pending) break;
        await new Promise((r) => setTimeout(r, 1500));
      }
      setSelected([]);
    } catch (e: any) {
      // a failed *request* creates no montage card, so say it out loud
      if (alive.current) setMontageErr(e?.message ?? "Could not create the montage.");
    } finally {
      if (alive.current) setMontaging(false);
    }
  };

  const clips = useMemo(() => {
    const list = project.clips.filter((c) => c.score >= minScore);
    const by: Record<Sort, (a: any, b: any) => number> = {
      score: (a, b) => b.score - a.score,
      timeline: (a, b) => a.start - b.start,
      duration: (a, b) => b.end - b.start - (a.end - a.start),
    };
    return [...list].sort(by[sort]);
  }, [project.clips, sort, minScore]);

  const selectedClips = useMemo(
    () => selected.map((id) => project.clips.find((c) => c.id === id)).filter(Boolean) as Project["clips"],
    [selected, project.clips],
  );
  const activeSelected =
    selectedClips.find((c) => c.id === activeSelectedId) ?? selectedClips[0] ?? null;
  const selectedOriginalSrc =
    project.source && activeSelected
      ? `/media/${project.source.path}#t=${activeSelected.start.toFixed(3)},${activeSelected.end.toFixed(3)}`
      : undefined;
  const selectedPreviewSrc =
    selectedPreviewMode === "original" ? selectedOriginalSrc : activeSelected?.export_url ?? undefined;

  const renderDirty = useMemo(() => {
    const s = project.settings;
    return (
      renderDraft.power_mode !== s.power_mode ||
      renderDraft.aspect !== s.aspect ||
      renderDraft.burn_captions !== s.burn_captions ||
      renderDraft.tighten !== s.tighten ||
      renderDraft.denoise !== s.denoise ||
      renderDraft.motion !== s.motion ||
      renderDraft.facecam_layout !== s.facecam_layout ||
      renderDraft.use_ocr !== s.use_ocr ||
      renderDraft.use_vlm !== s.use_vlm ||
      renderDraft.use_audio_events !== s.use_audio_events ||
      renderDraft.cue_learning !== s.cue_learning ||
      renderDraft.auto_length !== s.auto_length ||
      renderDraft.lead_seconds !== (s.lead_seconds ?? 16) ||
      renderDraft.tail_seconds !== (s.tail_seconds ?? 20)
    );
  }, [renderDraft, project.settings]);

  const ready = project.clips.filter((c) => c.export_url).length;
  const [learn, setLearn] = useState<Learning | null>(null);
  useEffect(() => {
    api.learning().then(setLearn).catch(() => {});
  }, [project.id]);

  const learnTitle = learn
    ? Object.values(learn.learned_top_features)
        .flatMap((m) => Object.entries(m).map(([k, v]) => `${k} ${Math.round(v * 100)}%`))
        .join(", ") || "Rate clips 👍/👎 to personalize"
    : "";

  const resetLearning = async () => {
    await api.resetLearning().catch(() => {});
    setLearn(await api.learning().catch(() => null));
  };

  const rerun = async () => {
    try {
      await api.reprocess(project.id);
      window.location.reload(); // restart the processing view + polling
    } catch (e: any) {
      // e.g. 409 while a previous run is still processing
      setMontageErr(e?.message ?? "Could not re-run this project.");
    }
  };

  const rerunWithDraft = async () => {
    try {
      await api.reprocess(project.id, renderDraft as any);
      window.location.reload();
    } catch (e: any) {
      setMontageErr(e?.message ?? "Could not re-run with these settings.");
    }
  };

  const refresh = async () => onChange(await api.getProject(project.id));

  const rerenderSelected = async () => {
    if (selected.length === 0) return;
    setRerenderingSelected(true);
    setMontageErr(null);
    try {
      onChange(await api.rerenderClips(project.id, selected));
      for (let i = 0; i < 90; i++) {
        if (!alive.current) return;
        const p = await api.getProject(project.id);
        if (!alive.current) return;
        onChange(p);
        const pending = p.clips.some((c) => selected.includes(c.id) && c.status === "rendering");
        if (!pending) break;
        await new Promise((r) => setTimeout(r, 1500));
      }
    } catch (e: any) {
      if (alive.current) setMontageErr(e?.message ?? "Could not re-render selected clips.");
    } finally {
      if (alive.current) setRerenderingSelected(false);
    }
  };

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="col">
          <div className="row">
            <h2>{project.name}</h2>
            {project.content_type && (
              <span className="pill" style={{ color: project.content_type === "gameplay" ? "#ff9f43" : "var(--accent)" }}>
                {project.content_type === "gameplay" ? "🎮 Gameplay" : "🎙 Talking"}
              </span>
            )}
          </div>
          <span className="muted tiny">
            {fmtDuration(project.source?.duration ?? 0)} source ·{" "}
            {project.clips.length} clips · {ready} rendered · {project.settings.aspect} ·{" "}
            {POWER_LABELS[project.settings.power_mode] ?? "Balanced"}
          </span>
        </div>
        <div className="row">
          {learn && learn.total_ratings > 0 && (
            <span
              className="pill"
              title={`Learned: ${learnTitle}. Click to reset.`}
              onClick={resetLearning}
              style={{ cursor: "pointer", color: "var(--good)" }}
            >
              🧠 Personalizing · {learn.likes}👍 {learn.dislikes}👎
              {learn.trims > 0 ? ` · ${learn.trims}✂` : ""}
            </span>
          )}
          {project.content_type === "gameplay" && (
            <button className="btn ghost sm" onClick={() => setShowCues(true)}
              title="Add reference game sounds so key moments (kills, goals…) are detected exactly — then Re-run">
              🎯 Game cues
            </button>
          )}
          <button className="btn ghost sm" onClick={rerun}
            title="Re-run on the same video — applies your 👍/👎, new cues, and settings">
            ↻ Re-run
          </button>
          <Link className="btn ghost sm" to="/">
            ← New project
          </Link>
          <a className="btn primary sm" href={api.exportBatchUrl(project.id)}>
            ⬇ Export all ({ready})
          </a>
        </div>
      </div>

      {project.warnings?.length > 0 && (
        <div
          style={{
            marginTop: 16, padding: "12px 16px", borderRadius: 10,
            border: "1px solid #5a4a2b", background: "#2a2414", color: "#f5d98a",
            fontSize: 13,
          }}
        >
          {project.warnings.map((w, i) => (
            <div key={i}>⚠ {w}</div>
          ))}
        </div>
      )}

      {project.events?.length > 0 && (
        <div className="panel section" style={{ marginTop: 16 }}>
          <span className="muted tiny">
            Detected events ({project.events.length}) — the cues &amp; on-screen text the highlights keyed off
          </span>
          <div className="row" style={{ flexWrap: "wrap", gap: 6, marginTop: 8 }}>
            {project.events.slice(0, 30).map((e, i) => (
              <span key={i} className="pill"
                title={`${e.detail || e.label} · ${Math.round(e.confidence * 100)}% match`}>
                {e.source === "ocr" ? "🔤" : "🔊"} {e.label} · {fmtClock(e.t)}
              </span>
            ))}
            {project.events.length > 30 && (
              <span className="muted tiny">+{project.events.length - 30} more</span>
            )}
          </div>
        </div>
      )}

      <div className="panel section render-controls" style={{ marginTop: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", gap: 14 }}>
          <div className="col">
            <h3>Render controls</h3>
            <span className="muted tiny">
              Change engine mode or output treatment, then re-run this source.
            </span>
          </div>
          <button className="btn primary sm" onClick={rerunWithDraft} disabled={!renderDirty}>
            Re-run with controls
          </button>
        </div>
        <div className="render-control-grid">
          <div className="field">
            <label>Power mode</label>
            <select
              className="input"
              value={renderDraft.power_mode}
              onChange={(e) => setRenderDraft((d) => ({ ...d, power_mode: e.target.value as any }))}
            >
              <option value="balanced">Balanced</option>
              <option value="max_gpu">Max GPU</option>
              <option value="quality">Quality</option>
            </select>
          </div>
          <div className="field">
            <label>Output aspect</label>
            <select
              className="input"
              value={renderDraft.aspect}
              onChange={(e) => setRenderDraft((d) => ({ ...d, aspect: e.target.value }))}
            >
              <option value="9:16">9:16 vertical</option>
              <option value="4:5">4:5 feed</option>
              <option value="1:1">1:1 square</option>
              <option value="16:9">16:9 wide</option>
            </select>
          </div>
          {project.content_type === "gameplay" && (
            <div className="field">
              <label>Facecam</label>
              <select
                className="input"
                value={renderDraft.facecam_layout}
                onChange={(e) => setRenderDraft((d) => ({ ...d, facecam_layout: e.target.value }))}
              >
                <option value="auto">Auto</option>
                <option value="split">Stacked</option>
                <option value="framed">PiP</option>
                <option value="off">Off</option>
              </select>
            </div>
          )}
          <div className="field toggle-field">
            <label>Toggles</label>
            <div className="toggle-stack compact">
              <button
                className={"toggle" + (renderDraft.burn_captions ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, burn_captions: !d.burn_captions }))}
              >
                <span>Captions</span>
                <i>{renderDraft.burn_captions ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.tighten ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, tighten: !d.tighten }))}
              >
                <span>Jump cuts</span>
                <i>{renderDraft.tighten ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.motion === "push" ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, motion: d.motion === "push" ? "none" : "push" }))}
              >
                <span>Push-in</span>
                <i>{renderDraft.motion === "push" ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.denoise ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, denoise: !d.denoise }))}
              >
                <span>Clean voice</span>
                <i>{renderDraft.denoise ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_ocr ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_ocr: !d.use_ocr }))}
              >
                <span>OCR</span>
                <i>{renderDraft.use_ocr ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_vlm ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_vlm: !d.use_vlm }))}
              >
                <span>AI vision</span>
                <i>{renderDraft.use_vlm ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_audio_events ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_audio_events: !d.use_audio_events }))}
              >
                <span>Audio events</span>
                <i>{renderDraft.use_audio_events ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.cue_learning ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, cue_learning: !d.cue_learning }))}
              >
                <span>Cue learning</span>
                <i>{renderDraft.cue_learning ? "On" : "Off"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.auto_length ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, auto_length: !d.auto_length }))}
              >
                <span>Auto length</span>
                <i>{renderDraft.auto_length ? "On" : "Off"}</i>
              </button>
            </div>
          </div>
          {project.content_type === "gameplay" && (
            <div className="field wide timing-controls">
              <label>Event padding</label>
              <div className="range-label">
                <span>Seconds before event</span>
                <b>{renderDraft.lead_seconds}s</b>
              </div>
              <input
                type="range"
                min={0}
                max={30}
                step={1}
                value={renderDraft.lead_seconds}
                onChange={(e) => setRenderDraft((d) => ({ ...d, lead_seconds: Number(e.target.value) }))}
              />
              <div className="range-label">
                <span>Seconds after event</span>
                <b>{renderDraft.tail_seconds}s</b>
              </div>
              <input
                type="range"
                min={2}
                max={30}
                step={1}
                value={renderDraft.tail_seconds}
                onChange={(e) => setRenderDraft((d) => ({ ...d, tail_seconds: Number(e.target.value) }))}
              />
            </div>
          )}
        </div>
      </div>

      <div className="grid-head" style={{ marginTop: 22 }}>
        <span className="muted tiny">Sort</span>
        <div className="seg" style={{ width: "auto" }}>
          {(["score", "timeline", "duration"] as Sort[]).map((s) => (
            <button
              key={s}
              className={sort === s ? "on" : ""}
              style={{ minWidth: 84, textTransform: "capitalize" }}
              onClick={() => setSort(s)}
            >
              {s === "score" ? "Virality" : s}
            </button>
          ))}
        </div>
        <span className="muted tiny" style={{ marginLeft: 10 }}>Format</span>
        <select
          className="input"
          style={{ width: "auto", padding: "7px 10px" }}
          value={project.settings.aspect}
          title="Change the output format now — re-renders every clip; moments, scores and captions stay the same"
          onChange={async (e) => {
            try {
              await api.setAspect(project.id, e.target.value);
              window.location.reload(); // show live render progress
            } catch (err: any) {
              setMontageErr(err?.message ?? "Could not change the format.");
            }
          }}
        >
          <option value="9:16">9:16 vertical</option>
          <option value="4:5">4:5 feed</option>
          <option value="1:1">1:1 square</option>
          <option value="16:9">16:9 wide</option>
        </select>
        <div className="spacer" />
        <span className="muted tiny">Min score: {minScore}+</span>
        <input
          type="range"
          min={0}
          max={90}
          step={5}
          value={minScore}
          onChange={(e) => setMinScore(Number(e.target.value))}
          style={{ width: 160 }}
        />
        <button className="btn ghost sm" onClick={refresh} title="Refresh">
          ↻
        </button>
      </div>

      {clips.length === 0 ? (
        <div className="empty">No clips match this filter. Lower the minimum score.</div>
      ) : (
        <div className="clip-grid">
          {clips.map((c) => (
            <ClipCard
              key={c.id}
              clip={c}
              projectId={project.id}
              selected={selected.includes(c.id)}
              onToggleSelect={toggleSelect}
            />
          ))}
        </div>
      )}

      {selectedClips.length > 0 && (
        <div className="panel section selected-preview">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
            <div className="col">
              <h3>Selected preview</h3>
              <span className="muted tiny">
                {selectedClips.length} clip{selectedClips.length === 1 ? "" : "s"} in montage order
              </span>
            </div>
            <div className="row selected-preview-actions">
              <div className="seg preview-tabs">
                <button
                  className={selectedPreviewMode === "rendered" ? "on" : ""}
                  onClick={() => setSelectedPreviewMode("rendered")}
                >
                  Rendered
                </button>
                <button
                  className={selectedPreviewMode === "original" ? "on" : ""}
                  onClick={() => setSelectedPreviewMode("original")}
                >
                  Original
                </button>
              </div>
              <button
                className="btn ghost sm"
                onClick={rerenderSelected}
                disabled={rerenderingSelected}
              >
                {rerenderingSelected ? <><span className="spinner" /> Rerendering...</> : "Rerender selected"}
              </button>
              <button className="btn ghost sm" onClick={() => setSelected([])}>
                Clear
              </button>
            </div>
          </div>
          <div className="selected-preview-layout">
            <div className="selected-video">
              {selectedPreviewSrc ? (
                <video
                  key={`${selectedPreviewMode}-${activeSelected?.id}-${selectedPreviewSrc}`}
                  src={selectedPreviewSrc}
                  controls
                  playsInline
                  poster={selectedPreviewMode === "rendered" ? activeSelected?.thumb_url ?? undefined : undefined}
                />
              ) : (
                <div className="empty">Select a clip to preview it.</div>
              )}
            </div>
            <div className="selected-list">
              {selectedClips.map((c, i) => (
                <button
                  key={c.id}
                  className={"selected-row" + (activeSelected?.id === c.id ? " on" : "")}
                  onClick={() => setActiveSelectedId(c.id)}
                >
                  <span className="order">{i + 1}</span>
                  <span
                    className="mini-thumb"
                    style={c.thumb_url ? { backgroundImage: `url(${c.thumb_url})` } : undefined}
                  />
                  <span className="selected-title">{c.title || "Untitled clip"}</span>
                  <span className="muted tiny">{fmtDuration(c.tightened_duration ?? c.end - c.start)}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Montage builder */}
      <div className="row" style={{ marginTop: 22, gap: 12, flexWrap: "wrap" }}>
        <span className="muted tiny">
          {selected.length === 0
            ? "Tip: tick clips (✓ on the thumbnail) to combine them into a montage."
            : `${selected.length} selected — they'll play in the order you ticked them.`}
        </span>
        <div className="spacer" style={{ flex: 1 }} />
        {selected.length > 0 && (
          <button className="btn ghost sm" onClick={() => setSelected([])}>
            Clear
          </button>
        )}
        <button
          className="btn primary sm"
          onClick={makeMontage}
          disabled={selected.length < 2 || montaging}
        >
          {montaging ? <><span className="spinner" /> Building…</> : `🎬 Create montage (${selected.length})`}
        </button>
      </div>

      {montageErr && <div className="toast err">{montageErr}</div>}

      {showCues && <CueModal onClose={() => setShowCues(false)} />}

      {project.montages.length > 0 && (
        <div style={{ marginTop: 28 }}>
          <h3 style={{ marginBottom: 12 }}>Montages</h3>
          <div className="clip-grid">
            {project.montages.map((m) => (
              <div className="clip-card" key={m.id}>
                <div
                  className="thumb"
                  style={{
                    aspectRatio: "16 / 10",
                    backgroundImage: m.thumb_url ? `url(${m.thumb_url})` : undefined,
                  }}
                >
                  {m.status !== "ready" && (
                    <span className="status-chip" style={{ color: m.status === "failed" ? "var(--bad)" : "var(--warn)" }}>
                      {m.status === "failed" ? "Failed" : "Rendering…"}
                    </span>
                  )}
                  {m.status === "ready" && <span className="dur">{fmtDuration(m.duration)}</span>}
                </div>
                <div className="clip-body">
                  <span className="score-badge" style={{ ["--c" as string]: scoreColor(m.score) }}>
                    <span className="ring" style={{ ["--p" as string]: m.score }}>
                      <i>{m.score}</i>
                    </span>
                    <span>virality</span>
                  </span>
                  <div className="clip-title">{m.title}</div>
                  <div className="factors">
                    {m.factors.slice(0, 2).map((f, i) => (
                      <span className="factor" key={i} title={f.detail}>{f.label}</span>
                    ))}
                  </div>
                  {m.status === "ready" && m.export_url && (
                    <div className="card-actions">
                      <a className="btn sm" href={api.downloadMontageUrl(project.id, m.id)} download>
                        ⬇ Download
                      </a>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
