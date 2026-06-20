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
    use_cues: project.settings.use_cues,
    use_audio_events: project.settings.use_audio_events,
    cue_learning: project.settings.cue_learning,
    auto_length: project.settings.auto_length,
    lead_seconds: project.settings.lead_seconds,
    tail_seconds: project.settings.tail_seconds,
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
      use_cues: project.settings.use_cues,
      use_audio_events: project.settings.use_audio_events,
      cue_learning: project.settings.cue_learning,
      auto_length: project.settings.auto_length,
      lead_seconds: project.settings.lead_seconds,
      tail_seconds: project.settings.tail_seconds,
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
        if (!alive.current) return; // user navigated away â€” stop polling
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
  const contextAuto = renderDraft.lead_seconds === null && renderDraft.tail_seconds === null;
  const normalizedRenderDraft = {
    ...renderDraft,
    lead_seconds: contextAuto ? null : renderDraft.lead_seconds ?? 16,
    tail_seconds: contextAuto ? null : renderDraft.tail_seconds ?? 20,
  };

  const renderDirty = useMemo(() => {
    const s = project.settings;
    return (
      normalizedRenderDraft.power_mode !== s.power_mode ||
      normalizedRenderDraft.aspect !== s.aspect ||
      normalizedRenderDraft.burn_captions !== s.burn_captions ||
      normalizedRenderDraft.tighten !== s.tighten ||
      normalizedRenderDraft.denoise !== s.denoise ||
      normalizedRenderDraft.motion !== s.motion ||
      normalizedRenderDraft.facecam_layout !== s.facecam_layout ||
      normalizedRenderDraft.use_ocr !== s.use_ocr ||
      normalizedRenderDraft.use_vlm !== s.use_vlm ||
      normalizedRenderDraft.use_cues !== s.use_cues ||
      normalizedRenderDraft.use_audio_events !== s.use_audio_events ||
      normalizedRenderDraft.cue_learning !== s.cue_learning ||
      normalizedRenderDraft.auto_length !== s.auto_length ||
      normalizedRenderDraft.lead_seconds !== s.lead_seconds ||
      normalizedRenderDraft.tail_seconds !== s.tail_seconds
    );
  }, [normalizedRenderDraft, project.settings]);

  // Rank by virality across the whole project (independent of the current sort),
  // so the strongest clips always wear their "Top pick" ribbon.
  const scoreRank = useMemo(() => {
    const ranked = [...project.clips].sort((a, b) => b.score - a.score);
    const m: Record<string, number> = {};
    ranked.forEach((c, i) => (m[c.id] = i + 1));
    return m;
  }, [project.clips]);

  const ready = project.clips.filter((c) => c.export_url).length;
  const [learn, setLearn] = useState<Learning | null>(null);
  useEffect(() => {
    api.learning().then(setLearn).catch(() => {});
  }, [project.id]);

  const learnTitle = learn
    ? Object.values(learn.learned_top_features)
        .flatMap((m) => Object.entries(m).map(([k, v]) => `${k} ${Math.round(v * 100)}%`))
        .join(", ") || "Rate clips ðŸ‘/ðŸ‘Ž to personalize"
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
      await api.reprocess(project.id, normalizedRenderDraft as any);
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
                {project.content_type === "gameplay" ? "ðŸŽ® Gameplay" : "ðŸŽ™ Talking"}
              </span>
            )}
          </div>
          <span className="muted tiny">
            {fmtDuration(project.source?.duration ?? 0)} source -{" "}
            {project.clips.length} clips - {ready} rendered - {project.settings.aspect} -{" "}
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
              ðŸ§  Personalizing - {learn.likes}ðŸ‘ {learn.dislikes}ðŸ‘Ž
              {learn.trims > 0 ? ` - ${learn.trims}âœ‚` : ""}
            </span>
          )}
          {project.content_type === "gameplay" && (
            <button className="btn ghost sm" onClick={() => setShowCues(true)}
              title="Add and test game sounds or OCR phrases, then re-run detection">
              ðŸŽ¯ Game cues
            </button>
          )}
          <button className="btn ghost sm" onClick={rerun}
            title="Re-run on the same video â€” applies your ðŸ‘/ðŸ‘Ž, new cues, and settings">
            Neu berechnen
          </button>
          <Link className="btn ghost sm" to="/">
            Zurueck - Neues Projekt
          </Link>
          <a className="btn primary sm" href={api.exportBatchUrl(project.id)}>
            Alle exportieren ({ready})
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
            <div key={i}>Hinweis: {w}</div>
          ))}
        </div>
      )}

      {project.events?.length > 0 && (
        <div className="panel section" style={{ marginTop: 16 }}>
          <span className="muted tiny">
            Verwendete Ereignisse ({project.events.length}) - uebernommene Cues und Bildschirmtreffer in den finalen Clips
          </span>
          <div className="row" style={{ flexWrap: "wrap", gap: 6, marginTop: 8 }}>
            {project.events.slice(0, 30).map((e, i) => (
              <span key={i} className="pill"
                title={`${e.detail || e.label} - ${Math.round(e.confidence * 100)}% match`}>
                {e.source === "ocr" ? "OCR" : "Audio"} {e.label} - {fmtClock(e.t)}
              </span>
            ))}
            {project.events.length > 30 && (
              <span className="muted tiny">+{project.events.length - 30} weitere</span>
            )}
          </div>
        </div>
      )}

      <div className="panel section render-controls" style={{ marginTop: 16 }}>
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", gap: 14 }}>
          <div className="col">
            <h3>Render-Steuerung</h3>
            <span className="muted tiny">
              Passe Modus oder Ausgabe an und starte dieses Material erneut.
            </span>
          </div>
          <button className="btn primary sm" onClick={rerunWithDraft} disabled={!renderDirty}>
            Mit Einstellungen neu starten
          </button>
        </div>
        <div className="render-control-grid">
          <div className="field">
            <label>Leistungsmodus</label>
            <select
              className="input"
              value={renderDraft.power_mode}
              onChange={(e) => setRenderDraft((d) => ({ ...d, power_mode: e.target.value as any }))}
            >
              <option value="balanced">Ausgewogen</option>
              <option value="max_gpu">Max GPU</option>
              <option value="quality">Qualitaet</option>
            </select>
          </div>
          <div className="field">
            <label>Ausgabeformat</label>
            <select
              className="input"
              value={renderDraft.aspect}
              onChange={(e) => setRenderDraft((d) => ({ ...d, aspect: e.target.value }))}
            >
              <option value="9:16">9:16 vertikal</option>
              <option value="4:5">4:5 Feed</option>
              <option value="1:1">1:1 Quadrat</option>
              <option value="16:9">16:9 breit</option>
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
                <option value="split">Gestapelt</option>
                <option value="framed">PiP</option>
                <option value="off">Aus</option>
              </select>
            </div>
          )}
          <div className="field toggle-field">
            <label>Schalter</label>
            <div className="toggle-stack compact">
              <button
                className={"toggle" + (renderDraft.burn_captions ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, burn_captions: !d.burn_captions }))}
              >
                <span>Untertitel</span>
                <i>{renderDraft.burn_captions ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.tighten ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, tighten: !d.tighten }))}
              >
                <span>Jump cuts</span>
                <i>{renderDraft.tighten ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.motion === "push" ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, motion: d.motion === "push" ? "none" : "push" }))}
              >
                <span>Push-in</span>
                <i>{renderDraft.motion === "push" ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.denoise ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, denoise: !d.denoise }))}
              >
                <span>Clean voice</span>
                <i>{renderDraft.denoise ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_ocr ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_ocr: !d.use_ocr }))}
              >
                <span>OCR</span>
                <i>{renderDraft.use_ocr ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_vlm ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_vlm: !d.use_vlm }))}
              >
                <span>KI-Bildanalyse</span>
                <i>{renderDraft.use_vlm ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_audio_events ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_audio_events: !d.use_audio_events }))}
              >
                <span>Audio-Ereignisse</span>
                <i>{renderDraft.use_audio_events ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.use_cues ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, use_cues: !d.use_cues }))}
                title="Use installed custom game sounds as exact-match evidence"
              >
                <span>Custom sounds</span>
                <i>{renderDraft.use_cues ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.cue_learning ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, cue_learning: !d.cue_learning }))}
              >
                <span>Cue-Lernen</span>
                <i>{renderDraft.cue_learning ? "An" : "Aus"}</i>
              </button>
              <button
                className={"toggle" + (renderDraft.auto_length ? " on" : "")}
                onClick={() => setRenderDraft((d) => ({ ...d, auto_length: !d.auto_length }))}
              >
                <span>Auto length</span>
                <i>{renderDraft.auto_length ? "An" : "Aus"}</i>
              </button>
            </div>
          </div>
          {project.content_type === "gameplay" && (
            <div className="field wide timing-controls">
              <label>Clip context</label>
              <button
                className={"toggle" + (contextAuto ? " on" : "")}
                onClick={() =>
                  setRenderDraft((d) =>
                    d.lead_seconds === null && d.tail_seconds === null
                      ? { ...d, lead_seconds: 16, tail_seconds: 20 }
                      : { ...d, lead_seconds: null, tail_seconds: null },
                  )
                }
                style={{ marginBottom: 8 }}
              >
                <span>Automatic context</span>
                <i>{contextAuto ? "An" : "Aus"}</i>
              </button>
              {(renderDraft.lead_seconds !== null || renderDraft.tail_seconds !== null) && (
                <>
              <div className="range-label">
                <span>Before detected moment</span>
                <b>{renderDraft.lead_seconds ?? 16}s</b>
              </div>
              <input
                type="range"
                min={0}
                max={30}
                step={1}
                value={renderDraft.lead_seconds ?? 16}
                onChange={(e) => setRenderDraft((d) => ({ ...d, lead_seconds: Number(e.target.value) }))}
              />
              <div className="range-label">
                <span>After detected moment</span>
                <b>{renderDraft.tail_seconds ?? 20}s</b>
              </div>
              <input
                type="range"
                min={2}
                max={30}
                step={1}
                value={renderDraft.tail_seconds ?? 20}
                onChange={(e) => setRenderDraft((d) => ({ ...d, tail_seconds: Number(e.target.value) }))}
              />
                </>
              )}
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
          title="Change the output format now â€” re-renders every clip; moments, scores and captions stay the same"
          onChange={async (e) => {
            try {
              await api.setAspect(project.id, e.target.value);
              window.location.reload(); // show live render progress
            } catch (err: any) {
              setMontageErr(err?.message ?? "Could not change the format.");
            }
          }}
        >
          <option value="9:16">9:16 vertikal</option>
          <option value="4:5">4:5 Feed</option>
          <option value="1:1">1:1 Quadrat</option>
          <option value="16:9">16:9 breit</option>
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
          â†»
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
              rank={scoreRank[c.id]}
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
            ? "Tip: tick clips (âœ“ on the thumbnail) to combine them into a montage."
            : `${selected.length} selected â€” they'll play in the order you ticked them.`}
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
          {montaging ? <><span className="spinner" /> Buildingâ€¦</> : `ðŸŽ¬ Create montage (${selected.length})`}
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
                      {m.status === "failed" ? "Fehlgeschlagen" : "Renderingâ€¦"}
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
                        â¬‡ Download
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


