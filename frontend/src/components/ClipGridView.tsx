import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { ImportSettings, PowerMode, Project } from "../lib/types";
import { fmtClock, fmtDuration } from "../lib/format";
import { mediaTimeUrl } from "../lib/media";
import { useT } from "../lib/i18n";
import ClipCard from "./ClipCard";
import CueModal from "./CueModal";
import GoalsByMatch from "./GoalsByMatch";
import ScoreBadge from "./ScoreBadge";
import VisualCueCalibration from "./VisualCueCalibration";

interface Learning {
  total_ratings: number;
  likes: number;
  dislikes: number;
  trims: number;
  personalized: boolean;
  learned_top_features: Record<string, Record<string, number>>;
}

type Sort = "score" | "timeline" | "duration";
type ProjectClip = Project["clips"][number];

const POWER_LABEL_KEYS: Record<string, string> = {
  balanced: "up.powerBalanced",
  max_gpu: "up.powerMaxGpu",
  quality: "up.powerQuality",
};

const SORT_LABEL_KEYS: Record<Sort, string> = {
  score: "cg.sortScore",
  timeline: "cg.sortTimeline",
  duration: "cg.sortDuration",
};

export default function ClipGridView({
  project,
  onChange,
  onPhoneView,
}: {
  project: Project;
  onChange: (p: Project) => void;
  /** Switch the project to the phone-style swipe review. */
  onPhoneView?: () => void;
}) {
  const { t } = useT();
  const [sort, setSort] = useState<Sort>("score");
  const [minScore, setMinScore] = useState(0);
  const [selected, setSelected] = useState<string[]>([]); // selection order = montage order
  const [activeSelectedId, setActiveSelectedId] = useState<string | null>(null);
  const [montaging, setMontaging] = useState(false);
  const [rerenderingSelected, setRerenderingSelected] = useState(false);
  const [montageErr, setMontageErr] = useState<string | null>(null);
  const [showCues, setShowCues] = useState(false);
  const [scanTab, setScanTab] = useState<"events" | "visual">("events");
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
        if (!alive.current) return; // user navigated away - stop polling
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
      if (alive.current) setMontageErr(e?.message ?? t("cg.errMontage"));
    } finally {
      if (alive.current) setMontaging(false);
    }
  };

  const clips = useMemo(() => {
    const list = project.clips.filter((c) => c.score >= minScore);
    const by: Record<Sort, (a: ProjectClip, b: ProjectClip) => number> = {
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
    activeSelected
      ? mediaTimeUrl(project.source?.path, activeSelected.start, activeSelected.end)
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
        .join(", ") || t("cg.learnEmpty")
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
      setMontageErr(e?.message ?? t("cg.errRerun"));
    }
  };

  const rerunWithDraft = async () => {
    try {
      await api.reprocess(project.id, normalizedRenderDraft as Partial<ImportSettings>);
      window.location.reload();
    } catch (e: any) {
      setMontageErr(e?.message ?? t("cg.errRerunDraft"));
    }
  };

  const refresh = async () => {
    try {
      onChange(await api.getProject(project.id));
    } catch { /* refresh errors are transient; UI stays consistent */ }
  };

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
      if (alive.current) setMontageErr(e?.message ?? t("cg.errRerender"));
    } finally {
      if (alive.current) setRerenderingSelected(false);
    }
  };

  const onOff = (on: boolean) => (on ? t("up.on") : t("up.off"));
  const showScan = project.events?.length > 0 || (project.content_type === "gameplay" && project.source);

  return (
    <div className="container">
      <div className="project-head">
        <div className="col project-title">
          <Link className="back-link" to="/">← {t("cg.back")}</Link>
          <div className="row">
            <h2>{project.name}</h2>
            {project.content_type && (
              <span className="pill">
                {project.content_type === "gameplay" ? t("cg.typeGameplay") : t("cg.typeTalking")}
              </span>
            )}
          </div>
          <div className="project-meta">
            <span><b>{fmtDuration(project.source?.duration ?? 0)}</b> {t("cg.metaSource")}</span>
            <span><b>{project.clips.length}</b> {t("cg.metaClips")}</span>
            <span><b>{ready}</b> {t("cg.metaRendered")}</span>
            <span><b>{project.settings.aspect}</b></span>
            <span>{t(POWER_LABEL_KEYS[project.settings.power_mode] ?? "up.powerBalanced")}</span>
          </div>
        </div>
        <div className="project-actions">
          {learn && learn.total_ratings > 0 && (
            <span
              className="pill ok"
              title={t("cg.learnTitle", { list: learnTitle })}
              onClick={resetLearning}
              style={{ cursor: "pointer" }}
            >
              {t("cg.learnPill", { likes: learn.likes, dislikes: learn.dislikes })}
              {learn.trims > 0 ? t("cg.learnTrims", { trims: learn.trims }) : ""}
            </span>
          )}
          {onPhoneView && (
            <div className="seg view-seg">
              <button className="on" aria-pressed="true">{t("pv.viewGrid")}</button>
              <button onClick={onPhoneView}>{t("pv.viewPhone")}</button>
            </div>
          )}
          {project.content_type === "gameplay" && (
            <button className="btn ghost sm" onClick={() => setShowCues(true)}
              title={t("cg.cuesTitle")}>
              {t("nav.cues")}
            </button>
          )}
          <button className="btn ghost sm" onClick={rerun}
            title={t("cg.rerunTitle")}>
            {t("cg.rerun")}
          </button>
          <a className="btn ghost sm" href={api.exportPremiereUrl(project.id)}>
            {t("cg.premiere")}
          </a>
          <a className="btn primary sm" href={api.exportBatchUrl(project.id)}>
            {t("cg.exportAll", { count: ready })}
          </a>
        </div>
      </div>

      {project.warnings?.length > 0 && (
        <div className="notice-list">
          {project.warnings.map((w, i) => {
            const sev = typeof w === "string" ? "warn" : w.severity;
            const message = typeof w === "string" ? w : w.message;
            const tag = sev === "error" ? t("notice.error") : sev === "info" ? t("notice.info") : t("notice.warn");
            return (
              <div key={i} className={"notice " + (sev === "error" ? "error" : sev === "info" ? "info" : "warn")}>
                <strong>{tag}:</strong> {message}
              </div>
            );
          })}
        </div>
      )}

      <div className="grid-head">
        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <span className="eyebrow">{t("cg.filter")}</span>
          <div className="seg">
            <button
              className={minScore === 0 ? "on" : ""}
              onClick={() => setMinScore(0)}
            >
              {t("cg.filterAll", { count: project.clips.length })}
            </button>
            <button
              className={minScore === 85 ? "on" : ""}
              onClick={() => setMinScore(85)}
            >
              {t("cg.filter85", { count: project.clips.filter((c) => c.score >= 85).length })}
            </button>
            <button
              className={minScore === 70 ? "on" : ""}
              onClick={() => setMinScore(70)}
            >
              {t("cg.filter70", { count: project.clips.filter((c) => c.score >= 70).length })}
            </button>
          </div>
        </div>

        <div className="spacer" />

        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <span className="eyebrow">{t("cg.sort")}</span>
          <div className="seg">
            {(["score", "timeline", "duration"] as Sort[]).map((s) => (
              <button
                key={s}
                className={sort === s ? "on" : ""}
                onClick={() => setSort(s)}
              >
                {t(SORT_LABEL_KEYS[s])}
              </button>
            ))}
          </div>

          <select
            className="input"
            value={project.settings.aspect}
            title={t("cg.aspectTitle")}
            aria-label={t("cg.aspectTitle")}
            onChange={async (e) => {
              try {
                await api.setAspect(project.id, e.target.value);
                window.location.reload();
              } catch (err: any) {
                setMontageErr(err?.message ?? t("cg.errAspect"));
              }
            }}
          >
            <option value="9:16">{t("cg.aspect916")}</option>
            <option value="4:5">{t("cg.aspect45")}</option>
            <option value="1:1">{t("cg.aspect11")}</option>
            <option value="16:9">{t("cg.aspect169")}</option>
          </select>

          <button className="btn ghost sm" onClick={refresh} title={t("cg.refresh")}>
            {t("cg.refresh")}
          </button>
          <button className="btn ghost sm" onClick={() => setSelected(clips.map((c) => c.id))} title={t("cg.selectAll")}>
            {t("cg.selectAll")}
          </button>
          {selected.length > 0 && (
            <button className="btn ghost sm" onClick={() => setSelected([])} title={t("cg.deselectTitle")}>
              {t("cg.deselect", { count: selected.length })}
            </button>
          )}
        </div>
      </div>

      <GoalsByMatch project={project} onChange={onChange} />

      {clips.length === 0 ? (
        <div className="empty">{t("cg.emptyFilter")}</div>
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

      {selected.length === 0 && clips.length > 1 && (
        <p className="grid-hint muted tiny">{t("cg.montageHint")}</p>
      )}

      {selectedClips.length > 0 && (
        <div className="panel section selected-preview">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 12, flexWrap: "wrap" }}>
            <div className="col">
              <h3>{t("cg.selPreview")}</h3>
              <span className="muted tiny">
                {selectedClips.length === 1
                  ? t("cg.selCountOne")
                  : t("cg.selCountMany", { count: selectedClips.length })}
              </span>
            </div>
            <div className="row selected-preview-actions">
              <div className="seg preview-tabs">
                <button
                  className={selectedPreviewMode === "rendered" ? "on" : ""}
                  onClick={() => setSelectedPreviewMode("rendered")}
                >
                  {t("ce.tabRendered")}
                </button>
                <button
                  className={selectedPreviewMode === "original" ? "on" : ""}
                  onClick={() => setSelectedPreviewMode("original")}
                >
                  {t("ce.tabOriginal")}
                </button>
              </div>
              <button
                className="btn ghost sm"
                onClick={rerenderSelected}
                disabled={rerenderingSelected}
              >
                {rerenderingSelected ? <><span className="spinner" /> {t("cg.rerendering")}</> : t("cg.rerenderSel")}
              </button>
            </div>
          </div>
          <div className="row" style={{ gap: 8, marginTop: 8, flexWrap: "wrap" }}>
            {/* Batch style change */}
            <select className="input" style={{ width: "auto" }}
              value=""
              aria-label={t("cg.applyStyle")}
              onChange={async (e) => {
                const styleId = e.target.value;
                if (!styleId) return;
                e.target.value = "";
                // Apply the selected style to all selected clips
                for (const c of selectedClips) {
                  try {
                    await api.editClip(project.id, c.id, { style_id: styleId });
                  } catch (err) {
                    console.warn("Failed to update style for clip", c.id, err);
                  }
                }
                // Refresh project
                const p = await api.getProject(project.id);
                onChange(p);
              }}
            >
              <option value="">{t("cg.applyStyle")}</option>
              {[...new Set(project.clips.map(c => c.captions.style_id))].map(sid => (
                <option key={sid} value={sid}>{sid}</option>
              ))}
            </select>
            {/* Batch download as ZIP */}
            {selectedClips.some(c => c.status === "ready" && c.export_url) && (
              <a className="btn sm ghost"
                href={api.exportBatchUrl(project.id)}
                download>
                {t("cg.zipSel", { count: selectedClips.filter(c => c.export_url).length })}
              </a>
            )}
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
                <div className="empty">{t("cg.pickToWatch")}</div>
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
                  <span className="selected-title">{c.title || t("cc.untitled")}</span>
                  <span className="muted tiny">{fmtDuration(c.tightened_duration ?? c.end - c.start)}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Montage builder: a selection bar that stays in view while clips are picked */}
      {selected.length > 0 && (
        <div className="montage-bar">
          <span className="montage-count">{selected.length}</span>
          <span className="muted tiny">
            {selected.length < 2 ? t("cg.montageNeedTwo") : t("cg.montageSelected")}
          </span>
          <div className="spacer" />
          <button className="btn ghost sm" onClick={() => setSelected([])}>
            {t("cg.clear")}
          </button>
          <button
            className="btn primary sm"
            onClick={makeMontage}
            disabled={selected.length < 2 || montaging}
          >
            {montaging ? <><span className="spinner" /> {t("cg.montageBuilding")}</> : t("cg.montageCreate", { count: selected.length })}
          </button>
        </div>
      )}

      {montageErr && <div className="toast err">{montageErr}</div>}

      {showCues && <CueModal onClose={() => setShowCues(false)} />}

      {project.montages.length > 0 && (
        <div>
          <h3 className="section-title">{t("cg.montages")}</h3>
          <div className="clip-grid">
            {project.montages.map((m) => (
              <div className="clip-card" key={m.id}>
                <div
                  className="thumb"
                  style={m.thumb_url ? { backgroundImage: `url(${m.thumb_url})` } : undefined}
                >
                  {m.status !== "ready" && (
                    <span className="status-chip" style={{ color: m.status === "failed" ? "var(--bad)" : "var(--warn)" }}>
                      {m.status === "failed" ? t("cg.montageFailed") : t("cg.montageRendering")}
                    </span>
                  )}
                  {m.status === "ready" && <span className="dur">{fmtDuration(m.duration)}</span>}
                </div>
                <div className="clip-body">
                  <div className="clip-top">
                    <ScoreBadge score={m.score} />
                  </div>
                  <div className="clip-title">{m.title}</div>
                  {m.factors.length > 0 && (
                    <div className="factors inline">
                      {m.factors.slice(0, 2).map((f, i) => (
                        <span className="factor" key={i} title={f.detail}>{f.label}<span className="w">+{Math.round(f.weight)}</span></span>
                      ))}
                    </div>
                  )}
                  {m.status === "ready" && m.export_url && (
                    <div className="card-actions">
                      <a className="btn sm primary" href={api.downloadMontageUrl(project.id, m.id)} download>
                        {t("cc.download")}
                      </a>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="fold-group project-folds">
        <h3 className="fold-group-title eyebrow">{t("up.secOutput")} &amp; {t("up.secDetect")}</h3>
        {showScan && (
          <details className="fold">
            <summary>
              <span>{t("cg.scan")}</span>
              <span className="fold-summary">{t("cg.eventsCount", { count: project.events?.length ?? 0 })}</span>
            </summary>
            <div className="fold-body">
            <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
              <span className="muted tiny">{t("cg.scanIntro")}</span>
              {project.content_type === "gameplay" && project.source && (
                <div className="seg scan-tabs">
                  <button className={scanTab === "events" ? "on" : ""} onClick={() => setScanTab("events")}>
                    {t("cg.tabEvents")}
                  </button>
                  <button className={scanTab === "visual" ? "on" : ""} onClick={() => setScanTab("visual")}>
                    {t("cg.tabVisual")}
                  </button>
                </div>
              )}
            </div>
            {scanTab === "events" || project.content_type !== "gameplay" || !project.source ? (
              <>
                <span className="muted tiny">
                  {t("cg.eventsUsed", { count: project.events.length })}
                </span>
                <div className="row" style={{ flexWrap: "wrap", gap: 6, marginTop: 8 }}>
                  {project.events.slice(0, 30).map((e, i) => (
                    <span key={i} className="pill"
                      title={t("cg.eventMatch", { detail: e.detail || e.label, pct: Math.round(e.confidence * 100) })}>
                      {e.source === "ocr" ? "OCR" : "Audio"} {e.label} - {fmtClock(e.t)}
                    </span>
                  ))}
                  {project.events.length > 30 && (
                    <span className="muted tiny">{t("cg.moreEvents", { count: project.events.length - 30 })}</span>
                  )}
                  {project.events.length === 0 && (
                    <span className="muted tiny">{t("cg.noEvents")}</span>
                  )}
                </div>
              </>
            ) : (
              <VisualCueCalibration project={project} />
            )}
            </div>
          </details>
        )}

        <details className="fold render-controls">
          <summary>
            <span>{t("cg.render")}</span>
            <span className="fold-summary">
              {t(POWER_LABEL_KEYS[project.settings.power_mode] ?? "up.powerBalanced")} · {project.settings.aspect}
            </span>
          </summary>
          <div className="fold-body">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", gap: 14, flexWrap: "wrap", marginBottom: 12 }}>
            <span className="muted tiny">{t("cg.renderIntro")}</span>
            <button className="btn primary sm" onClick={rerunWithDraft} disabled={!renderDirty}>
              {t("cg.rerunWith")}
            </button>
          </div>
          <div className="render-control-grid">
            <div className="field">
              <label>{t("up.powerMode")}</label>
              <select
                className="input"
                value={renderDraft.power_mode}
                onChange={(e) => setRenderDraft((d) => ({ ...d, power_mode: e.target.value as PowerMode }))}
              >
                <option value="balanced">{t("up.powerBalanced")}</option>
                <option value="max_gpu">{t("up.powerMaxGpu")}</option>
                <option value="quality">{t("up.powerQuality")}</option>
              </select>
            </div>
            <div className="field">
              <label>{t("cg.outputFormat")}</label>
              <select
                className="input"
                value={renderDraft.aspect}
                onChange={(e) => setRenderDraft((d) => ({ ...d, aspect: e.target.value }))}
              >
                <option value="9:16">{t("cg.aspect916")}</option>
                <option value="4:5">{t("cg.aspect45")}</option>
                <option value="1:1">{t("cg.aspect11")}</option>
                <option value="16:9">{t("cg.aspect169")}</option>
              </select>
            </div>
            {project.content_type === "gameplay" && (
              <div className="field">
                <label>{t("up.facecam")}</label>
                <select
                  className="input"
                  value={renderDraft.facecam_layout}
                  onChange={(e) => setRenderDraft((d) => ({ ...d, facecam_layout: e.target.value }))}
                >
                  <option value="auto">{t("cg.camAuto")}</option>
                  <option value="split">{t("cg.camSplit")}</option>
                  <option value="framed">{t("cg.camPip")}</option>
                  <option value="off">{t("cg.camOff")}</option>
                </select>
              </div>
            )}
            <div className="field toggle-field">
              <label>{t("cg.switches")}</label>
              <div className="toggle-stack compact">
                <button
                  className={"toggle" + (renderDraft.burn_captions ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, burn_captions: !d.burn_captions }))}
                >
                  <span>{t("up.captions")}</span>
                  <i>{onOff(renderDraft.burn_captions)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.tighten ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, tighten: !d.tighten }))}
                >
                  <span>{t("up.jumpCuts")}</span>
                  <i>{onOff(renderDraft.tighten)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.motion === "push" ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, motion: d.motion === "push" ? "none" : "push" }))}
                >
                  <span>{t("up.slowPush")}</span>
                  <i>{onOff(renderDraft.motion === "push")}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.denoise ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, denoise: !d.denoise }))}
                >
                  <span>{t("up.cleanVoice")}</span>
                  <i>{onOff(renderDraft.denoise)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.use_ocr ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, use_ocr: !d.use_ocr }))}
                >
                  <span>OCR</span>
                  <i>{onOff(renderDraft.use_ocr)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.use_vlm ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, use_vlm: !d.use_vlm }))}
                >
                  <span>{t("up.aiVision")}</span>
                  <i>{onOff(renderDraft.use_vlm)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.use_audio_events ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, use_audio_events: !d.use_audio_events }))}
                >
                  <span>{t("up.audioEvents")}</span>
                  <i>{onOff(renderDraft.use_audio_events)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.use_cues ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, use_cues: !d.use_cues }))}
                  title={t("cg.ownSoundsTitle")}
                >
                  <span>{t("cg.ownSounds")}</span>
                  <i>{onOff(renderDraft.use_cues)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.cue_learning ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, cue_learning: !d.cue_learning }))}
                >
                  <span>{t("up.cueLearning")}</span>
                  <i>{onOff(renderDraft.cue_learning)}</i>
                </button>
                <button
                  className={"toggle" + (renderDraft.auto_length ? " on" : "")}
                  onClick={() => setRenderDraft((d) => ({ ...d, auto_length: !d.auto_length }))}
                >
                  <span>{t("up.autoLength")}</span>
                  <i>{onOff(renderDraft.auto_length)}</i>
                </button>
              </div>
            </div>
            {project.content_type === "gameplay" && (
              <div className="field wide timing-controls">
                <label>{t("up.clipContext")}</label>
                <button
                  className={"toggle context-toggle" + (contextAuto ? " on" : "")}
                  onClick={() =>
                    setRenderDraft((d) =>
                      d.lead_seconds === null && d.tail_seconds === null
                        ? { ...d, lead_seconds: 16, tail_seconds: 20 }
                        : { ...d, lead_seconds: null, tail_seconds: null },
                    )
                  }
                >
                  <span>{t("up.autoContext")}</span>
                  <i>{onOff(contextAuto)}</i>
                </button>
                {(renderDraft.lead_seconds !== null || renderDraft.tail_seconds !== null) && (
                  <div className="context-ranges">
                    <div>
                      <div className="range-label">
                        <span>{t("up.beforeMoment")}</span>
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
                    </div>
                    <div>
                      <div className="range-label">
                        <span>{t("up.afterMoment")}</span>
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
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        </details>

        {/* Streamer.bot webhook: the mark-highlight URL to paste into
            Streamer.bot's HTTP request action. */}
        <details className="fold">
          <summary>
            <span>{t("cg.webhook")}</span>
            <span className="fold-summary">POST /mark-highlight</span>
          </summary>
          <div className="fold-body">
            <p className="muted tiny fold-intro">{t("cg.webhookIntro")}</p>
            <code className="code-line">
              POST /api/projects/{project.id}/mark-highlight?timestamp=SEC&amp;duration=30
            </code>
          </div>
        </details>
      </div>
    </div>
  );
}
