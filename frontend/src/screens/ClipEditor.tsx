import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { CaptionWord, Clip, Project, Rect, StyleTemplate } from "../lib/types";
import { fmtClock, fmtDuration } from "../lib/format";
import { mediaTimeUrl } from "../lib/media";
import { useT } from "../lib/i18n";
import { useUndo, type UndoState } from "../lib/useUndo";
import Toast, { type ToastMsg } from "../components/Toast";
import ScoreBadge from "../components/ScoreBadge";
import PublishPanel from "../components/PublishPanel";
import Waveform from "../components/Waveform";
import InteractiveCanvas from "../components/InteractiveCanvas";
import AudioWaveformHeatmap from "../components/AudioWaveformHeatmap";
import AiSuggestionsCoPilot, { type AiSuggestion } from "../components/AiSuggestionsCoPilot";
import PhoneSimulator from "../components/PhoneSimulator";
import MontageTimeline from "../components/MontageTimeline";
import StreamerWorkflowBar, { isInputIsolated } from "../components/StreamerWorkflowBar";
import type { PlatformType } from "../lib/collision";

type EditableCaptionWord = Pick<CaptionWord, "t" | "d" | "text" | "speaker">;

const editableCaptionWords = (c: Clip): EditableCaptionWord[] =>
  c.captions.words.map((w) => ({
    t: w.t,
    d: w.d,
    text: w.text,
    speaker: w.speaker ?? null,
  }));

const originalCropCx = (c: Clip): number | null =>
  c.reframe.cx_overridden ? c.reframe.keyframes[0]?.cx ?? 0.5 : null;

const cropChanged = (c: Clip, next: number | null): boolean => {
  const orig = originalCropCx(c);
  if (next === null) return orig !== null;
  return orig === null || Math.abs(next - orig) > 0.01;
};

// The settings column is split into tabs; labels live in i18n as `ce.tab.<id>`.
type EditorTab = "cut" | "captions" | "frame" | "ai" | "montage" | "publish";
const EDITOR_TABS: EditorTab[] = ["cut", "captions", "frame", "ai", "montage", "publish"];
const TAB_STORAGE_KEY = "clipforge.editorTab";

// The workflow bar brings its own loud chrome; in the editor it sits quietly.
const WORKFLOW_BAR_STYLE: CSSProperties = {
  background: "var(--panel)",
  border: "1px solid var(--border)",
  boxShadow: "none",
  borderRadius: 10,
};

export default function ClipEditor() {
  const { t } = useT();
  const { projectId, clipId } = useParams();
  const nav = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [styles, setStyles] = useState<StyleTemplate[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<ToastMsg | null>(null);
  // Montage: the order arranged in the sequencer (null = the project's order).
  const [montageOrder, setMontageOrder] = useState<string[] | null>(null);
  const [montageBusy, setMontageBusy] = useState(false);
  const rendered = (project?.clips ?? []).filter((c) => c.status === "ready" && c.export_url);
  const montageClipIds = (montageOrder ?? rendered.map((c) => c.id)).filter((id) =>
    rendered.some((c) => c.id === id));
  const latestMontage = project?.montages?.[0] ?? null;
  const buildMontage = async () => {
    if (!projectId || montageClipIds.length < 2 || montageBusy) return;
    setMontageBusy(true);
    try {
      await api.createMontage(projectId, montageClipIds);
      // The montage renders in the background: poll until it is done.
      for (let i = 0; i < 400; i++) {
        const p = await api.getProject(projectId);
        setProject(p);
        const m = p.montages?.[0];
        if (!m || m.status !== "rendering") {
          setMsg(m?.status === "ready"
            ? { text: t("ce.montageDone"), type: "info" }
            : { text: t("ce.montageFailed", { error: m?.error ?? t("ce.renderUnknown") }), type: "error" });
          break;
        }
        await new Promise((r) => setTimeout(r, 1500));
      }
    } catch (e: any) {
      setMsg({ text: e?.message ?? t("cg.errMontage"), type: "error" });
    } finally {
      setMontageBusy(false);
    }
  };
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [ver, setVer] = useState(0); // cache-buster for the rendered <video>
  const [previewMode, setPreviewMode] = useState<"rendered" | "original">("rendered");
  const alive = useRef(true); // stops the re-render poll after unmount

  // local editable state
  const [title, setTitle] = useState("");
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [styleId, setStyleId] = useState("");
  const [cx, setCx] = useState<number | null>(null);
  const [words, setWords] = useState<EditableCaptionWord[]>([]);
  const [fb, setFb] = useState<"up" | "down" | null>(null);
  const [layout, setLayout] = useState<string>("center");
  const [cam, setCam] = useState<Rect | null>(null);
  const [aspect, setAspect] = useState<string>(""); // "" = project default
  const [capSpeakers, setCapSpeakers] = useState<number[] | null>(null); // null = all
  const [loopPreviewSeconds, setLoopPreviewSeconds] = useState(0);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showSafeGuides, setShowSafeGuides] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [captionY, setCaptionY] = useState(0.78);
  const [activeSimulatorPlatform, setActiveSimulatorPlatform] = useState<PlatformType>("tiktok");
  const [showPhoneSimulator, setShowPhoneSimulator] = useState<boolean>(true);
  // The workflow bar's batch row (multi-clip actions) is hidden until asked for.
  const [showBatch, setShowBatch] = useState(false);
  // Active settings tab, remembered in this browser between clips.
  const [tab, setTab] = useState<EditorTab>(() => {
    try {
      const v = localStorage.getItem(TAB_STORAGE_KEY);
      if (v && (EDITOR_TABS as string[]).includes(v)) return v as EditorTab;
    } catch {
      /* storage blocked: start on the first tab */
    }
    return "cut";
  });
  const chooseTab = (next: EditorTab) => {
    setTab(next);
    try {
      localStorage.setItem(TAB_STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  };

  const activeStyle = useMemo(() => styles.find((s) => s.id === styleId), [styles, styleId]);

  useEffect(() => {
    if (activeStyle?.y_frac) {
      setCaptionY(activeStyle.y_frac);
    }
  }, [activeStyle]);

  // Undo/redo history.
  const undo = useUndo({
    title, start, end, styleId, cx, words, layout, cam, aspect, capSpeakers, loopPreviewSeconds,
  });
  // Patch the undo system into the editor state — it records changes and
  // lets us restore snapshots.
  const applyUndo = useCallback((snapshot: UndoState) => {
    setTitle(snapshot.title);
    setStart(snapshot.start);
    setEnd(snapshot.end);
    setStyleId(snapshot.styleId);
    setCx(snapshot.cx);
    setWords(snapshot.words);
    setLayout(snapshot.layout);
    setCam(snapshot.cam);
    setAspect(snapshot.aspect);
    setCapSpeakers(snapshot.capSpeakers);
    setLoopPreviewSeconds(snapshot.loopPreviewSeconds);
  }, []);

  // Refs for keyboard-driven controls
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoWrapperRef = useRef<HTMLDivElement | null>(null);

  // Keyboard shortcut ref — values are written by a useEffect below once all
  // variables are initialized, so the handler never goes stale.
  const keyboardRefs = useRef({
    start: 0, end: 0, srcDur: 0, dirty: false, busy: false,
    apply: async () => {}, canUndo: false, canRedo: false,
  });
  
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture when typing in an input, textarea, contenteditable, or caption word input.
      if (isInputIsolated(e.target)) return;

      const st = keyboardRefs.current;
      const video = videoRef.current;
      const step = e.shiftKey ? 0.1 : 0.5;
      const dur = st.srcDur;

      switch (e.code) {
        case "ArrowLeft":
          e.preventDefault();
          if (video) video.currentTime = Math.max(0, video.currentTime - step);
          break;
        case "ArrowRight":
          e.preventDefault();
          if (video) video.currentTime = Math.min(dur, video.currentTime + step);
          break;
        case "Enter":
          e.preventDefault();
          if (st.dirty && !st.busy) st.apply();
          break;
        case "KeyZ":
          if (e.metaKey || e.ctrlKey) {
            e.preventDefault();
            if (e.shiftKey) {
              if (st.canRedo) applyUndo(undo.redo());
            } else {
              if (st.canUndo) applyUndo(undo.undo());
            }
          }
          break;
        case "Slash":
          if (!e.shiftKey) {
            e.preventDefault();
            setShowShortcuts((s) => !s);
          }
          break;
        case "Escape":
          setShowShortcuts(false);
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Keep keyboard ref in sync with live variables (avoids TS hoisting errors).
  useEffect(() => {
    keyboardRefs.current = { start, end, srcDur, dirty, busy, apply, canUndo: undo.canUndo, canRedo: undo.canRedo };
  });

  // Record undo snapshots when editor state changes.
  const undoSnapshot = useRef<UndoState | null>(null);
  useEffect(() => {
    const snap = { title, start, end, styleId, cx, words, layout, cam, aspect, capSpeakers, loopPreviewSeconds };
    // Skip the initial hydration — only record user edits.
    if (clip && !undoSnapshot.current) {
      undoSnapshot.current = snap;
      return;
    }
    // Only push when something actually changed.
    if (clip && JSON.stringify(snap) !== JSON.stringify(undoSnapshot.current)) {
      undo.set(snap);
      undoSnapshot.current = snap;
    }
  }, [title, start, end, styleId, cx, words, layout, cam, aspect, capSpeakers, loopPreviewSeconds, clip]);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    api.styles().then(setStyles).catch(() => {});
  }, []);

  useEffect(() => {
    if (!projectId || !clipId) return;
    let live = true;
    api
      .getProject(projectId)
      .then((p) => {
        if (!live) return;
        setProject(p);
        const c = p.clips.find((x) => x.id === clipId) ?? null;
        if (c) hydrate(c);
        else setLoadErr(t("ce.notFound"));
      })
      .catch((e) => {
        if (live) setLoadErr(e?.message ?? t("ce.loadFail"));
      });
    return () => {
      live = false;
    };
  }, [projectId, clipId]);

  const hydrate = (c: Clip) => {
    setClip(c);
    setTitle(c.title);
    setStart(c.start);
    setEnd(c.end);
    setStyleId(c.captions.style_id);
    // cx_overridden (not overridden): layout/facecam edits also set
    // `overridden`, and showing their keyframe as a manual crop is wrong.
    setCx(originalCropCx(c));
    setWords(editableCaptionWords(c));
    setFb(c.feedback);
    setLayout(c.reframe.layout);
    setCam(c.reframe.facecam ?? null);
    setAspect(c.aspect ?? "");
    setCapSpeakers(c.caption_speakers ?? null);
    setLoopPreviewSeconds(c.loop_preview_seconds ?? 0);
    // Reset undo history so the first undo doesn't restore empty defaults.
    undo.reset({
      title: c.title,
      start: c.start,
      end: c.end,
      styleId: c.captions.style_id,
      cx: originalCropCx(c),
      words: editableCaptionWords(c),
      layout: c.reframe.layout,
      cam: c.reframe.facecam ?? null,
      aspect: c.aspect ?? "",
      capSpeakers: c.caption_speakers ?? null,
      loopPreviewSeconds: c.loop_preview_seconds ?? 0,
    });
  };

  // The set of speakers currently kept in captions (null on the clip = all).
  const keptSpeakers = (c: Clip | null, sel: number[] | null): number[] =>
    sel ?? c?.speakers ?? [];

  const toggleSpeaker = (sp: number) => {
    if (!clip) return;
    const kept = keptSpeakers(clip, capSpeakers);
    const next = kept.includes(sp) ? kept.filter((x) => x !== sp) : [...kept, sp].sort((a, b) => a - b);
    // A full set is the default ("all") - store it as null so re-renders show every speaker.
    setCapSpeakers(next.length === clip.speakers.length ? null : next);
  };

  const rate = async (r: "up" | "down") => {
    if (!projectId || !clipId) return;
    const next = fb === r ? "none" : r;
    setFb(next === "none" ? null : (next as "up" | "down"));
    await api.rateClip(projectId, clipId, next).catch(() => {});
  };

  const srcDur = project?.source?.duration ?? end;

  const dirty = useMemo(() => {
    if (!clip) return false;
    return (
      title !== clip.title ||
      Math.abs(start - clip.start) > 0.01 ||
      Math.abs(end - clip.end) > 0.01 ||
      styleId !== clip.captions.style_id ||
      cropChanged(clip, cx) ||
      layout !== clip.reframe.layout ||
      aspect !== (clip.aspect ?? "") ||
      JSON.stringify(capSpeakers) !== JSON.stringify(clip.caption_speakers ?? null) ||
      Math.abs(loopPreviewSeconds - (clip.loop_preview_seconds ?? 0)) > 0.01 ||
      (cam !== null && JSON.stringify(cam) !== JSON.stringify(clip.reframe.facecam)) ||
      JSON.stringify(words) !== JSON.stringify(editableCaptionWords(clip))
    );
  }, [clip, title, start, end, styleId, cx, words, layout, cam, aspect, capSpeakers, loopPreviewSeconds]);

  const spanChanged = clip && (Math.abs(start - clip.start) > 0.01 || Math.abs(end - clip.end) > 0.01);
  const captionSpeakersDirty =
    !!clip && JSON.stringify(capSpeakers) !== JSON.stringify(clip.caption_speakers ?? null);

  const apply = async () => {
    if (!projectId || !clipId || !clip) return;
    setBusy(true);
    setMsg({ text: t("ce.rendering"), type: "info" });
    try {
      const edit: Partial<{
        title: string;
        start: number;
        end: number;
        style_id: string;
        reframe_cx: number | null;
        caption_words: EditableCaptionWord[];
        caption_speakers: number[] | null;
        layout: string;
        facecam: { x: number; y: number; w: number; h: number };
        loop_preview_seconds: number;
        aspect: string;
      }> = {};
      if (title !== clip.title) edit.title = title;
      if (spanChanged) {
        edit.start = start;
        edit.end = end;
      }
      if (styleId !== clip.captions.style_id) edit.style_id = styleId;
      if (cropChanged(clip, cx)) edit.reframe_cx = cx;
      if (layout !== clip.reframe.layout) edit.layout = layout;
      if (aspect !== (clip.aspect ?? "")) edit.aspect = aspect;
      if (captionSpeakersDirty)
        edit.caption_speakers = capSpeakers;
      if (cam !== null && JSON.stringify(cam) !== JSON.stringify(clip.reframe.facecam))
        edit.facecam = cam;
      if (Math.abs(loopPreviewSeconds - (clip.loop_preview_seconds ?? 0)) > 0.01)
        edit.loop_preview_seconds = loopPreviewSeconds;
      // Only send manual caption edits if the span didn't change (a new span
      // re-derives captions from the transcript on the server).
      if (!spanChanged) {
        const orig = JSON.stringify(editableCaptionWords(clip));
        if (JSON.stringify(words) !== orig) edit.caption_words = words;
      }
      await api.editClip(projectId, clipId, edit);
      await pollUntilReady();
    } catch (e: any) {
      setMsg({ text: e.message ?? t("ce.editFail"), type: "error" });
      setBusy(false);
    }
  };

  const pollUntilReady = async () => {
    if (!projectId || !clipId) return;
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (!alive.current) return; // user navigated away - stop polling
      const p = await api.getProject(projectId);
      if (!alive.current) return;
      const c = p.clips.find((x) => x.id === clipId);
      if (c && c.status === "ready") {
        setProject(p);
        hydrate(c);
        setVer((v) => v + 1);
        setBusy(false);
        setMsg({ text: t("ce.updated"), type: "success", duration: 2000 });
        return;
      }
      if (c && c.status === "failed") {
        setBusy(false);
        setMsg({ text: t("ce.renderFailed", { error: c.error ?? t("ce.renderUnknown") }), type: "error" });
        return;
      }
    }
    setBusy(false);
    setMsg({ text: t("ce.stillRendering"), type: "info" });
  };

  if (loadErr)
    return (
      <div className="container">
        <div className="empty">
          <div className="col" style={{ alignItems: "center", gap: 12 }}>
            <span>{loadErr}</span>
            <Link className="btn ghost sm" to={projectId ? `/p/${projectId}` : "/"}>
              {t("ce.back")}
            </Link>
          </div>
        </div>
      </div>
    );

  if (!clip || !project)
    return (
      <div className="container">
        <div className="empty"><span className="spinner" /></div>
      </div>
    );

  const renderedKey = encodeURIComponent(`${project.updated_at}-${clip.status}-${ver}`);
  const renderedSrc = clip.export_url ? `${clip.export_url}?v=${renderedKey}` : undefined;
  const originalSrc = mediaTimeUrl(project.source?.path, start, end);
  const videoSrc = previewMode === "original" ? originalSrc : renderedSrc;
  const baseExportDuration = clip.tightened_duration ?? Math.max(end - start, 0);
  const loopPreviewEffective = loopPreviewSeconds > 0
    ? Math.min(loopPreviewSeconds, Math.max(baseExportDuration - 0.25, 0))
    : 0;
  const finalExportDuration = baseExportDuration + loopPreviewEffective;
  // Publishing needs a rendered clip; an unavailable remembered tab falls back.
  const tabs = EDITOR_TABS.filter((id) => id !== "publish" || clip.status === "ready");
  const activeTab: EditorTab = tabs.includes(tab) ? tab : "cut";
  const simulatorShown = Boolean(videoSrc) && showPhoneSimulator && previewMode === "rendered";

  return (
    <div className="container">
      <div className="editor-bar">
        <Link className="btn ghost sm" to={`/p/${projectId}`}>
          {t("ce.allClips")}
        </Link>
        <h2 title={title}>{title || clip.title}</h2>
        <div className="row">
          <button className="btn sm ghost" title={t("ce.likeTitle")}
            onClick={() => rate("up")}
            style={{ color: fb === "up" ? "var(--good)" : undefined }}>
            {t("ce.like")}
          </button>
          <button className="btn sm ghost" title={t("ce.dislikeTitle")}
            onClick={() => rate("down")}
            style={{ color: fb === "down" ? "var(--bad)" : undefined }}>
            {t("ce.dislike")}
          </button>
          {clip.export_url && (
            <a className="btn sm" href={api.downloadClipUrl(projectId!, clipId!)} download>
              {t("ce.download")}
            </a>
          )}
          {clip.captions.words.length > 0 && (
            <a className="btn sm ghost" href={api.downloadSrtUrl(projectId!, clipId!)} download
              title={t("ce.srtTitle")}>
              {t("ce.srt")}
            </a>
          )}
          <button
            className={"btn sm ghost" + (showBatch ? " on" : "")}
            aria-pressed={showBatch}
            onClick={() => setShowBatch((v) => !v)}
            title={t("ce.batchToggleTitle")}
          >
            {t("ce.batchToggle")}
          </button>
          <button className="btn sm ghost" disabled={!undo.canUndo}
            onClick={() => applyUndo(undo.undo())}
            title={t("ce.shortcutUndo")} style={{ fontWeight: undo.canUndo ? 600 : 400 }}>
            ↩
          </button>
          <button className="btn sm ghost" disabled={!undo.canRedo}
            onClick={() => applyUndo(undo.redo())}
            title={t("ce.shortcutRedo")} style={{ fontWeight: undo.canRedo ? 600 : 400 }}>
            ↪
          </button>
          <button className="btn primary sm" onClick={apply} disabled={!dirty || busy}>
            {busy ? <><span className="spinner" /> {t("ce.applyRendering")}</> : t("ce.apply")}
          </button>
        </div>
      </div>

      {/* Streamer Quick Workflow Bar: Hotkeys, Tactile/HUD Feedback, Batch Toolbar */}
      <div className={"editor-workflow" + (showBatch ? "" : " no-batch")}>
        <StreamerWorkflowBar
          style={WORKFLOW_BAR_STYLE}
          projectId={projectId}
          clips={project?.clips}
          activeClipId={clipId}
          currentTime={currentTime}
          duration={srcDur}
          start={start}
          end={end}
          currentRating={clip?.score ? Math.round(clip.score / 20) : (fb === "up" ? 5 : (fb === "down" ? 1 : null))}
          videoRef={videoRef}
          listenToKeyboard={true}
          onPlayToggle={(playing) => {
            if (videoRef.current) {
              if (playing && videoRef.current.paused) videoRef.current.play();
              else if (!playing && !videoRef.current.paused) videoRef.current.pause();
            }
          }}
          onShuttleRewind={(newT) => {
            if (newT !== undefined) setCurrentTime(newT);
          }}
          onShuttlePause={() => {
            if (videoRef.current) {
              videoRef.current.pause();
              videoRef.current.playbackRate = 1.0;
            }
          }}
          onShuttleForward={(newT) => {
            if (newT !== undefined) setCurrentTime(newT);
          }}
          onMarkIn={(t) => {
            const absT = previewMode === "rendered" ? start + t : t;
            const nextStart = Math.max(0, Math.min(absT, end - 1));
            setStart(nextStart);
            undo.set({ ...undo.state, start: nextStart });
          }}
          onMarkOut={(t) => {
            const absT = previewMode === "rendered" ? start + t : t;
            const nextEnd = Math.max(start + 1, Math.min(absT, srcDur));
            setEnd(nextEnd);
            undo.set({ ...undo.state, end: nextEnd });
          }}
          onRate={async (score) => {
            const nextFb = score >= 4 ? "up" : (score <= 2 ? "down" : null);
            setFb(nextFb);
            if (clip) {
              clip.score = score * 20;
              clip.feedback = nextFb;
            }
            if (projectId && clipId) {
              await api.rateClip(projectId, clipId, nextFb ?? "none").catch(() => {});
            }
            setMsg({ text: t("ce.rated", { score, pts: score * 20 }), type: "success", duration: 2000 });
          }}
          onBatchAction={async (action, clipIds, payload) => {
            if (!projectId) return;
            try {
              const res = await fetch(`/api/projects/${projectId}/clips/batch`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  clip_ids: clipIds,
                  action: action === "rate_up" ? "rate" : action,
                  payload: payload || {},
                }),
              });
              if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                throw new Error(errJson.detail || t("ce.batchFailed", { action }));
              }
              const data = await res.json();
              if (data.project) {
                setProject(data.project);
                const updated = data.project.clips.find((c: Clip) => c.id === clipId);
                if (updated) hydrate(updated);
              }
              setMsg({
                text: t("ce.batchApplied", { action, count: clipIds.length }),
                type: "success",
                duration: 3000,
              });
            } catch (e: any) {
              setMsg({ text: e.message || t("ce.batchFailed", { action }), type: "error" });
            }
          }}
        />
      </div>

      <div className="editor">
        <div className="preview-pane">
          <div className="preview-tools">
            <div className="seg preview-tabs">
              <button
                className={previewMode === "rendered" ? "on" : ""}
                onClick={() => setPreviewMode("rendered")}
              >
                {t("ce.tabRendered")}
              </button>
              <button
                className={previewMode === "original" ? "on" : ""}
                onClick={() => setPreviewMode("original")}
              >
                {t("ce.tabOriginal")}
              </button>
            </div>
            <div className="preview-toggles">
              <button
                className={"btn ghost sm" + (showPhoneSimulator ? " on" : "")}
                onClick={() => setShowPhoneSimulator(!showPhoneSimulator)}
                aria-pressed={showPhoneSimulator}
                data-testid="toggle-phone-simulator"
                title={t("ce.phoneFrameTitle")}
              >
                {t("ce.phoneFrame")}
              </button>
              <button
                className={"btn ghost sm" + (showSafeGuides ? " on" : "")}
                onClick={() => setShowSafeGuides(!showSafeGuides)}
                aria-pressed={showSafeGuides}
                title={t("ce.safeZoneTitle")}
              >
                {t("ce.safeZone")}
              </button>
            </div>
          </div>
          <div
            className={"video-wrap" + (simulatorShown ? " with-sim" : "")}
            ref={videoWrapperRef}
            style={{ position: "relative" }}
          >
            {videoSrc ? (
              simulatorShown ? (
                <PhoneSimulator
                  platform={activeSimulatorPlatform}
                  onPlatformChange={(p) => setActiveSimulatorPlatform(p)}
                  captionYFrac={captionY}
                  captionWords={words}
                  facecamRect={layout !== "center" ? cam : null}
                  currentTime={currentTime}
                  creatorHandle="@streamer"
                  clipTitle={title || clip.title}
                  onNudgeCaption={(_safeRect, safeYFrac) => {
                    setCaptionY(safeYFrac);
                    if (activeStyle) {
                      activeStyle.y_frac = safeYFrac;
                    }
                    undo.set({ ...undo.state });
                    setMsg({ text: t("ce.nudgedCaptions"), type: "success" });
                  }}
                  onNudgeFacecam={(safeCam) => {
                    setCam(safeCam);
                    undo.set({ ...undo.state, cam: safeCam });
                    setMsg({ text: t("ce.nudgedFacecam"), type: "success" });
                  }}
                >
                  <InteractiveCanvas
                    facecam={layout !== "center" ? cam : null}
                    onFacecamChange={(nextCam) => {
                      setCam(nextCam);
                      undo.set({ ...undo.state, cam: nextCam });
                    }}
                    showFacecam={layout !== "center" && !!cam}
                    cropCx={cx}
                    onCropCxChange={(nextCx) => {
                      setCx(nextCx);
                      undo.set({ ...undo.state, cx: nextCx });
                    }}
                    showCropOverlay={false}
                    captionWords={words}
                    onCaptionWordsChange={(nextWords) => {
                      setWords(nextWords);
                      undo.set({ ...undo.state, words: nextWords });
                    }}
                    captionYFrac={captionY}
                    onCaptionYFracChange={(nextY) => {
                      setCaptionY(nextY);
                      if (activeStyle) {
                        activeStyle.y_frac = nextY;
                      }
                    }}
                    showCaptions={true}
                    currentTime={currentTime}
                    activeStyle={activeStyle}
                    showSafeGuides={showSafeGuides}
                    onUndo={() => {
                      const prev = undo.undo();
                      if (prev) applyUndo(prev);
                    }}
                    onRedo={() => {
                      const next = undo.redo();
                      if (next) applyUndo(next);
                    }}
                    canUndo={undo.canUndo}
                    canRedo={undo.canRedo}
                  >
                    <video
                      key={videoSrc}
                      ref={videoRef}
                      src={videoSrc}
                      controls
                      playsInline
                      onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                      poster={clip.thumb_url ?? undefined}
                      style={{ width: "100%", height: "100%", display: "block" }}
                    />
                  </InteractiveCanvas>
                </PhoneSimulator>
              ) : (
                <InteractiveCanvas
                  facecam={layout !== "center" ? cam : null}
                  onFacecamChange={(nextCam) => {
                    setCam(nextCam);
                    undo.set({ ...undo.state, cam: nextCam });
                  }}
                  showFacecam={layout !== "center" && !!cam}
                  cropCx={cx}
                  onCropCxChange={(nextCx) => {
                    setCx(nextCx);
                    undo.set({ ...undo.state, cx: nextCx });
                  }}
                  showCropOverlay={previewMode === "original"}
                  captionWords={words}
                  onCaptionWordsChange={(nextWords) => {
                    setWords(nextWords);
                    undo.set({ ...undo.state, words: nextWords });
                  }}
                  captionYFrac={captionY}
                  onCaptionYFracChange={(nextY) => {
                    setCaptionY(nextY);
                    if (activeStyle) {
                      activeStyle.y_frac = nextY;
                    }
                  }}
                  showCaptions={true}
                  currentTime={currentTime}
                  activeStyle={activeStyle}
                  showSafeGuides={showSafeGuides}
                  onUndo={() => {
                    const prev = undo.undo();
                    if (prev) applyUndo(prev);
                  }}
                  onRedo={() => {
                    const next = undo.redo();
                    if (next) applyUndo(next);
                  }}
                  canUndo={undo.canUndo}
                  canRedo={undo.canRedo}
                >
                  <video
                    key={videoSrc}
                    ref={videoRef}
                    src={videoSrc}
                    controls
                    playsInline
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    poster={previewMode === "rendered" ? clip.thumb_url ?? undefined : undefined}
                    style={{ width: "100%", height: "100%", display: "block" }}
                  />
                </InteractiveCanvas>
              )
            ) : (
              <div className="empty">{t("ce.notRendered")}</div>
            )}
          </div>
          {/* Timeline under the viewer, so the phone preview fits on screen */}
          <div className="preview-wave">
            <AudioWaveformHeatmap
              projectId={projectId}
              clipId={clipId}
              start={start}
              end={end}
              currentTime={currentTime}
              duration={end - start}
              videoSrc={videoSrc}
              previewMode={previewMode}
              onSeek={(absT, relT) => {
                if (videoRef.current) {
                  if (previewMode === "rendered") {
                    videoRef.current.currentTime = relT;
                  } else {
                    videoRef.current.currentTime = absT;
                  }
                  setCurrentTime(videoRef.current.currentTime);
                }
              }}
            />
          </div>
        </div>

        <div className="controls-pane">
          {/* What the clip scored and why, above the settings tabs */}
          <div className="panel clip-summary">
            <ScoreBadge score={clip.score} />
            <div className="clip-summary-body">
              {clip.factors.length > 0 && (
                <div className="factors inline">
                  {clip.factors.map((f, i) => (
                    <span className="factor" key={i} title={f.detail}>
                      {f.label}
                      <span className="w">+{Math.round(f.weight)}</span>
                    </span>
                  ))}
                </div>
              )}
              {clip.hashtags?.length > 0 && (
                <div className="hashtag-row">
                  {clip.hashtags.map((h) => (
                    <span key={h} className="hashtag">{h}</span>
                  ))}
                </div>
              )}
            </div>
            <span className="pill">{clip.reframe.tracked ? t("ce.tracked") : clip.reframe.overridden ? t("ce.manualCrop") : t("ce.centerCrop")}</span>
          </div>

          <div className="editor-tabbar">
            <div className="editor-tabs" role="tablist" aria-label={t("ce.tabsLabel")}>
              {tabs.map((id) => (
                <button
                  key={id}
                  role="tab"
                  aria-selected={activeTab === id}
                  className={activeTab === id ? "on" : ""}
                  data-testid={id === "montage" ? "toggle-montage-timeline-btn" : undefined}
                  onClick={() => chooseTab(id)}
                >
                  {t(`ce.tab.${id}`)}
                </button>
              ))}
            </div>
            <button
              className="btn ghost sm shortcuts-btn"
              onClick={() => setShowShortcuts(true)}
              title={t("ce.shortcuts")}
              aria-label={t("ce.shortcuts")}
            >
              ?
            </button>
          </div>

          {/* AI Suggestions Co-Pilot Panel */}
          <div
            className="panel section"
            hidden={activeTab !== "ai"}
          >
            <AiSuggestionsCoPilot
              projectId={projectId}
              clipId={clipId}
              clip={clip}
              onApplySuggestion={(sug, updatedClip) => {
                const patch = sug.patch || {};
                let nextUndo = { ...undo.state };

                if (patch.start_t !== undefined || patch.start !== undefined) {
                  const newStart = Number(patch.start_t ?? patch.start);
                  setStart(newStart);
                  nextUndo = { ...nextUndo, start: newStart };
                }
                if (patch.end_t !== undefined || patch.end !== undefined) {
                  const newEnd = Number(patch.end_t ?? patch.end);
                  setEnd(newEnd);
                  nextUndo = { ...nextUndo, end: newEnd };
                }
                if (patch.layout) {
                  setLayout(patch.layout);
                  if (patch.layout !== "center" && !cam) {
                    setCam(project?.facecam ?? { x: 0.02, y: 0.55, w: 0.24, h: 0.34 });
                  }
                  nextUndo = { ...nextUndo, layout: patch.layout };
                }
                if (patch.reframe_cx !== undefined) {
                  const newCx = Number(patch.reframe_cx);
                  setCx(newCx);
                  nextUndo = { ...nextUndo, cx: newCx };
                }
                if (patch.teaser_loop !== undefined) {
                  const loopSec = patch.teaser_loop ? 2 : 0;
                  setLoopPreviewSeconds(loopSec);
                  nextUndo = { ...nextUndo, loopPreviewSeconds: loopSec };
                }
                undo.set(nextUndo);

                if (updatedClip) {
                  setClip((prev) => (prev ? { ...prev, ...updatedClip } : prev));
                }
                setMsg({ text: t("ce.applied", { title: sug.title }), type: "success", duration: 3000 });
              }}
            />
          </div>

          <div className="panel section" hidden={activeTab !== "cut"}>
            <h3>{t("ce.titleHook")}</h3>
            <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>

          <div className="panel section" hidden={activeTab !== "cut"}>
            <h3>{t("ce.cut")}</h3>
            <div className="muted tiny" style={{ marginBottom: 10 }}>
              {fmtClock(start)} → {fmtClock(end)} · {fmtDuration(end - start)}
            </div>
            <label className="tiny muted">{t("ce.start")}</label>
            <input type="range" min={0} max={srcDur} step={0.1} value={start}
              onChange={(e) => setStart(Math.min(Number(e.target.value), end - 1))} />
            <label className="tiny muted">{t("ce.end")}</label>
            <input type="range" min={0} max={srcDur} step={0.1} value={end}
              onChange={(e) => setEnd(Math.max(Number(e.target.value), start + 1))} />
            {spanChanged && (
              <div className="tiny muted" style={{ marginTop: 8 }}>
                {t("ce.spanNote")}
              </div>
            )}
          </div>

          <div className="panel section" hidden={activeTab !== "cut"}>
            <h3>{t("ce.loopPreview")}</h3>
            <div className="seg" style={{ marginBottom: 8 }}>
              {[0, 3, 4, 5].map((seconds) => (
                <button
                  key={seconds}
                  className={Math.abs(loopPreviewSeconds - seconds) < 0.01 ? "on" : ""}
                  onClick={() => setLoopPreviewSeconds(seconds)}
                  title={seconds === 0 ? t("ce.loopPreviewOffTitle") : t("ce.loopPreviewTitle", { seconds })}
                >
                  {seconds === 0 ? t("ce.loopPreviewNone") : t("ce.loopPreviewSeconds", { seconds })}
                </button>
              ))}
            </div>
            <span className="muted tiny">
              {loopPreviewEffective > 0
                ? t("ce.loopPreviewNote", {
                    seconds: loopPreviewEffective.toFixed(loopPreviewEffective % 1 ? 1 : 0),
                    duration: fmtDuration(finalExportDuration),
                  })
                : t("ce.loopPreviewOff")}
            </span>
          </div>

          <div className="panel section" hidden={activeTab !== "captions"}>
            <div className="panel-head">
              <h3>{t("ce.captionStyle")}</h3>
              <button className="btn sm ghost"
                onClick={async () => {
                  const name = prompt(t("ce.templatePrompt"), t("ce.templateDefault"));
                  if (!name) return;
                  try {
                    // Save current style as a new template via the typed api
                    // client (timeout + error-detail parsing). The raw fetch
                    // here used to swallow failures with console.error only,
                    // so a 400 (e.g. duplicate id) silently did nothing.
                    const style = styles.find(s => s.id === styleId);
                    if (!style) return;
                    const id = "custom_" + name.toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 30);
                    await api.createStyle({ ...style, id, name });
                    const updated = await api.styles();
                    setStyles(updated);
                    setStyleId(id);
                    setMsg({ text: t("ce.templateSaved"), type: "success", duration: 2000 });
                  } catch (e) {
                    setMsg({ text: (e as Error).message ?? t("ce.templateFail"), type: "error" });
                  }
                }}>
                {t("ce.saveAsTemplate")}
              </button>
            </div>
            <div className="style-picker">
              {styles.map((s) => (
                <div
                  key={s.id}
                  className={"style-chip" + (styleId === s.id ? " on" : "")}
                  onClick={() => setStyleId(s.id)}
                >
                  <div className="swatch" style={{ color: `#${s.highlight}` }}>
                    {s.uppercase ? "ABC" : "Abc"} <span style={{ color: `#${s.primary}` }}>123</span>
                  </div>
                  <div className="tiny muted">{s.name}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel section" hidden={activeTab !== "frame"}>
            <h3>{t("ce.reframe")}</h3>
            <div className="muted tiny" style={{ marginBottom: 10 }}>
              {cx === null
                ? t("ce.cropAuto")
                : t("ce.cropManual", { pct: Math.round(cx * 100) })}
              {cx !== null && Math.abs(cx - 0.5) < 0.005 && ` · ${t("ce.centerSnapped")}`}
            </div>
            {/* Crop presets; the crop window can also be dragged on the Original preview */}
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              <div className="seg">
                <button
                  className={cx !== null && Math.abs(cx - 0.25) < 0.03 ? "on" : ""}
                  onClick={() => {
                    const nextCx = 0.25;
                    setCx(nextCx);
                    undo.set({ ...undo.state, cx: nextCx });
                  }}
                  title={t("ce.cropLeftTitle")}
                >
                  {t("ce.cropLeft")}
                </button>
                <button
                  className={cx !== null && Math.abs(cx - 0.5) < 0.01 ? "on" : ""}
                  onClick={() => {
                    const nextCx = 0.5;
                    setCx(nextCx);
                    undo.set({ ...undo.state, cx: nextCx });
                  }}
                  title={t("ce.cropCenterTitle")}
                >
                  {t("ce.cropCenter")}
                </button>
                <button
                  className={cx !== null && Math.abs(cx - 0.75) < 0.03 ? "on" : ""}
                  onClick={() => {
                    const nextCx = 0.75;
                    setCx(nextCx);
                    undo.set({ ...undo.state, cx: nextCx });
                  }}
                  title={t("ce.cropRightTitle")}
                >
                  {t("ce.cropRight")}
                </button>
              </div>
              {cx !== null && (
                <button className="btn ghost sm" onClick={() => setCx(null)}>
                  {t("ce.resetAuto")}
                </button>
              )}
            </div>
            <div style={{ marginTop: 14 }}>
              <label className="tiny muted">{t("ce.aspectLabel")}</label>
              <select className="input" value={aspect}
                onChange={(e) => setAspect(e.target.value)}>
                <option value="">{t("ce.aspectProject", { aspect: project.settings.aspect })}</option>
                <option value="9:16">{t("ce.aspect916")}</option>
                <option value="4:5">{t("ce.aspect45")}</option>
                <option value="1:1">{t("ce.aspect11")}</option>
                <option value="16:9">{t("ce.aspect169")}</option>
              </select>
            </div>
          </div>

          {clip.kind === "gameplay" && (
            <div className="panel section" hidden={activeTab !== "frame"}>
              <h3>{t("ce.facecamLayout")}</h3>
              <div className="seg" style={{ marginBottom: 10 }}>
                {[
                  { id: "center", label: t("ce.layoutGameplay") },
                  { id: "split", label: t("ce.layoutSplit") },
                  { id: "framed", label: t("ce.layoutPip") },
                ].map((o) => (
                  <button
                    key={o.id}
                    className={layout === o.id ? "on" : ""}
                    onClick={() => {
                      setLayout(o.id);
                      if (o.id !== "center" && !cam)
                        setCam(project.facecam ?? { x: 0.02, y: 0.55, w: 0.24, h: 0.34 });
                    }}
                    title={
                      o.id === "split"
                        ? t("ce.layoutSplitTitle")
                        : o.id === "framed"
                          ? t("ce.layoutPipTitle")
                          : t("ce.layoutGameplayTitle")
                    }
                  >
                    {o.label}
                  </button>
                ))}
              </div>
              {layout !== "center" && cam && (
                <>
                  <div
                    style={{
                      position: "relative",
                      marginBottom: 10,
                      borderRadius: 8,
                      overflow: "hidden",
                      border: "1px solid var(--border)",
                    }}
                  >
                    <InteractiveCanvas
                      facecam={cam}
                      onFacecamChange={(nextCam) => {
                        setCam(nextCam);
                        undo.set({ ...undo.state, cam: nextCam });
                      }}
                      showFacecam={true}
                      showCaptions={false}
                      showCropOverlay={false}
                      canvasWidth={400}
                      canvasHeight={225}
                    >
                      <img
                        src={`/media/${projectId}/source.jpg`}
                        alt="source frame"
                        style={{ width: "100%", display: "block" }}
                        onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                      />
                    </InteractiveCanvas>
                  </div>
                  {/* Live Coordinates Readout & Quick Positioning Controls (Replacing raw sliders) */}
                  <div
                    className="row"
                    style={{
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 8,
                      fontSize: 11,
                    }}
                  >
                    <span className="pill" style={{ fontFamily: "monospace" }}>
                      X: {(cam.x * 100).toFixed(0)}% Y: {(cam.y * 100).toFixed(0)}%
                    </span>
                    <span className="pill" style={{ fontFamily: "monospace" }}>
                      W: {(cam.w * 100).toFixed(0)}% H: {(cam.h * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="row" style={{ gap: 6, flexWrap: "wrap", marginBottom: 8 }}>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        const next = { ...cam, x: 0.02, y: 0.04 };
                        setCam(next);
                        undo.set({ ...undo.state, cam: next });
                      }}
                      title={t("ce.camMoveTitle")}
                    >
                      {t("ce.camTopLeft")}
                    </button>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        const next = { ...cam, x: Math.max(0, 0.98 - cam.w), y: 0.04 };
                        setCam(next);
                        undo.set({ ...undo.state, cam: next });
                      }}
                      title={t("ce.camMoveTitle")}
                    >
                      {t("ce.camTopRight")}
                    </button>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        const next = { ...cam, x: 0.02, y: Math.max(0, 0.96 - cam.h) };
                        setCam(next);
                        undo.set({ ...undo.state, cam: next });
                      }}
                      title={t("ce.camMoveTitle")}
                    >
                      {t("ce.camBottomLeft")}
                    </button>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        const next = {
                          ...cam,
                          x: Math.max(0, 0.98 - cam.w),
                          y: Math.max(0, 0.96 - cam.h),
                        };
                        setCam(next);
                        undo.set({ ...undo.state, cam: next });
                      }}
                      title={t("ce.camMoveTitle")}
                    >
                      {t("ce.camBottomRight")}
                    </button>
                    <button
                      className="btn sm ghost"
                      onClick={() => {
                        const def = project.facecam ?? { x: 0.02, y: 0.55, w: 0.24, h: 0.34 };
                        setCam(def);
                        undo.set({ ...undo.state, cam: def });
                      }}
                      title={t("ce.camResetTitle")}
                    >
                      {t("ce.camReset")}
                    </button>
                  </div>
                  <span className="muted tiny">
                    {layout === "split" ? t("ce.camMarkSplit") : t("ce.camMarkPip")}
                  </span>
                </>
              )}
            </div>
          )}

          {clip.speakers.length > 1 && (
            <div className="panel section" hidden={activeTab !== "captions"}>
              <h3>{t("ce.speakers")} <span className="muted tiny">{t("ce.speakersHint")}</span></h3>
              <div className="seg" style={{ flexWrap: "wrap", marginBottom: 8 }}>
                {clip.speakers.map((sp) => {
                  const on = keptSpeakers(clip, capSpeakers).includes(sp);
                  return (
                    <button
                      key={sp}
                      className={on ? "on" : ""}
                      onClick={() => toggleSpeaker(sp)}
                      title={on ? t("ce.speakerOnTitle") : t("ce.speakerOffTitle")}
                    >
                      {t("ce.speakerState", { state: on ? t("up.on") : t("up.off"), index: sp + 1 })}
                    </button>
                  );
                })}
              </div>
              <span className="muted tiny">
                {captionSpeakersDirty ? t("ce.speakersDirty") : t("ce.speakersNote")}
              </span>
            </div>
          )}

          <div className="panel section" hidden={activeTab !== "captions"}>
            <h3>{t("ce.captions")} <span className="muted tiny">{t("ce.captionsHint")}</span></h3>
            {/* Active word tracker — highlights the word currently spoken */}
            <div className="caption-list" style={{ maxHeight: 300, overflowY: "auto" }}>
              {words.length === 0 && (
                <div className="muted tiny" style={{ padding: 12, textAlign: "center" }}>
                  {t("ce.noCaptions")}
                </div>
              )}
              {words.map((w, i) => {
                const isActive = videoRef.current
                  ? Math.abs(videoRef.current.currentTime - w.t) < w.d
                  : false;
                return (
                  <div
                    className={"caption-row" + (isActive ? " active" : "")}
                    key={i}
                    style={{
                      display: "flex", gap: 6, alignItems: "center",
                      padding: "4px 0",
                      background: isActive ? "var(--bg-hover)" : undefined,
                      borderRadius: 4,
                    }}
                  >
                    <span className="t muted" style={{
                      minWidth: 48, fontSize: 11, fontFamily: "monospace",
                      cursor: "pointer", userSelect: "none",
                    }}
                      onClick={() => {
                        if (videoRef.current) videoRef.current.currentTime = w.t;
                      }}
                      title={t("ce.seekToWord")}
                    >
                      {w.t.toFixed(1)}s
                    </span>
                    <input
                      className="input"
                      value={w.text}
                      style={{ flex: 1, fontSize: 13, padding: "4px 8px" }}
                      onChange={(e) => {
                        const next = [...words];
                        next[i] = { ...next[i], text: e.target.value };
                        setWords(next);
                      }}
                      onKeyDown={(e) => {
                        // Tab to next word, Shift+Tab to previous
                        if (e.key === "Tab") {
                          e.preventDefault();
                          const inputs = document.querySelectorAll(".caption-list input");
                          const idx = Array.from(inputs).indexOf(e.currentTarget);
                          const next = e.shiftKey
                            ? inputs[Math.max(0, idx - 1)]
                            : inputs[Math.min(inputs.length - 1, idx + 1)];
                          (next as HTMLInputElement)?.focus();
                        }
                      }}
                    />
                    <button
                      className="btn sm ghost"
                      style={{ padding: "2px 6px", fontSize: 12, opacity: 0.6 }}
                      onClick={() => {
                        const next = words.filter((_, idx) => idx !== i);
                        setWords(next);
                      }}
                      title={t("ce.deleteWord")}
                    >
                      ✕
                    </button>
                  </div>
                );
              })}
            </div>
            <div className="row" style={{ gap: 8, marginTop: 8 }}>
              <button className="btn sm ghost" style={{ fontSize: 12 }}
                onClick={() => {
                  const lastT = words.length > 0 ? words[words.length - 1].t + words[words.length - 1].d : 0;
                  setWords([...words, { t: lastT + 0.3, d: 0.3, text: "", speaker: null }]);
                }}>
                + {t("ce.addWord")}
              </button>
            </div>
          </div>

          {/* AI publish content — titles, description, hashtags per platform.
              Mounted while hidden so it loads as before. */}
          {clip.status === "ready" && projectId && clipId && (
            <div hidden={activeTab !== "publish"}>
              <PublishPanel projectId={projectId} clipId={clipId} />
            </div>
          )}

          {/* Montage sequencer: arrange the rendered clips, then build one video */}
          {activeTab === "montage" && (
            <div className="panel section montage-pane">
              <h3>{t("ce.montageTitle")} <span className="muted tiny">{t("ce.montageHint")}</span></h3>
              <MontageTimeline
                projectId={projectId}
                clips={project?.clips && project.clips.length > 0 ? project.clips : [clip]}
                onSequenceChange={(reordered) => {
                  setMontageOrder(reordered.map((r) => r.clipId));
                  setMsg({ text: t("ce.montageSeq", { count: reordered.length }), type: "info" });
                }}
                onTransitionsChange={() => {
                  setMsg({ text: t("ce.montageTransition"), type: "info" });
                }}
              />
              <div className="row" style={{ gap: 12, marginTop: 12, alignItems: "center", flexWrap: "wrap" }}>
                <button
                  className="btn primary"
                  data-testid="montage-build-btn"
                  disabled={montageBusy || montageClipIds.length < 2}
                  onClick={buildMontage}
                  title={t("ce.montageBuildTitle")}
                >
                  {montageBusy
                    ? <><span className="spinner" /> {t("cg.montageBuilding")}</>
                    : t("ce.montageBuild", { count: montageClipIds.length })}
                </button>
                {montageClipIds.length < 2 && (
                  <span className="muted tiny">{t("ce.montageNeedTwo")}</span>
                )}
                {latestMontage?.status === "ready" && latestMontage.export_url && (
                  <a className="btn ghost" href={api.downloadMontageUrl(projectId ?? "", latestMontage.id)} data-testid="montage-download">
                    {t("ce.montageDownload")}
                  </a>
                )}
                {latestMontage?.status === "failed" && (
                  <span className="notice error">
                    {t("ce.montageFailed", { error: latestMontage.error ?? t("ce.renderUnknown") })}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <Toast msg={msg} onDone={() => setMsg(null)} />

      {/* Keyboard shortcuts overlay */}
      {showShortcuts && (
        <div className="modal-overlay" onClick={() => setShowShortcuts(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}
               style={{ maxWidth: 420, padding: 24 }}>
            <h3 style={{ marginBottom: 16 }}>{t("ce.shortcuts")}</h3>
            <div className="shortcuts-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 24px" }}>
              {[
                ["Space", "ce.shortcutPlay"],
                ["J", "ce.shortcutRewind"],
                ["K", "ce.shortcutPause"],
                ["L", "ce.shortcutForward"],
                ["I", "ce.shortcutIn"],
                ["O", "ce.shortcutOut"],
                ["1-5", "ce.shortcutRating"],
                ["← →", "ce.shortcutStepBack"],
                ["⇧ ← →", "ce.shortcutFineBack"],
                ["⌘Z", "ce.shortcutUndo"],
                ["⌘⇧Z", "ce.shortcutRedo"],
                ["Enter", "ce.shortcutApply"],
                ["?", "ce.shortcuts"],
              ].map(([key, labelKey]) => (
                <div key={key} className="row" style={{ justifyContent: "space-between", gap: 12 }}>
                  <kbd style={{
                    background: "var(--bg)",
                    border: "1px solid var(--border)",
                    borderRadius: 4, padding: "2px 8px",
                    fontFamily: "monospace", fontSize: 13,
                    minWidth: 32, textAlign: "center",
                  }}>{key}</kbd>
                  <span className="muted" style={{ fontSize: 13 }}>{t(labelKey)}</span>
                </div>
              ))}
            </div>
            <button className="btn ghost sm" style={{ marginTop: 16 }}
                    onClick={() => setShowShortcuts(false)}>
              {t("diag.close")}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
