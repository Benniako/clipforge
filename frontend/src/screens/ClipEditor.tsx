import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

export default function ClipEditor() {
  const { t } = useT();
  const { projectId, clipId } = useParams();
  const nav = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [styles, setStyles] = useState<StyleTemplate[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<ToastMsg | null>(null);
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
      // Don't capture when typing in an input or textarea.
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const st = keyboardRefs.current;
      const video = videoRef.current;
      const step = e.shiftKey ? 0.1 : 0.5;
      const dur = st.srcDur;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          if (video) {
            if (video.paused) video.play();
            else video.pause();
          }
          break;
        case "KeyJ":
          e.preventDefault();
          if (video) {
            video.playbackRate = Math.max(0.25, video.playbackRate - 0.5);
            if (video.paused) video.play();
          }
          break;
        case "KeyK":
          e.preventDefault();
          if (video) { video.pause(); video.playbackRate = 1; }
          break;
        case "KeyL":
          e.preventDefault();
          if (video) {
            video.playbackRate = Math.min(4, video.playbackRate + 0.5);
            if (video.paused) video.play();
          }
          break;
        case "KeyI":
          e.preventDefault();
          if (video) setStart(Math.max(0, Math.min(video.currentTime, st.end - 1)));
          break;
        case "KeyO":
          e.preventDefault();
          if (video) setEnd(Math.max(st.start + 1, Math.min(video.currentTime, dur)));
          break;
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

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 18 }}>
        <Link className="btn ghost sm" to={`/p/${projectId}`}>
          {t("ce.allClips")}
        </Link>
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

      <div className="editor">
        <div className="preview-pane">
          <div className="seg preview-tabs" style={{ marginBottom: 10 }}>
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
          <Waveform src={videoSrc || ""} start={start} end={end} onSeek={(t) => { if (videoRef.current) videoRef.current.currentTime = t; }} />
          <div className="video-wrap" ref={videoWrapperRef}>
            {videoSrc ? (
              <video
                key={videoSrc}
                ref={videoRef}
                src={videoSrc}
                controls
                playsInline
                poster={previewMode === "rendered" ? clip.thumb_url ?? undefined : undefined}
              />
            ) : (
              <div className="empty">{t("ce.notRendered")}</div>
            )}
          </div>
          <div className="panel section" style={{ marginTop: 14 }}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
              <ScoreBadge score={clip.score} />
              <span className="pill">{clip.reframe.tracked ? t("ce.tracked") : clip.reframe.overridden ? t("ce.manualCrop") : t("ce.centerCrop")}</span>
            </div>
            <div className="factors">
              {clip.factors.map((f, i) => (
                <span className="factor" key={i} title={f.detail}>
                  {f.label} <span className="muted tiny">(+{f.weight})</span>
                </span>
              ))}
            </div>
            {clip.hashtags?.length > 0 && (
              <div className="row" style={{ flexWrap: "wrap", gap: 6, marginTop: 12 }}>
                {clip.hashtags.map((h) => (
                  <span key={h} className="pill" style={{ color: "var(--accent)" }}>
                    {h}
                  </span>
                ))}
              </div>
            )}
            <div className="shortcuts-hint muted tiny" style={{ marginTop: 10, cursor: "pointer" }}
                 onClick={() => setShowShortcuts(true)}>
              ⌨️ <span className="muted">{t("ce.shortcuts")}</span>
            </div>
          </div>
        </div>

        <div className="controls-pane">
          <div className="panel section">
            <h3>{t("ce.titleHook")}</h3>
            <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>

          <div className="panel section">
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

          <div className="panel section">
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

          <div className="panel section">
            <h3>{t("ce.captionStyle")}</h3>
            <div className="row" style={{ gap: 8, marginBottom: 10 }}>
              <button className="btn sm ghost" style={{ fontSize: 11 }}
                onClick={async () => {
                  const name = prompt("Template name:", "My Template");
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
                    setMsg({ text: (e as Error).message ?? "Failed to save template", type: "error" });
                  }
                }}>
                💾 {t("ce.saveAsTemplate")}
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

          <div className="panel section">
            <h3>{t("ce.reframe")}</h3>
            <div className="muted tiny" style={{ marginBottom: 10 }}>
              {cx === null
                ? t("ce.cropAuto")
                : t("ce.cropManual", { pct: Math.round(cx * 100) })}
            </div>
            <div className="range-row">
              <span className="tiny muted">{t("ce.left")}</span>
              <input type="range" min={0} max={1} step={0.01}
                value={cx ?? 0.5}
                onChange={(e) => setCx(Number(e.target.value))} />
              <span className="tiny muted">{t("ce.right")}</span>
            </div>
            {cx !== null && (
              <button className="btn ghost sm" style={{ marginTop: 10 }} onClick={() => setCx(null)}>
                {t("ce.resetAuto")}
              </button>
            )}
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
            <div className="panel section">
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
                  <div style={{ position: "relative", marginBottom: 10 }}>
                    <img
                      src={`/media/${projectId}/source.jpg`}
                      alt="source frame"
                      style={{ width: "100%", borderRadius: 8, display: "block" }}
                      onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
                    />
                    <div
                      style={{
                        position: "absolute",
                        border: "2px solid var(--accent)",
                        borderRadius: 4,
                        pointerEvents: "none",
                        left: `${cam.x * 100}%`,
                        top: `${cam.y * 100}%`,
                        width: `${cam.w * 100}%`,
                        height: `${cam.h * 100}%`,
                      }}
                    />
                  </div>
                  {(
                    [
                      ["x", "ce.camX", 0, 0.95],
                      ["y", "ce.camY", 0, 0.95],
                      ["w", "ce.camW", 0.05, 0.6],
                      ["h", "ce.camH", 0.05, 0.6],
                    ] as const
                  ).map(([k, labelKey, min, max]) => (
                    <div className="range-row" key={k}>
                      <span className="tiny muted" style={{ width: 48 }}>{t(labelKey)}</span>
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={0.005}
                        value={cam[k]}
                        onChange={(e) => setCam({ ...cam, [k]: Number(e.target.value) })}
                      />
                    </div>
                  ))}
                  <span className="muted tiny">
                    {layout === "split" ? t("ce.camMarkSplit") : t("ce.camMarkPip")}
                  </span>
                </>
              )}
            </div>
          )}

          {clip.speakers.length > 1 && (
            <div className="panel section">
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

          <div className="panel section">
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

          {/* AI publish content — titles, description, hashtags per platform. */}
          {clip.status === "ready" && projectId && clipId && (
            <PublishPanel projectId={projectId} clipId={clipId} />
          )}
        </div>
      </div>

      <Toast msg={msg} onDone={() => setMsg(null)} />

      {/* Keyboard shortcuts overlay */}
      {showShortcuts && (
        <div className="modal-overlay" onClick={() => setShowShortcuts(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}
               style={{ maxWidth: 420, padding: 24 }}>
            <h3 style={{ marginBottom: 16 }}>⌨️ {t("ce.shortcuts")}</h3>
            <div className="shortcuts-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 24px" }}>
              {[
                ["Space", "ce.shortcutPlay"],
                ["J", "ce.shortcutRewind"],
                ["K", "ce.shortcutPause"],
                ["L", "ce.shortcutForward"],
                ["I", "ce.shortcutIn"],
                ["O", "ce.shortcutOut"],
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
