import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Clip, Project, Rect, StyleTemplate } from "../lib/types";
import { fmtClock, fmtDuration } from "../lib/format";
import ScoreBadge from "../components/ScoreBadge";

export default function ClipEditor() {
  const { projectId, clipId } = useParams();
  const nav = useNavigate();
  const [project, setProject] = useState<Project | null>(null);
  const [clip, setClip] = useState<Clip | null>(null);
  const [styles, setStyles] = useState<StyleTemplate[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
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
  const [words, setWords] = useState<{ t: number; d: number; text: string }[]>([]);
  const [fb, setFb] = useState<"up" | "down" | null>(null);
  const [layout, setLayout] = useState<string>("center");
  const [cam, setCam] = useState<Rect | null>(null);
  const [aspect, setAspect] = useState<string>(""); // "" = project default
  const [capSpeakers, setCapSpeakers] = useState<number[] | null>(null); // null = all

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
        else setLoadErr("This clip no longer exists.");
      })
      .catch((e) => {
        if (live) setLoadErr(e?.message ?? "Failed to load the clip.");
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
    setCx(c.reframe.cx_overridden ? c.reframe.keyframes[0]?.cx ?? 0.5 : null);
    setWords(c.captions.words.map((w) => ({ t: w.t, d: w.d, text: w.text })));
    setFb(c.feedback);
    setLayout(c.reframe.layout);
    setCam(c.reframe.facecam ?? null);
    setAspect(c.aspect ?? "");
    setCapSpeakers(c.caption_speakers ?? null);
  };

  // The set of speakers currently kept in captions (null on the clip = all).
  const keptSpeakers = (c: Clip | null, sel: number[] | null): number[] =>
    sel ?? c?.speakers ?? [];

  const toggleSpeaker = (sp: number) => {
    if (!clip) return;
    const kept = keptSpeakers(clip, capSpeakers);
    const next = kept.includes(sp) ? kept.filter((x) => x !== sp) : [...kept, sp].sort((a, b) => a - b);
    // A full set is the default ("all") — store it as null so re-renders show every speaker.
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
      (cx !== null && (!clip.reframe.cx_overridden || Math.abs(cx - (clip.reframe.keyframes[0]?.cx ?? 0.5)) > 0.01)) ||
      layout !== clip.reframe.layout ||
      aspect !== (clip.aspect ?? "") ||
      JSON.stringify(capSpeakers) !== JSON.stringify(clip.caption_speakers ?? null) ||
      (cam !== null && JSON.stringify(cam) !== JSON.stringify(clip.reframe.facecam)) ||
      JSON.stringify(words) !== JSON.stringify(clip.captions.words.map((w) => ({ t: w.t, d: w.d, text: w.text })))
    );
  }, [clip, title, start, end, styleId, cx, words, layout, cam, aspect, capSpeakers]);

  const spanChanged = clip && (Math.abs(start - clip.start) > 0.01 || Math.abs(end - clip.end) > 0.01);

  const apply = async () => {
    if (!projectId || !clipId || !clip) return;
    setBusy(true);
    setMsg("Re-rendering this clip…");
    try {
      const edit: any = {};
      if (title !== clip.title) edit.title = title;
      if (spanChanged) {
        edit.start = start;
        edit.end = end;
      }
      if (styleId !== clip.captions.style_id) edit.style_id = styleId;
      if (cx !== null) edit.reframe_cx = cx;
      if (layout !== clip.reframe.layout) edit.layout = layout;
      if (aspect !== (clip.aspect ?? "")) edit.aspect = aspect;
      if (JSON.stringify(capSpeakers) !== JSON.stringify(clip.caption_speakers ?? null))
        edit.caption_speakers = capSpeakers;
      if (cam !== null && JSON.stringify(cam) !== JSON.stringify(clip.reframe.facecam))
        edit.facecam = cam;
      // Only send manual caption edits if the span didn't change (a new span
      // re-derives captions from the transcript on the server).
      if (!spanChanged) {
        const orig = JSON.stringify(clip.captions.words.map((w) => ({ t: w.t, d: w.d, text: w.text })));
        if (JSON.stringify(words) !== orig) edit.caption_words = words;
      }
      await api.editClip(projectId, clipId, edit);
      await pollUntilReady();
    } catch (e: any) {
      setMsg(e.message ?? "Edit failed");
      setBusy(false);
    }
  };

  const pollUntilReady = async () => {
    if (!projectId || !clipId) return;
    for (let i = 0; i < 120; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (!alive.current) return; // user navigated away — stop polling
      const p = await api.getProject(projectId);
      if (!alive.current) return;
      const c = p.clips.find((x) => x.id === clipId);
      if (c && c.status === "ready") {
        setProject(p);
        hydrate(c);
        setVer((v) => v + 1);
        setBusy(false);
        setMsg("Updated ✓");
        setTimeout(() => setMsg(null), 2500);
        return;
      }
      if (c && c.status === "failed") {
        setBusy(false);
        setMsg("Render failed: " + (c.error ?? "unknown"));
        return;
      }
    }
    setBusy(false);
    setMsg("Still rendering — check back shortly.");
  };

  if (loadErr)
    return (
      <div className="container">
        <div className="empty">
          <div className="col" style={{ alignItems: "center", gap: 12 }}>
            <span>{loadErr}</span>
            <Link className="btn ghost sm" to={projectId ? `/p/${projectId}` : "/"}>
              ← Back to clips
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

  const renderedSrc = clip.export_url ? `${clip.export_url}?v=${ver}` : undefined;
  const originalSrc = project.source
    ? `/media/${project.source.path}#t=${start.toFixed(3)},${end.toFixed(3)}`
    : undefined;
  const videoSrc = previewMode === "original" ? originalSrc : renderedSrc;

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 18 }}>
        <Link className="btn ghost sm" to={`/p/${projectId}`}>
          ← All clips
        </Link>
        <div className="row">
          <button className="btn sm ghost" title="More like this (personalizes scoring)"
            onClick={() => rate("up")}
            style={{ color: fb === "up" ? "var(--good)" : undefined }}>
            👍
          </button>
          <button className="btn sm ghost" title="Less like this"
            onClick={() => rate("down")}
            style={{ color: fb === "down" ? "var(--bad)" : undefined }}>
            👎
          </button>
          {clip.export_url && (
            <a className="btn sm" href={api.downloadClipUrl(projectId!, clipId!)} download>
              ⬇ Download
            </a>
          )}
          {clip.captions.words.length > 0 && (
            <a className="btn sm ghost" href={api.downloadSrtUrl(projectId!, clipId!)} download
              title="Caption file for Premiere/Resolve">
              .srt
            </a>
          )}
          <button className="btn primary sm" onClick={apply} disabled={!dirty || busy}>
            {busy ? <><span className="spinner" /> Rendering…</> : "Apply & re-render"}
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
              Rendered
            </button>
            <button
              className={previewMode === "original" ? "on" : ""}
              onClick={() => setPreviewMode("original")}
            >
              Original
            </button>
          </div>
          <div className="video-wrap">
            {videoSrc ? (
              <video
                key={videoSrc}
                src={videoSrc}
                controls
                playsInline
                poster={previewMode === "rendered" ? clip.thumb_url ?? undefined : undefined}
              />
            ) : (
              <div className="empty">Not rendered yet</div>
            )}
          </div>
          <div className="panel section" style={{ marginTop: 14 }}>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
              <ScoreBadge score={clip.score} />
              <span className="pill">{clip.reframe.tracked ? "Speaker-tracked" : clip.reframe.overridden ? "Manual crop" : "Center crop"}</span>
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
          </div>
        </div>

        <div className="controls-pane">
          <div className="panel section">
            <h3>Title / hook</h3>
            <input className="input" value={title} onChange={(e) => setTitle(e.target.value)} />
          </div>

          <div className="panel section">
            <h3>Trim</h3>
            <div className="muted tiny" style={{ marginBottom: 10 }}>
              {fmtClock(start)} → {fmtClock(end)} · {fmtDuration(end - start)}
            </div>
            <label className="tiny muted">Start</label>
            <input type="range" min={0} max={srcDur} step={0.1} value={start}
              onChange={(e) => setStart(Math.min(Number(e.target.value), end - 1))} />
            <label className="tiny muted">End</label>
            <input type="range" min={0} max={srcDur} step={0.1} value={end}
              onChange={(e) => setEnd(Math.max(Number(e.target.value), start + 1))} />
            {spanChanged && (
              <div className="tiny muted" style={{ marginTop: 8 }}>
                Captions &amp; score will be recomputed for the new range.
              </div>
            )}
          </div>

          <div className="panel section">
            <h3>Caption style</h3>
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
            <h3>Reframe</h3>
            <div className="muted tiny" style={{ marginBottom: 10 }}>
              {cx === null
                ? "Auto: following the speaker. Drag to set a fixed crop."
                : `Manual crop center: ${Math.round(cx * 100)}% from left`}
            </div>
            <div className="range-row">
              <span className="tiny muted">◀</span>
              <input type="range" min={0} max={1} step={0.01}
                value={cx ?? 0.5}
                onChange={(e) => setCx(Number(e.target.value))} />
              <span className="tiny muted">▶</span>
            </div>
            {cx !== null && (
              <button className="btn ghost sm" style={{ marginTop: 10 }} onClick={() => setCx(null)}>
                Reset to auto (needs new range to re-track)
              </button>
            )}
            <div style={{ marginTop: 14 }}>
              <label className="tiny muted">Output aspect (this clip only)</label>
              <select className="input" value={aspect}
                onChange={(e) => setAspect(e.target.value)}>
                <option value="">Project default ({project.settings.aspect})</option>
                <option value="9:16">9:16 (Reels/Shorts/TikTok)</option>
                <option value="4:5">4:5 (Feed)</option>
                <option value="1:1">1:1 (Square)</option>
                <option value="16:9">16:9 (YouTube)</option>
              </select>
            </div>
          </div>

          {clip.kind === "gameplay" && (
            <div className="panel section">
              <h3>Facecam layout</h3>
              <div className="seg" style={{ marginBottom: 10 }}>
                {[
                  { id: "center", label: "Plain" },
                  { id: "split", label: "Stacked" },
                  { id: "framed", label: "PiP" },
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
                        ? "Streamer cam stacked above the gameplay"
                        : o.id === "framed"
                          ? "Streamer cam overlaid on the gameplay"
                          : "Gameplay only, no facecam"
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
                      ["x", "Left", 0, 0.95],
                      ["y", "Top", 0, 0.95],
                      ["w", "Width", 0.05, 0.6],
                      ["h", "Height", 0.05, 0.6],
                    ] as const
                  ).map(([k, label, min, max]) => (
                    <div className="range-row" key={k}>
                      <span className="tiny muted" style={{ width: 48 }}>{label}</span>
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
                    Mark the streamer cam — it'll be{" "}
                    {layout === "split" ? "stacked above the gameplay" : "overlaid on the gameplay"}.
                  </span>
                </>
              )}
            </div>
          )}

          {clip.speakers.length > 1 && (
            <div className="panel section">
              <h3>Speakers <span className="muted tiny">— toggle who appears in captions</span></h3>
              <div className="seg" style={{ flexWrap: "wrap", marginBottom: 8 }}>
                {clip.speakers.map((sp) => {
                  const on = keptSpeakers(clip, capSpeakers).includes(sp);
                  return (
                    <button
                      key={sp}
                      className={on ? "on" : ""}
                      onClick={() => toggleSpeaker(sp)}
                      title={on ? "Shown in captions — click to hide" : "Hidden from captions — click to show"}
                    >
                      {on ? "🟢" : "⚪"} Speaker {sp + 1}
                    </button>
                  );
                })}
              </div>
              <span className="muted tiny">
                Captions only show the speakers turned on — useful when one mic picks up
                cross-talk or you only want the host's lines.
              </span>
            </div>
          )}

          <div className="panel section">
            <h3>Captions <span className="muted tiny">— fix any transcription error</span></h3>
            <div className="caption-list">
              {words.map((w, i) => (
                <div className="caption-row" key={i}>
                  <span className="t">{w.t.toFixed(1)}s</span>
                  <input
                    className="input"
                    value={w.text}
                    onChange={(e) => {
                      const next = [...words];
                      next[i] = { ...next[i], text: e.target.value };
                      setWords(next);
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {msg && <div className={"toast" + (msg.includes("fail") ? " err" : "")}>{msg}</div>}
    </div>
  );
}
