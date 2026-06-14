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
  const [montaging, setMontaging] = useState(false);
  const [montageErr, setMontageErr] = useState<string | null>(null);
  const [showCues, setShowCues] = useState(false);
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

  const toggleSelect = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

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

  const refresh = async () => onChange(await api.getProject(project.id));

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
            {project.clips.length} clips · {ready} rendered · {project.settings.aspect}
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
