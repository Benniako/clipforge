import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { StatusPayload } from "../lib/types";
import { fmtDuration, fmtHMS, scoreColor } from "../lib/format";
import { useT } from "../lib/i18n";

export default function ProcessingView({
  status,
  projectId,
  onStatus,
}: {
  status: StatusPayload;
  projectId: string;
  onStatus: (status: StatusPayload) => void;
}) {
  const { t } = useT();
  const p = status.progress;
  const stages = p.stages ?? [];
  const renderedClips = status.clips.filter((c) => c.thumb_url);
  const sys = status.system;
  const timing = status.timing;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const paused = status.status === "paused";
  const powerLabel =
    status.settings?.power_mode === "max_gpu"
      ? t("proc.powerMaxGpu")
      : status.settings?.power_mode === "quality"
        ? t("proc.powerQuality")
        : t("proc.powerBalanced");
  const activeStage = stages.find((s) => s.status === "active");
  // Compute render ETA locally from clip throughput: (remaining / done) × elapsed.
  // More accurate than the overall pipeline ETA during rendering because it only
  // counts time actually spent encoding, not transcription/detection/reframing.
  const rendering = activeStage?.name === "render";
  let renderEta: number | null = null;
  if (rendering && timing?.elapsed_seconds != null && timing.elapsed_seconds > 3) {
    const done = status.rendered_count ?? 0;
    const total = status.clips?.length ?? status.target_clips ?? 0;
    if (done > 0 && total > done) {
      const perClip = timing.elapsed_seconds / done;
      renderEta = Math.round(perClip * (total - done));
    }
  }
  const etaSeconds = renderEta ?? timing?.eta_seconds ?? null;

  const togglePause = async () => {
    setBusy(true);
    setErr(null);
    try {
      const next = paused ? await api.resumeProject(projectId) : await api.pauseProject(projectId);
      onStatus(next);
    } catch (e: any) {
      setErr(e?.message ?? t("proc.pauseError"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <h2>{paused ? t("proc.titlePaused") : t("proc.titleWorking")}</h2>
        <div className="row">
          <button className={paused ? "btn primary sm" : "btn ghost sm"} onClick={togglePause} disabled={busy}>
            {busy ? <><span className="spinner" /> {t("proc.working")}</> : paused ? t("proc.resume") : t("proc.pause")}
          </button>
          <Link className="btn ghost sm" to="/">
            {t("proc.home")}
          </Link>
        </div>
      </div>
      <p className="muted">
        {paused ? t("proc.runningPaused") : t("proc.runningBg")}{" "}
        {p.message}
      </p>
      {err && <p className="tiny" style={{ color: "var(--bad)" }}>{err}</p>}
      <div className="row" style={{ marginTop: 10, flexWrap: "wrap" }}>
        <span className="pill">{powerLabel}</span>
        <span className="pill">{status.settings?.aspect ?? "9:16"}</span>
        {paused && <span className="pill" style={{ color: "var(--warn)" }}>{t("proc.pausedPill")}</span>}
        {sys?.cpu_pct !== null && sys?.cpu_pct !== undefined && (
          <span className="pill">CPU {Math.round(sys.cpu_pct)}%</span>
        )}
        {sys?.gpu_pct !== null && sys?.gpu_pct !== undefined && (
          <span className="pill">GPU {Math.round(sys.gpu_pct)}%</span>
        )}
        {sys?.gpu_mem_mb !== null && sys?.gpu_mem_mb !== undefined && sys?.gpu_mem_total_mb ? (
          <span className="pill">
            VRAM {(sys.gpu_mem_mb / 1024).toFixed(1)}/{(sys.gpu_mem_total_mb / 1024).toFixed(1)} GB
          </span>
        ) : null}
        <span className="muted tiny">{t("proc.previewsHint")}</span>
      </div>

      <div className="panel" style={{ padding: 22, marginTop: 16 }}>
        {/* ETA + run info — the headline the user asked for. */}
        <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
          <strong style={{ fontSize: 18 }}>
            {paused
              ? t("proc.titlePaused")
              : etaSeconds != null
                ? t(rendering ? "proc.renderEta" : "proc.eta", { time: fmtHMS(etaSeconds) })
                : t("proc.etaCalc")}
          </strong>
          <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
            {activeStage && (
              <span className="pill">
                {t("proc.step", { index: (p.stage_index ?? 0) + 1, total: p.total_stages, stage: activeStage.label })}
              </span>
            )}
            {timing?.elapsed_seconds != null && (
              <span className="pill">{t("proc.elapsed", { time: fmtHMS(timing.elapsed_seconds) })}</span>
            )}
            {timing?.source_duration != null && (
              <span className="pill">{t("proc.sourceLen", { time: fmtHMS(timing.source_duration) })}</span>
            )}
            {/* Throughput: how much faster (or slower) than realtime we're
                processing. A 10-min video done in 5 min = 2× realtime. Useful
                for users to gauge whether to wait or walk away. */}
            {timing?.elapsed_seconds != null && timing?.source_duration != null
              && timing.elapsed_seconds > 2 && (
              <span className="pill" title={t("proc.throughputLabel")}>
                {t("proc.throughput", {
                  rate: (timing.source_duration / timing.elapsed_seconds).toFixed(1),
                })}
              </span>
            )}
            {status.target_clips != null && (
              <span className="pill">
                {t("proc.clipsProgress", { done: status.rendered_count ?? renderedClips.length, target: status.target_clips })}
              </span>
            )}
          </div>
        </div>
        <div className="bar" style={{ marginBottom: 6 }}>
          <i style={{ width: `${p.pct}%` }} />
        </div>
        <div className="tiny muted" style={{ textAlign: "right" }}>
          {Math.round(p.pct)}%
        </div>
        <div className="stages">
          {stages.map((s) => (
            <div key={s.name} className={"stage " + s.status}>
              <span className="ico">
                {s.status === "done" ? "OK" : s.status === "paused" ? "II" : s.status === "active" ? "*" : "o"}
              </span>
              <span className="label">{s.label}</span>
              <div className="spacer" style={{ flex: 1 }} />
              {s.status === "active" && (
                <span className="tiny muted" style={{ marginRight: 8 }}>
                  {Math.round(s.pct * 100)}%
                </span>
              )}
              {/* Per-stage elapsed: how long the active stage has run, or how
                  long a completed stage took. The anchor for repeat-run
                  expectations ("transcribe always takes ~40s on my machine"). */}
              {s.status === "active" && s.elapsed_seconds != null && (
                <span className="tiny muted">
                  {t("proc.stageElapsed", { time: fmtHMS(s.elapsed_seconds) })}
                </span>
              )}
              {s.status === "done" && s.elapsed_seconds != null && (
                <span className="tiny muted">
                  {t("proc.stageDone", { time: fmtHMS(s.elapsed_seconds) })}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {renderedClips.length > 0 && (
        <div style={{ marginTop: 28 }}>
          <h3 style={{ marginBottom: 12 }}>
            {t("proc.renderedDuring", { count: renderedClips.length })}
          </h3>
          <div className="clip-grid">
            {renderedClips.map((c) => (
              <Link
                key={c.id}
                to={`/p/${projectId}/clip/${c.id}`}
                className="clip-card"
              >
                <div
                  className="thumb"
                  style={c.thumb_url ? { backgroundImage: `url(${c.thumb_url})` } : undefined}
                >
                  <span className="dur">{fmtDuration(c.duration)}</span>
                </div>
                <div className="clip-body">
                  <span
                    className="score-badge"
                    style={{ ["--c" as string]: scoreColor(c.score) }}
                  >
                    <span className="ring" style={{ ["--p" as string]: c.score }}>
                      <i>{c.score}</i>
                    </span>
                  </span>
                  <div className="clip-title">{c.title}</div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
