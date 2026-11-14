import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { StatusPayload } from "../lib/types";
import { fmtDuration, fmtHMS, scoreColor } from "../lib/format";
import { useT } from "../lib/i18n";

/** One labelled usage bar (CPU, GPU, VRAM). */
function Meter({ label, pct, value }: { label: string; pct: number; value: string }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="meter">
      <span className="muted">{label}</span>
      <div className={"bar" + (clamped >= 90 ? " hot" : "")}>
        <i style={{ width: `${clamped}%` }} />
      </div>
      <b>{value}</b>
    </div>
  );
}

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
  // While paused, the stage it stopped in is marked "paused" instead of "active".
  const currentStage = activeStage ?? stages.find((s) => s.status === "paused");
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
  // Stage labels arrive from the backend in English; translate the known ones.
  const stageLabel = (name: string, fallback: string) => {
    const key = `proc.stage.${name}`;
    const label = t(key);
    return label === key ? fallback : label;
  };
  // Throughput: how much faster (or slower) than realtime we're processing.
  // A 10-min video done in 5 min = 2× realtime. Useful for users to gauge
  // whether to wait or walk away.
  const throughput =
    timing?.elapsed_seconds != null && timing?.source_duration != null && timing.elapsed_seconds > 2
      ? (timing.source_duration / timing.elapsed_seconds).toFixed(1)
      : null;
  const hasSystem =
    sys != null && (sys.cpu_pct != null || sys.gpu_pct != null || (sys.gpu_mem_mb != null && !!sys.gpu_mem_total_mb));

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
      <div className="page-head">
        <div>
          <h1>{paused ? t("proc.titlePaused") : t("proc.titleWorking")}</h1>
          <p>{paused ? t("proc.runningPaused") : t("proc.runningBg")}</p>
        </div>
        <div className="row">
          <button className={paused ? "btn primary sm" : "btn ghost sm"} onClick={togglePause} disabled={busy}>
            {busy ? <><span className="spinner" /> {t("proc.working")}</> : paused ? t("proc.resume") : t("proc.pause")}
          </button>
          <Link className="btn ghost sm" to="/">
            {t("proc.home")}
          </Link>
        </div>
      </div>
      {err && <div className="notice error" style={{ marginBottom: 16 }}>{err}</div>}

      <div className="proc-layout">
        <section className="card proc-main">
          {/* ETA + overall progress — the headline the user asked for. */}
          <div className="proc-eta-row">
            <div className="col" style={{ gap: 6, minWidth: 0 }}>
              <span className="eyebrow">
                {paused
                  ? `${t("proc.pausedPill")} · ${t("proc.stepShort", { index: (p.stage_index ?? 0) + 1, total: p.total_stages })}`
                  : activeStage
                    ? t("proc.step", {
                        index: (p.stage_index ?? 0) + 1,
                        total: p.total_stages,
                        stage: stageLabel(activeStage.name, activeStage.label),
                      })
                    : t("proc.preparing")}
              </span>
              <strong className="eta">
                {paused
                  ? currentStage
                    ? stageLabel(currentStage.name, currentStage.label)
                    : t("proc.titlePaused")
                  : etaSeconds != null
                    ? t(rendering ? "proc.renderEta" : "proc.eta", { time: fmtHMS(etaSeconds) })
                    : t("proc.etaCalc")}
              </strong>
            </div>
            <div className="proc-pct">
              {Math.round(p.pct)}
              <small>%</small>
            </div>
          </div>
          <div className={"bar lg" + (paused ? " paused" : "")}>
            <i style={{ width: `${p.pct}%` }} />
          </div>
          {p.message && <p className="proc-message muted tiny">{p.message}</p>}
          <div className="stages">
            {stages.map((s) => (
              <div key={s.name} className={"stage " + s.status}>
                <span className="ico">
                  {s.status === "done" ? "OK" : s.status === "paused" ? "II" : s.status === "active" ? "*" : "o"}
                </span>
                <span className="label">{stageLabel(s.name, s.label)}</span>
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
        </section>

        <aside className="proc-aside">
          <section className="card">
            <div className="card-head">
              <h3>{t("proc.run")}</h3>
              {paused && <span className="pill warn">{t("proc.pausedPill")}</span>}
            </div>
            <dl className="facts">
              <dt>{t("up.powerMode")}</dt>
              <dd>{powerLabel}</dd>
              <dt>{t("cg.outputFormat")}</dt>
              <dd>{status.settings?.aspect ?? "9:16"}</dd>
              {timing?.source_duration != null && (
                <>
                  <dt>{t("proc.factSource")}</dt>
                  <dd>{fmtHMS(timing.source_duration)}</dd>
                </>
              )}
              {timing?.elapsed_seconds != null && (
                <>
                  <dt>{t("proc.factElapsed")}</dt>
                  <dd>{fmtHMS(timing.elapsed_seconds)}</dd>
                </>
              )}
              {throughput != null && (
                <>
                  <dt>{t("proc.throughputLabel")}</dt>
                  <dd>{t("proc.throughput", { rate: throughput })}</dd>
                </>
              )}
              {status.target_clips != null && (
                <>
                  <dt>{t("proc.factClips")}</dt>
                  <dd>{status.rendered_count ?? renderedClips.length} / {status.target_clips}</dd>
                </>
              )}
            </dl>
          </section>

          {hasSystem && sys && (
            <section className="card">
              <div className="card-head">
                <h3>{t("proc.system")}</h3>
              </div>
              {sys.cpu_pct != null && (
                <Meter label="CPU" pct={sys.cpu_pct} value={`${Math.round(sys.cpu_pct)}%`} />
              )}
              {sys.gpu_pct != null && (
                <Meter label="GPU" pct={sys.gpu_pct} value={`${Math.round(sys.gpu_pct)}%`} />
              )}
              {sys.gpu_mem_mb != null && !!sys.gpu_mem_total_mb && (
                <Meter
                  label="VRAM"
                  pct={(sys.gpu_mem_mb / sys.gpu_mem_total_mb) * 100}
                  value={`${(sys.gpu_mem_mb / 1024).toFixed(1)}/${(sys.gpu_mem_total_mb / 1024).toFixed(1)} GB`}
                />
              )}
            </section>
          )}
        </aside>
      </div>

      {renderedClips.length > 0 ? (
        <div>
          <h3 className="section-title">
            {t("proc.renderedDuring", { count: renderedClips.length })}
          </h3>
          <div className="clip-grid compact">
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
      ) : (
        <p className="grid-hint muted tiny">{t("proc.previewsHint")}</p>
      )}
    </div>
  );
}
