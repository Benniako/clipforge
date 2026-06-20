import { Link } from "react-router-dom";
import type { StatusPayload } from "../lib/types";
import { fmtDuration, scoreColor } from "../lib/format";

export default function ProcessingView({
  status,
  projectId,
}: {
  status: StatusPayload;
  projectId: string;
}) {
  const p = status.progress;
  const stages = p.stages ?? [];
  const renderedClips = status.clips.filter((c) => c.thumb_url);
  const sys = status.system;
  const powerLabel =
    status.settings?.power_mode === "max_gpu"
      ? "Max GPU"
      : status.settings?.power_mode === "quality"
        ? "Qualitaet"
        : "Ausgewogen";

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Deine Clips werden erstellt...</h2>
        <Link className="btn ghost sm" to="/">
          Zurueck zur Startseite
        </Link>
      </div>
      <p className="muted">
        Das laeuft im Hintergrund weiter - du kannst diesen Tab sicher schliessen. {p.message}
      </p>
      <div className="row" style={{ marginTop: 10, flexWrap: "wrap" }}>
        <span className="pill">{powerLabel}</span>
        <span className="pill">{status.settings?.aspect ?? "9:16"}</span>
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
        <span className="muted tiny">Vorschauen erscheinen unten, sobald einzelne Clips fertig sind.</span>
      </div>

      <div className="panel" style={{ padding: 22, marginTop: 16 }}>
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
                {s.status === "done" ? "✓" : s.status === "active" ? "•" : "○"}
              </span>
              <span className="label">{s.label}</span>
              <div className="spacer" style={{ flex: 1 }} />
              {s.status === "active" && (
                <span className="tiny muted">{Math.round(s.pct * 100)}%</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {renderedClips.length > 0 && (
        <div style={{ marginTop: 28 }}>
          <h3 style={{ marginBottom: 12 }}>
            Fertige Clips waehrend des Renderns ({renderedClips.length})
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
