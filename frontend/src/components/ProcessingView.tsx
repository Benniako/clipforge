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

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Making your clips…</h2>
        <Link className="btn ghost sm" to="/">
          ← Leave & come back later
        </Link>
      </div>
      <p className="muted">
        This runs in the background — it’s safe to close this tab. {p.message}
      </p>

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
            Clips are finishing as they render ({renderedClips.length})
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
