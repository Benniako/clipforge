import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Goal, GoalMatch, Project } from "../lib/types";
import { fmtClock } from "../lib/format";

interface Props {
  project: Project;
  /** Called with the fresh project after a clip was requested or finished. */
  onChange: (p: Project) => void;
}

/**
 * Every goal of a session, grouped by match from first to last, so the user
 * chooses which goals become clips instead of taking the top-ranked few.
 */
export default function GoalsByMatch({ project, onChange }: Props) {
  const [matches, setMatches] = useState<GoalMatch[] | null>(null);
  const [open, setOpen] = useState<Record<number, boolean>>({});
  const [busy, setBusy] = useState<Record<number, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const load = useCallback(async () => {
    try {
      const res = await api.listGoals(project.id);
      if (alive.current) setMatches(res.matches);
    } catch (e: any) {
      if (alive.current) setError(e?.message ?? "Tore konnten nicht geladen werden.");
    }
  }, [project.id]);

  useEffect(() => {
    alive.current = true;
    load();
    return () => {
      alive.current = false;
    };
    // Reload when clips change: a goal's clip may have been made or finished.
  }, [load, project.clips.length, project.updated_at]);

  // While a goal clip renders, keep the project (and so this list) fresh.
  const rendering = (matches ?? []).some((m) => m.goals.some((g) => g.clip_status === "rendering" || g.clip_status === "pending"));
  useEffect(() => {
    if (!rendering) return;
    const id = setInterval(async () => {
      try {
        const p = await api.getProject(project.id);
        if (alive.current) onChange(p);
      } catch {
        /* next tick retries */
      }
    }, 2500);
    return () => clearInterval(id);
  }, [rendering, project.id, onChange]);

  if (!matches || matches.length === 0) return null;
  const total = matches.reduce((n, m) => n + m.goals.length, 0);

  const makeClips = async (goals: Goal[]) => {
    const todo = goals.filter((g) => !g.clip_id);
    if (todo.length === 0) return;
    setError(null);
    setBusy((b) => ({ ...b, ...Object.fromEntries(todo.map((g) => [g.t, true])) }));
    try {
      for (const g of todo) await api.clipGoal(project.id, g.t);
      const p = await api.getProject(project.id);
      if (alive.current) onChange(p);
      await load();
    } catch (e: any) {
      if (alive.current) setError(e?.message ?? "Clip konnte nicht erstellt werden.");
    } finally {
      if (alive.current) setBusy((b) => ({ ...b, ...Object.fromEntries(todo.map((g) => [g.t, false])) }));
    }
  };

  const who = (g: Goal) => (g.streamer === true ? "Dein Tor" : g.streamer === false ? "Gegentor" : "Tor");

  return (
    <div className="panel section goals-by-match" data-testid="goals-by-match">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <div>
          <span className="eyebrow">Tore nach Spiel</span>
          <div className="muted">
            {total} Tore in {matches.length} {matches.length === 1 ? "Spiel" : "Spielen"} - wähle, welche zu Clips werden.
          </div>
        </div>
      </div>
      {error && <div className="notice error">{error}</div>}
      {matches.map((m) => {
        const isOpen = open[m.index] ?? matches.length <= 3;
        const without = m.goals.filter((g) => !g.clip_id);
        return (
          <div key={m.index} className="goal-match" data-testid={`goal-match-${m.index}`}>
            <div className="row goal-match-head" style={{ gap: 12, alignItems: "center" }}>
              <button
                className="btn ghost sm"
                aria-expanded={isOpen}
                onClick={() => setOpen((o) => ({ ...o, [m.index]: !isOpen }))}
              >
                {isOpen ? "▾" : "▸"} Spiel {m.index}
              </button>
              <strong>Endstand {m.final_score}</strong>
              <span className="muted">
                {m.goals.length} {m.goals.length === 1 ? "Tor" : "Tore"} · ab {fmtClock(m.start)}
              </span>
              <div className="spacer" />
              {without.length > 0 && (
                <button className="btn ghost sm" onClick={() => makeClips(m.goals)}>
                  Alle {without.length} als Clips
                </button>
              )}
            </div>
            {isOpen && (
              <ul className="goal-list">
                {m.goals.map((g) => (
                  <li key={g.t} className={"goal-row" + (g.streamer === true ? " mine" : "")}>
                    <span className="goal-score">{g.score_after}</span>
                    <span className="goal-who">{who(g)}</span>
                    <span className="muted">
                      {g.minute != null ? `${g.minute}. Minute · ` : ""}
                      {fmtClock(g.t)} im Video
                    </span>
                    <div className="spacer" />
                    {g.clip_id ? (
                      <span className="muted" data-testid="goal-has-clip">
                        {g.clip_status === "ready" ? "Clip fertig" : g.clip_status === "failed" ? "Clip fehlgeschlagen" : "Clip wird gerendert…"}
                      </span>
                    ) : (
                      <button className="btn sm" disabled={!!busy[g.t]} onClick={() => makeClips([g])}>
                        {busy[g.t] ? "…" : "Clip erstellen"}
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </div>
  );
}
