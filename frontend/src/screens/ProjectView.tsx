import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Project, StatusPayload } from "../lib/types";
import ProcessingView from "../components/ProcessingView";
import ClipGridView from "../components/ClipGridView";

export default function ProjectView() {
  const { projectId } = useParams();
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let alive = true;
    let polling = false;
    let done = false;

    // Returns true once the project hit a terminal state.
    const handle = async (s: StatusPayload): Promise<boolean> => {
      setStatus(s);
      if (s.status === "ready") {
        const full = await api.getProject(projectId);
        if (alive) setProject(full);
        return true;
      }
      return s.status === "failed";
    };

    const poll = async () => {
      try {
        const s = await api.status(projectId);
        if (!alive) return;
        if (await handle(s)) return;
        timer.current = window.setTimeout(poll, 1200);
      } catch (e: any) {
        if (alive) setError(e.message ?? "could not load project");
      }
    };
    const startPolling = () => {
      if (!polling && alive) {
        polling = true;
        poll();
      }
    };

    // Live updates over WebSocket; any hiccup falls back to polling, so the
    // old behaviour is always the floor.
    let ws: WebSocket | null = null;
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/projects/${projectId}/ws`);
      ws.onmessage = (ev) => {
        if (!alive) return;
        const s = JSON.parse(ev.data);
        if (s.error) {
          setError(s.error);
          done = true;
          return;
        }
        if (s.status === "ready" || s.status === "failed") done = true;
        void handle(s);
      };
      ws.onerror = () => {
        if (!done) startPolling();
      };
      ws.onclose = () => {
        if (!done && alive) startPolling();
      };
    } catch {
      startPolling();
    }

    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
      ws?.close();
    };
  }, [projectId]);

  if (error)
    return (
      <div className="container">
        <div className="empty">
          <h3>Couldn’t load this project</h3>
          <p>{error}</p>
          <Link className="btn" to="/">
            Back to start
          </Link>
        </div>
      </div>
    );

  if (!status) return <div className="container"><div className="empty"><span className="spinner" /></div></div>;

  if (status.status === "failed")
    return (
      <div className="container">
        <div className="empty">
          <h3>Processing failed</h3>
          <p className="muted">{status.error ?? "The pipeline hit an error on this video."}</p>
          <Link className="btn" to="/">
            Try another video
          </Link>
        </div>
      </div>
    );

  if (status.status === "ready" && project)
    return <ClipGridView project={project} onChange={setProject} />;

  return <ProcessingView status={status} projectId={projectId!} />;
}
