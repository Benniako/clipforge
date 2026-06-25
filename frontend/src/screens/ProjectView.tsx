import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api";
import type { Project, StatusPayload } from "../lib/types";
import { useT } from "../lib/i18n";
import Toast, { type ToastMsg } from "../components/Toast";
import EmptyState from "../components/EmptyState";
import ProcessingView from "../components/ProcessingView";
import ClipGridView from "../components/ClipGridView";
import SwipeReviewScreen from "./SwipeReviewScreen";

export default function ProjectView() {
  const { t } = useT();
  const { projectId } = useParams();
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastMsg | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "swipe">("grid");
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
        if (alive) setError(e.message ?? t("pv.loadError"));
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
        let s: any;
        try { s = JSON.parse(ev.data); } catch { return; }
        if (s.error) {
          setError(s.error);
          done = true;
          return;
        }
        if (s.status === "ready" || s.status === "failed") done = true;
        handle(s).catch((err) => {
          if (alive) setError(err?.message ?? "Failed to load project");
        });
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
        <EmptyState icon="❌" title={t("pv.loadError")}
          text={error}
          action={<Link className="btn" to="/">{t("pv.back")}</Link>} />
      </div>
    );

  if (!status) return <div className="container"><div className="empty"><span className="spinner" /></div></div>;

  if (status.status === "failed")
    return (
      <div className="container">
        <EmptyState icon="💥" title={t("pv.failedTitle")}
          text={status.error ?? t("pv.failedGeneric")}
          action={<Link className="btn" to="/">{t("pv.tryAnother")}</Link>} />
      </div>
    );

  if (status.status === "ready" && project) {
    if (viewMode === "swipe") {
      return <SwipeReviewScreen project={project} onChange={setProject} onExit={() => setViewMode("grid")} />;
    }
    return (
      <>
        <div className="container view-switch">
          <button className="btn primary sm" onClick={() => setViewMode("swipe")}>
            {t("pv.swipe")}
          </button>
          {/* Streamer.bot webhook: show the mark-highlight URL so users
              can copy it into Streamer.bot's HTTP request action. */}
          <span className="muted tiny" style={{ marginLeft: 12, fontSize: 11 }}>
            🎯 Webhook:
            <code style={{ marginLeft: 6, fontSize: 10, background: "var(--bg)", padding: "2px 6px", borderRadius: 4 }}>
              POST /api/projects/{projectId}/mark-highlight?timestamp=SEC&amp;duration=30
            </code>
          </span>
        </div>
        <ClipGridView project={project} onChange={setProject} />
        <Toast msg={toast} onDone={() => setToast(null)} />
      </>
    );
  }

  return (
    <>
      <ProcessingView status={status} projectId={projectId!} onStatus={setStatus} />
      <Toast msg={toast} onDone={() => setToast(null)} />
    </>
  );
}
