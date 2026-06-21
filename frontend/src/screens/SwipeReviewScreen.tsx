import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import type { Project } from "../lib/types";
import { fmtDuration, scoreColor } from "../lib/format";
import { orderReviewClips } from "../lib/reviewOrder";
import { useT } from "../lib/i18n";

export default function SwipeReviewScreen({
  project,
  onChange,
  onExit,
}: {
  project: Project;
  onChange: (p: Project) => void;
  onExit: () => void;
}) {
  const { t } = useT();
  const clips = useMemo(() => orderReviewClips(project.clips), [project.clips]);
  const [idx, setIdx] = useState(0);
  const [dragY, setDragY] = useState(0);
  const [busy, setBusy] = useState(false);
  const startY = useRef<number | null>(null);

  // Clamp so a shrinking list (a clip losing its render) can't strand the index
  // past the end.
  const safeIdx = clips.length ? Math.min(idx, clips.length - 1) : 0;
  const active = clips[safeIdx] ?? null;
  const next = clips.length > 1 ? clips[(safeIdx + 1) % clips.length] : null;
  const prev = () => setIdx((i) => (clips.length ? (Math.min(i, clips.length - 1) - 1 + clips.length) % clips.length : 0));
  const forward = () => setIdx((i) => (clips.length ? (Math.min(i, clips.length - 1) + 1) % clips.length : 0));

  const rate = async (rating: "up" | "down" | "none") => {
    if (!active) return;
    setBusy(true);
    try {
      await api.rateClip(project.id, active.id, rating);
      onChange(await api.getProject(project.id));
      if (rating === "down") forward();
    } finally {
      setBusy(false);
    }
  };

  const pointerUp = () => {
    if (Math.abs(dragY) > 110) {
      dragY < 0 ? forward() : prev();
    }
    startY.current = null;
    setDragY(0);
  };

  if (!active) {
    return (
      <div className="container">
        <div className="empty">
          <h3>{t("swipe.emptyTitle")}</h3>
          <button className="btn" onClick={onExit}>{t("swipe.toGrid")}</button>
        </div>
      </div>
    );
  }

  const reasons = active.factors.slice(0, 3);

  return (
    <div className="swipe-review">
      <div className="swipe-topbar">
        <button className="btn ghost sm" onClick={onExit}>{t("swipe.grid")}</button>
        <span className="pill">{safeIdx + 1} / {clips.length}</span>
        <a className="btn ghost sm" href={api.exportPremiereUrl(project.id)}>{t("swipe.premiereEdl")}</a>
      </div>

      <div
        className="swipe-card"
        style={{ transform: `translateY(${dragY}px) rotate(${dragY * 0.015}deg)` }}
        onPointerDown={(e) => {
          startY.current = e.clientY;
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (startY.current === null) return;
          setDragY(e.clientY - startY.current);
        }}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
      >
        <video
          key={active.id}
          src={active.export_url ?? undefined}
          poster={active.thumb_url ?? undefined}
          controls
          autoPlay
          loop
          playsInline
        />
        <div className="swipe-gradient" />
        <div className="swipe-overlay">
          <div className="swipe-score" style={{ color: scoreColor(active.score) }}>{active.score}</div>
          <div>
            <h2>{active.title || t("swipe.untitled")}</h2>
            <p>{fmtDuration(active.tightened_duration ?? active.end - active.start)} - {active.kind}</p>
            <div className="swipe-reasons">
              {reasons.map((f, i) => (
                <span key={i} className="pill">{f.label}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {next?.export_url && <video className="preload-video" src={next.export_url} preload="auto" muted />}

      <div className="swipe-actions">
        <button className="btn ghost" onClick={prev}>{t("swipe.back")}</button>
        <button className="btn ghost" disabled={busy} onClick={() => rate("down")}>{t("swipe.bad")}</button>
        <button className="btn primary" disabled={busy} onClick={() => rate("up")}>{t("swipe.good")}</button>
        <button className="btn ghost" onClick={forward}>{t("swipe.next")}</button>
        <Link className="btn ghost" to={`/p/${project.id}/clip/${active.id}`}>{t("swipe.edit")}</Link>
        <a className="btn ghost" href={api.downloadClipUrl(project.id, active.id)} download>{t("swipe.download")}</a>
      </div>
    </div>
  );
}
