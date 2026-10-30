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
  const [showTikTokUI, setShowTikTokUI] = useState(true);
  const [showSafeGuides, setShowSafeGuides] = useState(false);
  const [copied, setCopied] = useState(false);
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

  const copyCaption = () => {
    if (!active) return;
    const tags = active.hashtags && active.hashtags.length
      ? active.hashtags.join(" ")
      : "#gaming #fyp #viral #streamer";
    const text = `${active.title || "Stream Highlight"}\n\n${tags}`;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
        <button
          className={"btn ghost sm" + (showTikTokUI ? " on" : "")}
          onClick={() => setShowTikTokUI(!showTikTokUI)}
          title="Toggle TikTok UI elements to verify layout"
        >
          📱 TikTok UI: {showTikTokUI ? "ON" : "OFF"}
        </button>
        <button
          className={"btn ghost sm" + (showSafeGuides ? " on" : "")}
          onClick={() => setShowSafeGuides(!showSafeGuides)}
          title="Toggle safe margin guide overlay"
        >
          🎯 Safe Zone: {showSafeGuides ? "ON" : "OFF"}
        </button>
        <a className="btn ghost sm" href={api.exportPremiereUrl(project.id)}>{t("swipe.premiereEdl")}</a>
      </div>

      <div className="phone-mockup-wrap">
        <div
          className="phone-mockup"
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
          <div className="dynamic-island" />
          {showSafeGuides && (
            <div className="safe-zone-guides">
              <span className="safe-zone-label">TikTok Safe Zone</span>
            </div>
          )}
          <div className="phone-screen">
            <video
              key={active.id}
              src={active.export_url ?? undefined}
              poster={active.thumb_url ?? undefined}
              controls={!showTikTokUI}
              autoPlay
              loop
              playsInline
            />
            {showTikTokUI && (
              <div className="tiktok-overlay">
                <div className="tiktok-top-tabs">
                  <span>Following</span>
                  <span className="active">For You</span>
                </div>
                <div className="tiktok-content-layer">
                  <div className="tiktok-meta">
                    <div className="tiktok-author">
                      <span>@streamer</span>
                      <span className="pill" style={{ background: "var(--accent)", color: "#fff", padding: "1px 6px", fontSize: 10 }}>LIVE</span>
                    </div>
                    <div className="tiktok-caption">
                      {active.title || t("swipe.untitled")}
                    </div>
                    <div className="tiktok-sound">
                      <span className="music-icon">♫</span>
                      <span>Original Audio - Stream Highlights</span>
                    </div>
                  </div>
                  <div className="tiktok-actions-rail">
                    <div className="tiktok-action-item">
                      <div className="tiktok-action-btn" style={{ background: "var(--accent)" }}>👤</div>
                    </div>
                    <div className="tiktok-action-item">
                      <div className="tiktok-action-btn" style={{ color: "var(--accent-2)" }}>❤️</div>
                      <span className="tiktok-action-count">{Math.round(active.score * 1.5)}K</span>
                    </div>
                    <div className="tiktok-action-item">
                      <div className="tiktok-action-btn">💬</div>
                      <span className="tiktok-action-count">1.4K</span>
                    </div>
                    <div className="tiktok-action-item">
                      <div className="tiktok-action-btn">🔖</div>
                      <span className="tiktok-action-count">8.2K</span>
                    </div>
                    <div className="tiktok-action-item">
                      <div className="tiktok-action-btn">↗️</div>
                      <span className="tiktok-action-count">Share</span>
                    </div>
                    <div className="tiktok-disc" />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {next?.export_url && <video className="preload-video" src={next.export_url} preload="auto" muted />}

      <div className="swipe-actions">
        <button className="btn ghost" onClick={prev}>{t("swipe.back")}</button>
        <button className="btn ghost" disabled={busy} onClick={() => rate("down")}>{t("swipe.bad")}</button>
        <button className="btn primary" disabled={busy} onClick={() => rate("up")}>{t("swipe.good")}</button>
        <button className="btn ghost" onClick={forward}>{t("swipe.next")}</button>
        <button
          className={"copy-tiktok-btn" + (copied ? " copied" : "")}
          onClick={copyCaption}
          style={{ width: "100%", justifyContent: "center" }}
        >
          {copied ? "✓ Copied!" : "📋 Copy Caption"}
        </button>
        <Link className="btn ghost" to={`/p/${project.id}/clip/${active.id}`}>{t("swipe.edit")}</Link>
        <a className="btn ghost" href={api.downloadClipUrl(project.id, active.id)} download>{t("swipe.download")}</a>
      </div>
    </div>
  );
}
