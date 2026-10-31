/**
 * Multi-Platform Phone Canvas Simulator Component (R5)
 * Milestone M4: Requirements R5 (PROJECT.md § 16, 17 & Explorer UI Handoff § 2.3)
 *
 * Provides real-time switching between TikTok, YouTube Shorts, and Instagram Reels overlays,
 * pixel-accurate UI chrome with action rails, bottom metadata, and top system bars,
 * real-time AABB collision warnings, and a one-click Auto-Nudge safe repositioning action.
 */

import React, { useState, useMemo } from "react";
import type { CaptionWord, Rect } from "../lib/types";
import {
  PLATFORM_UI_ZONES,
  checkPlatformCollisions,
  autoNudgeSafePosition,
  type PlatformType,
  type PlatformUIZone,
  type CollisionResult,
} from "../lib/collision";

export interface PhoneSimulatorProps {
  children?: React.ReactNode;
  /** Normalized caption bounding box [0, 1] */
  captionRect?: Rect | null;
  /** Normalized caption vertical position fraction [0, 1] */
  captionYFrac?: number;
  /** Active caption words */
  captionWords?: CaptionWord[];
  /** Normalized facecam bounding box [0, 1] */
  facecamRect?: Rect | null;
  /** Current playback time in seconds */
  currentTime?: number;
  /** Active platform overlay */
  platform?: PlatformType;
  initialPlatform?: PlatformType;
  onPlatformChange?: (platform: PlatformType) => void;
  /** Callback when user clicks Auto-Nudge to reposition colliding caption */
  onNudgeCaption?: (safeRect: Rect, safeYFrac: number) => void;
  /** Callback when user clicks Auto-Nudge to reposition colliding facecam */
  onNudgeFacecam?: (safeRect: Rect) => void;
  /** Whether platform UI overlay is visible */
  showOverlay?: boolean;
  onToggleOverlay?: (visible: boolean) => void;
  /** Custom creator details */
  creatorHandle?: string;
  clipTitle?: string;
  audioTitle?: string;
  className?: string;
  style?: React.CSSProperties;
}

export default function PhoneSimulator({
  children,
  captionRect,
  captionYFrac = 0.78,
  captionWords = [],
  facecamRect = null,
  currentTime = 0,
  platform: controlledPlatform,
  initialPlatform = "tiktok",
  onPlatformChange,
  onNudgeCaption,
  onNudgeFacecam,
  showOverlay: controlledShowOverlay,
  onToggleOverlay,
  creatorHandle = "@streamer",
  clipTitle = "Insane clutch play! #gaming #viral #clipforge",
  audioTitle = "Original Audio - Stream Highlights",
  className = "",
  style = {},
}: PhoneSimulatorProps) {
  const [internalPlatform, setInternalPlatform] = useState<PlatformType>(initialPlatform);
  const [internalShowOverlay, setInternalShowOverlay] = useState(true);
  const [nudgeFeedback, setNudgeFeedback] = useState<string | null>(null);

  const activePlatform: PlatformType = controlledPlatform ?? internalPlatform;
  const overlayVisible: boolean = controlledShowOverlay ?? internalShowOverlay;

  const handleSelectPlatform = (p: PlatformType) => {
    setInternalPlatform(p);
    onPlatformChange?.(p);
  };

  const handleToggleOverlay = () => {
    const next = !overlayVisible;
    setInternalShowOverlay(next);
    onToggleOverlay?.(next);
  };

  // Derive effective caption rect in normalized [0, 1] space
  const effectiveCaptionRect: Rect = useMemo(() => {
    if (captionRect) return captionRect;
    // Standard caption box: centered inside safe width (leaving right rail clear at x=0.82)
    return {
      x: 0.08,
      y: Math.max(0.02, captionYFrac - 0.04),
      w: 0.72,
      h: 0.08,
    };
  }, [captionRect, captionYFrac]);

  // Real-time collision detection against current platform UI zones
  const captionCollisions: CollisionResult[] = useMemo(() => {
    return checkPlatformCollisions(effectiveCaptionRect, activePlatform, "Captions");
  }, [effectiveCaptionRect, activePlatform]);

  const facecamCollisions: CollisionResult[] = useMemo(() => {
    if (!facecamRect) return [];
    return checkPlatformCollisions(facecamRect, activePlatform, "Facecam");
  }, [facecamRect, activePlatform]);

  const allCollisions: CollisionResult[] = useMemo(() => {
    return [...captionCollisions, ...facecamCollisions];
  }, [captionCollisions, facecamCollisions]);

  const hasCollision = allCollisions.length > 0;

  // Collision map for zone highlighting
  const collidingZoneNames = useMemo(() => {
    const set = new Set<string>();
    for (const c of allCollisions) {
      set.add(c.zoneName);
    }
    return set;
  }, [allCollisions]);

  // Handle Auto-Nudge action
  const handleAutoNudge = () => {
    if (captionCollisions.length > 0) {
      const safeRect = autoNudgeSafePosition(effectiveCaptionRect, activePlatform, {
        elementRole: "caption",
      });
      const safeYFrac = Number((safeRect.y + safeRect.h / 2).toFixed(3));
      onNudgeCaption?.(safeRect, safeYFrac);
      setNudgeFeedback("Captions nudged to safe zone!");
      setTimeout(() => setNudgeFeedback(null), 2500);
    }

    if (facecamCollisions.length > 0 && facecamRect) {
      const safeCam = autoNudgeSafePosition(facecamRect, activePlatform, {
        elementRole: "facecam",
      });
      onNudgeFacecam?.(safeCam);
      setNudgeFeedback("Facecam nudged to safe zone!");
      setTimeout(() => setNudgeFeedback(null), 2500);
    }
  };

  return (
    <div
      className={`phone-simulator-container ${className}`}
      data-testid="phone-simulator"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "100%",
        maxWidth: 480,
        margin: "0 auto",
        ...style,
      }}
    >
      {/* Platform Selection Toolbar */}
      <div
        className="simulator-toolbar"
        data-testid="simulator-toolbar"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          width: "100%",
          padding: "8px 12px",
          background: "rgba(17, 22, 36, 0.95)",
          borderRadius: "10px 10px 0 0",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          borderBottom: "none",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: "var(--muted, #94a3b8)" }}>
            📱 Platform:
          </span>
          <div className="seg platform-tabs" style={{ display: "flex", background: "rgba(0,0,0,0.4)", borderRadius: 6, padding: 2 }}>
            <button
              type="button"
              data-testid="platform-tab-tiktok"
              className={activePlatform === "tiktok" ? "on" : ""}
              onClick={() => handleSelectPlatform("tiktok")}
              style={{
                background: activePlatform === "tiktok" ? "var(--accent, #8b5cf6)" : "transparent",
                color: activePlatform === "tiktok" ? "#ffffff" : "var(--muted, #94a3b8)",
                border: "none",
                borderRadius: 4,
                padding: "4px 10px",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              TikTok
            </button>
            <button
              type="button"
              data-testid="platform-tab-shorts"
              className={activePlatform === "shorts" ? "on" : ""}
              onClick={() => handleSelectPlatform("shorts")}
              style={{
                background: activePlatform === "shorts" ? "#ef4444" : "transparent",
                color: activePlatform === "shorts" ? "#ffffff" : "var(--muted, #94a3b8)",
                border: "none",
                borderRadius: 4,
                padding: "4px 10px",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              Shorts
            </button>
            <button
              type="button"
              data-testid="platform-tab-reels"
              className={activePlatform === "reels" ? "on" : ""}
              onClick={() => handleSelectPlatform("reels")}
              style={{
                background: activePlatform === "reels" ? "#ec4899" : "transparent",
                color: activePlatform === "reels" ? "#ffffff" : "var(--muted, #94a3b8)",
                border: "none",
                borderRadius: 4,
                padding: "4px 10px",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              Reels
            </button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <button
            type="button"
            data-testid="toggle-overlay"
            onClick={handleToggleOverlay}
            style={{
              background: overlayVisible ? "rgba(6, 182, 212, 0.2)" : "rgba(255, 255, 255, 0.05)",
              color: overlayVisible ? "#06b6d4" : "var(--muted, #94a3b8)",
              border: "1px solid rgba(6, 182, 212, 0.3)",
              borderRadius: 6,
              padding: "4px 10px",
              fontSize: 12,
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            {overlayVisible ? "👁️ Hide UI" : "👁️ Show UI"}
          </button>
        </div>
      </div>

      {/* Collision Warning & Auto-Nudge Action Bar */}
      <div
        data-testid="collision-status-bar"
        style={{
          width: "100%",
          padding: "6px 12px",
          background: hasCollision ? "rgba(239, 68, 68, 0.18)" : "rgba(16, 185, 129, 0.12)",
          borderLeft: `3px solid ${hasCollision ? "#ef4444" : "#10b981"}`,
          borderRight: "1px solid rgba(255, 255, 255, 0.1)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: 12,
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, flex: 1, minWidth: 200 }}>
          {hasCollision ? (
            <div
              data-testid="collision-warning-badge"
              style={{
                color: "#fca5a5",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span>⚠️</span>
              <span>
                {allCollisions[0]?.message || `Collision with ${activePlatform.toUpperCase()} UI`}
                {allCollisions.length > 1 ? ` (+${allCollisions.length - 1} more)` : ""}
              </span>
            </div>
          ) : (
            <div
              data-testid="safe-status-badge"
              style={{
                color: "#6ee7b7",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <span>✅</span>
              <span>Safe: No UI collisions detected on {activePlatform.toUpperCase()}</span>
            </div>
          )}
        </div>

        {hasCollision && (
          <button
            type="button"
            data-testid="auto-nudge-btn"
            onClick={handleAutoNudge}
            style={{
              background: "#ef4444",
              color: "#ffffff",
              border: "none",
              borderRadius: 6,
              padding: "4px 12px",
              fontSize: 11,
              fontWeight: 700,
              cursor: "pointer",
              boxShadow: "0 2px 8px rgba(239, 68, 68, 0.4)",
              transition: "transform 0.1s ease",
            }}
            title="Auto-nudge colliding elements into platform safe zone"
          >
            ⚡ Auto-Nudge Safe
          </button>
        )}

        {nudgeFeedback && (
          <span
            data-testid="nudge-feedback"
            style={{ color: "#38bdf8", fontSize: 11, fontWeight: 700 }}
          >
            {nudgeFeedback}
          </span>
        )}
      </div>

      {/* Phone Mockup Frame (9:16 Vertical Screen Viewport) */}
      <div
        className="phone-bezel"
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "9 / 16",
          maxHeight: "72vh",
          background: "#000000",
          borderRadius: "0 0 24px 24px",
          overflow: "hidden",
          border: "2px solid rgba(255, 255, 255, 0.12)",
          boxShadow: "0 12px 36px rgba(0, 0, 0, 0.75)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Dynamic Island / Phone Top Speaker */}
        <div
          className="phone-dynamic-island"
          data-testid="phone-dynamic-island"
          style={{
            position: "absolute",
            top: 6,
            left: "50%",
            transform: "translateX(-50%)",
            width: 80,
            height: 14,
            background: "#111827",
            borderRadius: 10,
            zIndex: 40,
            pointerEvents: "none",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
          }}
        >
          <div style={{ width: 5, height: 5, borderRadius: "50%", background: "#1f2937" }} />
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#065f46" }} />
        </div>

        {/* Video & Interactive Canvas Viewport (Underlay) */}
        <div
          className="phone-screen"
          data-testid="phone-screen"
          style={{
            position: "relative",
            width: "100%",
            height: "100%",
            overflow: "hidden",
          }}
        >
          {children}

          {/* ============================================================== */}
          {/* Platform Overlays Layer                                        */}
          {/* ============================================================== */}
          {overlayVisible && (
            <div
              className={`platform-overlay overlay-${activePlatform}`}
              data-testid="platform-overlay"
              data-platform={activePlatform}
              style={{
                position: "absolute",
                inset: 0,
                pointerEvents: "none",
                zIndex: 30,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                padding: "8px 12px 14px 12px",
                color: "#ffffff",
                textShadow: "0 1px 3px rgba(0,0,0,0.8)",
                boxSizing: "border-box",
              }}
            >
              {/* ------------------------------------------------------------ */}
              {/* 1. TIKTOK OVERLAY                                            */}
              {/* ------------------------------------------------------------ */}
              {activePlatform === "tiktok" && (
                <>
                  {/* Top Navigation Tabs */}
                  <div
                    data-testid="top-tabs-tiktok"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: 18,
                      padding: "0 8px",
                      border: collidingZoneNames.has("top_tabs")
                        ? "2px solid #ef4444"
                        : "none",
                      borderRadius: 6,
                      background: collidingZoneNames.has("top_tabs")
                        ? "rgba(239, 68, 68, 0.2)"
                        : "transparent",
                    }}
                  >
                    <div style={{ fontSize: 13, opacity: 0.8 }}>🔴 LIVE</div>
                    <div style={{ display: "flex", gap: 14, fontSize: 13, fontWeight: 700 }}>
                      <span style={{ opacity: 0.6 }}>Following</span>
                      <span style={{ borderBottom: "2px solid #fff", paddingBottom: 2 }}>
                        For You
                      </span>
                    </div>
                    <div style={{ fontSize: 14 }}>🔍</div>
                  </div>

                  {/* Mid-to-Bottom: Action Rail & Creator Metadata */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-end",
                      width: "100%",
                      marginBottom: 10,
                    }}
                  >
                    {/* Bottom Metadata */}
                    <div
                      data-testid="bottom-meta-tiktok"
                      style={{
                        flex: 1,
                        maxWidth: "76%",
                        display: "flex",
                        flexDirection: "column",
                        gap: 4,
                        border: collidingZoneNames.has("bottom_creator_sound")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: 4,
                        background: collidingZoneNames.has("bottom_creator_sound")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ fontWeight: 800, fontSize: 13 }}>{creatorHandle}</span>
                        <span
                          style={{
                            background: "var(--accent, #8b5cf6)",
                            color: "#fff",
                            fontSize: 9,
                            fontWeight: 800,
                            padding: "1px 5px",
                            borderRadius: 4,
                          }}
                        >
                          FOLLOW
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          lineHeight: 1.3,
                          opacity: 0.95,
                          whiteSpace: "pre-wrap",
                        }}
                      >
                        {clipTitle}
                      </div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 5,
                          fontSize: 10,
                          opacity: 0.85,
                        }}
                      >
                        <span>♫</span>
                        <span>{audioTitle}</span>
                      </div>
                    </div>

                    {/* Right-Side Action Rail */}
                    <div
                      data-testid="action-rail-tiktok"
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 12,
                        border: collidingZoneNames.has("right_rail_actions")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: "6px 2px",
                        background: collidingZoneNames.has("right_rail_actions")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      {/* Avatar */}
                      <div
                        style={{
                          width: 36,
                          height: 36,
                          borderRadius: "50%",
                          background: "linear-gradient(135deg, #8b5cf6, #ec4899)",
                          border: "2px solid #ffffff",
                          display: "grid",
                          placeItems: "center",
                          fontSize: 16,
                        }}
                      >
                        👤
                      </div>

                      {/* Like */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 20 }}>❤️</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>1.4M</span>
                      </div>

                      {/* Comments */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 19 }}>💬</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>8,241</span>
                      </div>

                      {/* Bookmark */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>🔖</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>120K</span>
                      </div>

                      {/* Share */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>↗️</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>34.5K</span>
                      </div>

                      {/* Rotating Vinyl Disc */}
                      <div
                        className="tiktok-disc"
                        style={{
                          width: 30,
                          height: 30,
                          borderRadius: "50%",
                          background: "#111",
                          border: "4px solid #333",
                          boxShadow: "0 0 0 1px rgba(255,255,255,0.4)",
                        }}
                      />
                    </div>
                  </div>
                </>
              )}

              {/* ------------------------------------------------------------ */}
              {/* 2. YOUTUBE SHORTS OVERLAY                                    */}
              {/* ------------------------------------------------------------ */}
              {activePlatform === "shorts" && (
                <>
                  {/* Top Navigation */}
                  <div
                    data-testid="top-nav-shorts"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: 18,
                      padding: "0 6px",
                      border: collidingZoneNames.has("top_search_controls")
                        ? "2px solid #ef4444"
                        : "none",
                      borderRadius: 6,
                      background: collidingZoneNames.has("top_search_controls")
                        ? "rgba(239, 68, 68, 0.2)"
                        : "transparent",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ color: "#ef4444", fontWeight: 900, fontSize: 16 }}>▶</span>
                      <span style={{ fontWeight: 800, fontSize: 14 }}>Shorts</span>
                    </div>
                    <div style={{ display: "flex", gap: 14, fontSize: 16 }}>
                      <span>🔍</span>
                      <span>📷</span>
                      <span>⋮</span>
                    </div>
                  </div>

                  {/* Mid-to-Bottom: Action Rail & Channel Metadata */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-end",
                      width: "100%",
                      marginBottom: 10,
                    }}
                  >
                    {/* Bottom Metadata */}
                    <div
                      data-testid="bottom-meta-shorts"
                      style={{
                        flex: 1,
                        maxWidth: "76%",
                        display: "flex",
                        flexDirection: "column",
                        gap: 5,
                        border: collidingZoneNames.has("bottom_channel_title")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: 4,
                        background: collidingZoneNames.has("bottom_channel_title")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div
                          style={{
                            width: 24,
                            height: 24,
                            borderRadius: "50%",
                            background: "#ef4444",
                            display: "grid",
                            placeItems: "center",
                            fontSize: 12,
                          }}
                        >
                          👤
                        </div>
                        <span style={{ fontWeight: 800, fontSize: 12 }}>{creatorHandle}</span>
                        <span
                          style={{
                            background: "#ef4444",
                            color: "#fff",
                            fontSize: 9,
                            fontWeight: 800,
                            padding: "2px 7px",
                            borderRadius: 12,
                          }}
                        >
                          Subscribe
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          lineHeight: 1.3,
                          opacity: 0.95,
                        }}
                      >
                        {clipTitle}
                      </div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 4,
                          fontSize: 10,
                          opacity: 0.85,
                        }}
                      >
                        <span>🎶</span>
                        <span>{audioTitle}</span>
                      </div>
                    </div>

                    {/* Right-Side Engagement Rail */}
                    <div
                      data-testid="action-rail-shorts"
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 12,
                        border: collidingZoneNames.has("right_rail_engagement")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: "6px 2px",
                        background: collidingZoneNames.has("right_rail_engagement")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      {/* Thumbs Up */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>👍</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>852K</span>
                      </div>

                      {/* Thumbs Down */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>👎</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>Dislike</span>
                      </div>

                      {/* Comments */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>💬</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>3,110</span>
                      </div>

                      {/* Share */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>➡️</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>Share</span>
                      </div>

                      {/* Remix */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>🔀</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>Remix</span>
                      </div>

                      {/* Audio square thumbnail */}
                      <div
                        style={{
                          width: 26,
                          height: 26,
                          borderRadius: 4,
                          background: "#333",
                          border: "1px solid #666",
                          display: "grid",
                          placeItems: "center",
                          fontSize: 11,
                        }}
                      >
                        🎵
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* ------------------------------------------------------------ */}
              {/* 3. INSTAGRAM REELS OVERLAY                                   */}
              {/* ------------------------------------------------------------ */}
              {activePlatform === "reels" && (
                <>
                  {/* Top Navigation */}
                  <div
                    data-testid="top-nav-reels"
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: 18,
                      padding: "0 6px",
                      border: collidingZoneNames.has("top_camera_nav")
                        ? "2px solid #ef4444"
                        : "none",
                      borderRadius: 6,
                      background: collidingZoneNames.has("top_camera_nav")
                        ? "rgba(239, 68, 68, 0.2)"
                        : "transparent",
                    }}
                  >
                    <span style={{ fontWeight: 800, fontSize: 16 }}>Reels</span>
                    <span style={{ fontSize: 18 }}>📷</span>
                  </div>

                  {/* Mid-to-Bottom: Action Rail & Profile Metadata */}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-end",
                      width: "100%",
                      marginBottom: 10,
                    }}
                  >
                    {/* Bottom Metadata */}
                    <div
                      data-testid="bottom-meta-reels"
                      style={{
                        flex: 1,
                        maxWidth: "76%",
                        display: "flex",
                        flexDirection: "column",
                        gap: 4,
                        border: collidingZoneNames.has("bottom_reels_audio_bar")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: 4,
                        background: collidingZoneNames.has("bottom_reels_audio_bar")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div
                          style={{
                            width: 24,
                            height: 24,
                            borderRadius: "50%",
                            background: "linear-gradient(45deg, #f09433, #e6683c, #dc2743, #cc2366, #bc1888)",
                            display: "grid",
                            placeItems: "center",
                            fontSize: 12,
                          }}
                        >
                          👤
                        </div>
                        <span style={{ fontWeight: 800, fontSize: 12 }}>{creatorHandle}</span>
                        <span
                          style={{
                            border: "1px solid #ffffff",
                            fontSize: 9,
                            fontWeight: 700,
                            padding: "1px 6px",
                            borderRadius: 4,
                          }}
                        >
                          Follow
                        </span>
                      </div>
                      <div
                        style={{
                          fontSize: 11,
                          lineHeight: 1.3,
                          opacity: 0.95,
                        }}
                      >
                        {clipTitle}
                      </div>
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 4,
                          fontSize: 10,
                          opacity: 0.85,
                        }}
                      >
                        <span>🎵</span>
                        <span>{audioTitle}</span>
                      </div>
                    </div>

                    {/* Right-Side Action Rail */}
                    <div
                      data-testid="action-rail-reels"
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 12,
                        border: collidingZoneNames.has("right_rail_reels_actions")
                          ? "2px solid #ef4444"
                          : "none",
                        borderRadius: 6,
                        padding: "6px 2px",
                        background: collidingZoneNames.has("right_rail_reels_actions")
                          ? "rgba(239, 68, 68, 0.2)"
                          : "transparent",
                      }}
                    >
                      {/* Heart */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 20 }}>🤍</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>420K</span>
                      </div>

                      {/* Comment */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 19 }}>💬</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>1,890</span>
                      </div>

                      {/* Paper Airplane (Send) */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 18 }}>✈️</span>
                        <span style={{ fontSize: 10, fontWeight: 700 }}>45K</span>
                      </div>

                      {/* 3 dots */}
                      <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                        <span style={{ fontSize: 16 }}>⋮</span>
                      </div>

                      {/* Audio square */}
                      <div
                        style={{
                          width: 24,
                          height: 24,
                          borderRadius: 4,
                          border: "1px solid #ffffff",
                          background: "#222",
                          display: "grid",
                          placeItems: "center",
                          fontSize: 10,
                        }}
                      >
                        🎧
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
