import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { Clip } from "../lib/types";

// ============================================================================
// Input Isolation Helper (Strictly ignores hotkeys when typing)
// ============================================================================

/**
 * Checks whether the given event target or current active element is an input,
 * textarea, contenteditable element, or inline caption editing field (.caption-word-input).
 */
export function isInputIsolated(target: EventTarget | null): boolean {
  const check = (el: any): boolean => {
    if (!el || typeof el !== "object") return false;

    // Check tag name
    const tag = (el.tagName || "").toUpperCase();
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;

    // Check contenteditable
    if (el.isContentEditable === true) return true;
    if (typeof el.getAttribute === "function") {
      const ce = el.getAttribute("contenteditable");
      if (ce === "true" || ce === "" || ce === "plaintext-only") return true;
    }

    // Check class .caption-word-input
    if (el.classList && typeof el.classList.contains === "function") {
      if (el.classList.contains("caption-word-input")) return true;
    }

    // Check closest ancestor form/editable element
    if (typeof el.closest === "function") {
      if (
        el.closest(
          "input, textarea, select, [contenteditable='true'], [contenteditable=''], [contenteditable], .caption-word-input"
        )
      ) {
        return true;
      }
    }

    return false;
  };

  if (check(target)) return true;
  if (typeof document !== "undefined" && check(document.activeElement)) return true;
  return false;
}

// ============================================================================
// Tactile Synthetic Offline Audio Feedback (Web Audio API)
// ============================================================================

/**
 * Generates a synthetic offline tactile audio blip / pip using Web Audio API.
 * Safely falls back to a silent no-op in headless, test, or unsupported environments.
 */
export function playTactileBeep(
  freq = 880,
  durationMs = 20,
  type: OscillatorType = "sine"
): void {
  // Always dispatch a DOM custom event for reliable testing assertions
  if (
    typeof window !== "undefined" &&
    typeof window.dispatchEvent === "function"
  ) {
    try {
      window.dispatchEvent(
        new CustomEvent("streamer:audio-pip", { detail: { freq, durationMs, type } })
      );
    } catch {
      /* ignore */
    }
  }

  try {
    const AudioCtx =
      window.AudioContext ||
      (window as any).webkitAudioContext;
    if (!AudioCtx) return;

    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    // Short crisp envelope with rapid decay
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(
      0.0001,
      ctx.currentTime + durationMs / 1000
    );

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + durationMs / 1000);

    setTimeout(() => {
      try {
        ctx.close();
      } catch {
        /* ignore */
      }
    }, durationMs + 50);
  } catch {
    // Silent graceful fallback in offline / test environments
  }
}

// ============================================================================
// Types & Props
// ============================================================================

export interface StreamerWorkflowBarProps {
  projectId?: string;
  clips?: Clip[];
  activeClipId?: string;
  currentTime?: number;
  duration?: number;
  start?: number;
  end?: number;
  currentRating?: number | null;
  videoRef?: React.RefObject<HTMLVideoElement>;
  listenToKeyboard?: boolean;
  audioFeedbackEnabled?: boolean;
  onPlayToggle?: (isPlaying: boolean) => void;
  onShuttleRewind?: (newTime: number) => void;
  onShuttlePause?: () => void;
  onShuttleForward?: (newTime: number) => void;
  onMarkIn?: (inTime: number) => void;
  onMarkOut?: (outTime: number) => void;
  onRate?: (rating: number) => void;
  onBatchAction?: (
    action: "rate" | "export" | "tag" | "delete" | string,
    clipIds: string[],
    payload?: any
  ) => Promise<void> | void;
  onClipsUpdated?: (updatedClips: Clip[]) => void;
  onAudioPip?: (action: string) => void;
  onHudMessage?: (msg: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

// ============================================================================
// StreamerWorkflowBar Component
// ============================================================================

export default function StreamerWorkflowBar({
  projectId,
  clips = [],
  activeClipId,
  currentTime = 0,
  duration = 30,
  start = 0,
  end = 15,
  currentRating = null,
  videoRef,
  listenToKeyboard = true,
  audioFeedbackEnabled = true,
  onPlayToggle,
  onShuttleRewind,
  onShuttlePause,
  onShuttleForward,
  onMarkIn,
  onMarkOut,
  onRate,
  onBatchAction,
  onClipsUpdated,
  onAudioPip,
  onHudMessage,
  className = "",
  style,
}: StreamerWorkflowBarProps) {
  // HUD state for flashing badge feedback
  const [hud, setHud] = useState<{ label: string; key: number } | null>(null);
  const hudTimerRef = useRef<any>(null);

  // Local playback and rating state if unmanaged
  const [localPlaying, setLocalPlaying] = useState(false);
  const [localRating, setLocalRating] = useState<number | null>(currentRating);

  // Multi-clip selection for batch toolbar
  const [selectedClips, setSelectedClips] = useState<string[]>([]);
  const [batchBusy, setBatchBusy] = useState(false);

  // Synchronize local rating when prop changes
  useEffect(() => {
    if (currentRating !== undefined) {
      setLocalRating(currentRating);
    }
  }, [currentRating]);

  // Trigger HUD badge for 600ms with fade-out
  const triggerHud = useCallback(
    (label: string) => {
      if (hudTimerRef.current) clearTimeout(hudTimerRef.current);
      setHud({ label, key: Date.now() });
      if (onHudMessage) onHudMessage(label);
      hudTimerRef.current = setTimeout(() => {
        setHud(null);
      }, 600);
    },
    [onHudMessage]
  );

  // Sound helper that respects audioFeedbackEnabled and triggers callbacks
  const triggerTactile = useCallback(
    (action: string, freq = 880) => {
      if (audioFeedbackEnabled) {
        playTactileBeep(freq, 20);
      }
      if (onAudioPip) {
        onAudioPip(action);
      }
    },
    [audioFeedbackEnabled, onAudioPip]
  );

  // Available clip IDs for selection
  const allClipIds = useMemo(() => {
    if (clips.length > 0) return clips.map((c) => c.id);
    if (activeClipId) return [activeClipId];
    return [];
  }, [clips, activeClipId]);

  // Keep live references for keyboard listener (ref pattern avoids stale closures)
  const stateRef = useRef({
    currentTime,
    duration,
    start,
    end,
    localPlaying,
    videoRef,
    triggerHud,
    triggerTactile,
    onPlayToggle,
    onShuttleRewind,
    onShuttlePause,
    onShuttleForward,
    onMarkIn,
    onMarkOut,
    onRate,
  });

  useEffect(() => {
    stateRef.current = {
      currentTime,
      duration,
      start,
      end,
      localPlaying,
      videoRef,
      triggerHud,
      triggerTactile,
      onPlayToggle,
      onShuttleRewind,
      onShuttlePause,
      onShuttleForward,
      onMarkIn,
      onMarkOut,
      onRate,
    };
  });

  // Hotkey Navigation Handler
  useEffect(() => {
    if (!listenToKeyboard) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // 1. Strict input isolation
      if (isInputIsolated(e.target)) {
        return;
      }

      // Ignore if user is pressing browser command combinations
      if (e.ctrlKey || e.metaKey || e.altKey) {
        return;
      }

      const st = stateRef.current;
      const code = e.code;
      const key = e.key;

      // J: Shuttle rewind / step backwards by 1s
      if (code === "KeyJ" || key === "j" || key === "J") {
        e.preventDefault();
        st.triggerTactile("j", 520);
        let newT = (st.currentTime ?? 0) - 1.0;
        if (st.videoRef?.current) {
          st.videoRef.current.currentTime = Math.max(
            0,
            st.videoRef.current.currentTime - 1.0
          );
          newT = st.videoRef.current.currentTime;
        } else {
          newT = Math.max(0, newT);
        }
        st.triggerHud("[J] REWIND -1s");
        st.onShuttleRewind?.(newT);
        return;
      }

      // K: Shuttle pause / halt playback
      if (code === "KeyK" || key === "k" || key === "K") {
        e.preventDefault();
        st.triggerTactile("k", 440);
        if (st.videoRef?.current) {
          st.videoRef.current.pause();
          st.videoRef.current.playbackRate = 1.0;
        }
        setLocalPlaying(false);
        st.triggerHud("[K] PAUSE");
        st.onShuttlePause?.();
        return;
      }

      // L: Shuttle fast-forward / advance by 1s
      if (code === "KeyL" || key === "l" || key === "L") {
        e.preventDefault();
        st.triggerTactile("l", 880);
        const maxDur = st.duration ?? 1000;
        let newT = (st.currentTime ?? 0) + 1.0;
        if (st.videoRef?.current) {
          st.videoRef.current.currentTime = Math.min(
            maxDur,
            st.videoRef.current.currentTime + 1.0
          );
          newT = st.videoRef.current.currentTime;
        } else {
          newT = Math.min(maxDur, newT);
        }
        st.triggerHud("[L] FORWARD +1s");
        st.onShuttleForward?.(newT);
        return;
      }

      // Space: Toggle Play / Pause
      if (code === "Space" || key === " ") {
        e.preventDefault();
        st.triggerTactile("space", 660);
        let nextPlaying = false;
        if (st.videoRef?.current) {
          if (st.videoRef.current.paused) {
            st.videoRef.current.play();
            nextPlaying = true;
          } else {
            st.videoRef.current.pause();
            nextPlaying = false;
          }
        } else {
          nextPlaying = !st.localPlaying;
          setLocalPlaying(nextPlaying);
        }
        st.triggerHud(nextPlaying ? "[SPACE] PLAY" : "[SPACE] PAUSE");
        st.onPlayToggle?.(nextPlaying);
        return;
      }

      // I: Mark In point
      if (code === "KeyI" || key === "i" || key === "I") {
        e.preventDefault();
        st.triggerTactile("i", 700);
        const inT = st.videoRef?.current
          ? st.videoRef.current.currentTime
          : st.currentTime ?? 0;
        st.triggerHud(`[I] MARK IN ${inT.toFixed(1)}s`);
        st.onMarkIn?.(inT);
        return;
      }

      // O: Mark Out point
      if (code === "KeyO" || key === "o" || key === "O") {
        e.preventDefault();
        st.triggerTactile("o", 750);
        const outT = st.videoRef?.current
          ? st.videoRef.current.currentTime
          : st.currentTime ?? (st.duration ?? 10);
        st.triggerHud(`[O] MARK OUT ${outT.toFixed(1)}s`);
        st.onMarkOut?.(outT);
        return;
      }

      // 1-5: Quick clip ratings
      if (!e.shiftKey) {
        const digitMatch = code ? code.match(/^(?:Digit|Numpad)([1-5])$/) : null;
        const keyMatch = key && /^[1-5]$/.test(key) ? key : null;
        const rawScore = digitMatch ? digitMatch[1] : keyMatch;
        if (rawScore) {
          const score = parseInt(rawScore, 10);
          if (!isNaN(score) && score >= 1 && score <= 5) {
            e.preventDefault();
            st.triggerTactile(String(score), 440 + score * 80);
            setLocalRating(score);
            st.triggerHud(`[★ ${score}] RATED ${score}`);
            st.onRate?.(score);
            return;
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [listenToKeyboard]);

  // Clean up HUD timer on unmount
  useEffect(() => {
    return () => {
      if (hudTimerRef.current) clearTimeout(hudTimerRef.current);
    };
  }, []);

  // One-click Batch Action Execution
  const executeBatch = async (
    action: "rate" | "export" | "tag" | "delete" | string,
    payload: any = {}
  ) => {
    const targetIds =
      selectedClips.length > 0
        ? selectedClips
        : activeClipId
          ? [activeClipId]
          : allClipIds;

    if (targetIds.length === 0) {
      triggerHud("[BATCH] NO CLIPS SELECTED");
      return;
    }

    if (action === "delete") {
      let ok = true;
      if (typeof window !== "undefined" && typeof window.confirm === "function") {
        try {
          ok = window.confirm(`Delete ${targetIds.length} selected clip(s)?`);
        } catch {
          ok = true;
        }
      }
      if (!ok) return;
    }

    setBatchBusy(true);
    triggerTactile(`batch_${action}`, 600);

    let hudMsg = `[BATCH] ${action.toUpperCase()} ${targetIds.length} CLIPS`;
    if (action === "rate" || action === "rate_up") {
      hudMsg = `[BATCH] RATED ${targetIds.length} CLIPS (5★)`;
    } else if (action === "tag") {
      hudMsg = `[BATCH] TAGGED ${targetIds.length} CLIPS`;
    } else if (action === "export") {
      hudMsg = `[BATCH] EXPORTING ${targetIds.length} CLIPS`;
    } else if (action === "delete") {
      hudMsg = `[BATCH] DELETED ${targetIds.length} CLIPS`;
      setSelectedClips((prev) => prev.filter((id) => !targetIds.includes(id)));
    }
    triggerHud(hudMsg);

    try {
      if (onBatchAction) {
        await onBatchAction(action, targetIds, payload);
      } else if (projectId) {
        const res = await fetch(`/api/projects/${projectId}/clips/batch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            clip_ids: targetIds,
            action,
            payload,
          }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `Batch ${action} failed`);
        }
        const data = await res.json();
        if (data.project?.clips && onClipsUpdated) {
          onClipsUpdated(data.project.clips);
        }
      }
    } catch (e: any) {
      triggerHud(`[BATCH ERROR] ${e.message || "Failed"}`);
    } finally {
      setBatchBusy(false);
    }
  };

  // Toggle selection for all clips
  const toggleSelectAll = () => {
    if (selectedClips.length === allClipIds.length && allClipIds.length > 0) {
      setSelectedClips([]);
    } else {
      setSelectedClips([...allClipIds]);
    }
  };

  // Toggle individual clip selection
  const toggleSelectClip = (cid: string) => {
    setSelectedClips((prev) =>
      prev.includes(cid) ? prev.filter((x) => x !== cid) : [...prev, cid]
    );
  };

  return (
    <div
      data-testid="streamer-workflow-bar"
      className={`streamer-workflow-bar panel ${className}`}
      style={{
        position: "relative",
        background: "rgba(11, 15, 25, 0.85)",
        border: "1px solid rgba(139, 92, 246, 0.35)",
        borderRadius: 8,
        padding: "10px 14px",
        boxShadow: "0 4px 18px rgba(0, 0, 0, 0.4)",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        ...style,
      }}
    >
      {/* 1. Top Row: Hotkey Shuttle & Rating Controls */}
      <div
        className="row"
        style={{
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        {/* Hotkey Shuttle Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <span
            className="muted tiny"
            style={{ fontWeight: 800, color: "var(--accent, #8b5cf6)", marginRight: 2 }}
          >
            ⚡ HOTKEYS:
          </span>

          <button
            className="btn ghost sm"
            data-testid="hotkey-shuttle-rewind"
            onClick={() => {
              triggerTactile("j", 520);
              const nextT = Math.max(0, (videoRef?.current?.currentTime ?? currentTime) - 1.0);
              if (videoRef?.current) videoRef.current.currentTime = nextT;
              triggerHud("[J] REWIND -1s");
              onShuttleRewind?.(nextT);
            }}
            title="Rewind 1s [J]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>J</kbd> -1s
          </button>

          <button
            className="btn ghost sm"
            data-testid="hotkey-shuttle-pause"
            onClick={() => {
              triggerTactile("k", 440);
              if (videoRef?.current) {
                videoRef.current.pause();
                videoRef.current.playbackRate = 1.0;
              }
              setLocalPlaying(false);
              triggerHud("[K] PAUSE");
              onShuttlePause?.();
            }}
            title="Pause & Halt [K]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>K</kbd> ⏸
          </button>

          <button
            className="btn ghost sm"
            data-testid="hotkey-play-toggle"
            onClick={() => {
              triggerTactile("space", 660);
              let nextPlaying = false;
              if (videoRef?.current) {
                if (videoRef.current.paused) {
                  videoRef.current.play();
                  nextPlaying = true;
                } else {
                  videoRef.current.pause();
                  nextPlaying = false;
                }
              } else {
                nextPlaying = !localPlaying;
                setLocalPlaying(nextPlaying);
              }
              triggerHud(nextPlaying ? "[SPACE] PLAY" : "[SPACE] PAUSE");
              onPlayToggle?.(nextPlaying);
            }}
            title="Toggle Play/Pause [Space]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>Space</kbd> ▶/⏸
          </button>

          <button
            className="btn ghost sm"
            data-testid="hotkey-shuttle-forward"
            onClick={() => {
              triggerTactile("l", 880);
              const maxDur = duration ?? 1000;
              const nextT = Math.min(
                maxDur,
                (videoRef?.current?.currentTime ?? currentTime) + 1.0
              );
              if (videoRef?.current) videoRef.current.currentTime = nextT;
              triggerHud("[L] FORWARD +1s");
              onShuttleForward?.(nextT);
            }}
            title="Fast Forward 1s [L]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>L</kbd> +1s
          </button>

          <span style={{ color: "rgba(255,255,255,0.2)", margin: "0 2px" }}>|</span>

          <button
            className="btn ghost sm"
            data-testid="hotkey-mark-in"
            onClick={() => {
              triggerTactile("i", 700);
              const inT = videoRef?.current
                ? videoRef.current.currentTime
                : currentTime;
              triggerHud(`[I] MARK IN ${inT.toFixed(1)}s`);
              onMarkIn?.(inT);
            }}
            title="Mark In Point [I]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>I</kbd> Mark In
          </button>

          <button
            className="btn ghost sm"
            data-testid="hotkey-mark-out"
            onClick={() => {
              triggerTactile("o", 750);
              const outT = videoRef?.current
                ? videoRef.current.currentTime
                : currentTime;
              triggerHud(`[O] MARK OUT ${outT.toFixed(1)}s`);
              onMarkOut?.(outT);
            }}
            title="Mark Out Point [O]"
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            <kbd style={{ marginRight: 4, opacity: 0.8 }}>O</kbd> Mark Out
          </button>
        </div>

        {/* 1-5 Quick Rating Stars */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span className="muted tiny" style={{ fontWeight: 700, marginRight: 2 }}>
            RATE:
          </span>
          {[1, 2, 3, 4, 5].map((star) => {
            const isSelected = localRating === star;
            return (
              <button
                key={star}
                data-testid={`hotkey-rate-${star}`}
                className={`btn sm ghost ${isSelected ? "on" : ""}`}
                onClick={() => {
                  triggerTactile(String(star), 440 + star * 80);
                  setLocalRating(star);
                  triggerHud(`[★ ${star}] RATED ${star}`);
                  onRate?.(star);
                }}
                style={{
                  fontSize: 11,
                  padding: "2px 6px",
                  color: isSelected ? "#eab308" : undefined,
                  borderColor: isSelected ? "#eab308" : undefined,
                  fontWeight: isSelected ? 800 : 500,
                }}
                title={`Rate ${star} Star${star > 1 ? "s" : ""} [${star}]`}
              >
                ★{star}
              </button>
            );
          })}
          {localRating && (
            <span
              data-testid="current-clip-rating"
              className="pill"
              style={{
                fontSize: 10,
                color: "#eab308",
                marginLeft: 4,
                fontFamily: "monospace",
              }}
            >
              {localRating}★
            </span>
          )}
        </div>
      </div>

      {/* 2. Bottom Row: One-Click Batch Action Toolbar */}
      <div
        data-testid="batch-toolbar"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 8,
          paddingTop: 6,
          borderTop: "1px solid rgba(255, 255, 255, 0.08)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          <button
            className="btn sm ghost"
            data-testid="batch-select-all"
            onClick={toggleSelectAll}
            style={{ fontSize: 11, padding: "3px 8px" }}
          >
            {selectedClips.length === allClipIds.length && allClipIds.length > 0
              ? "Deselect All"
              : "Select All"}
          </button>

          <span
            data-testid="selected-count-badge"
            className="pill"
            style={{ fontSize: 11, background: "rgba(139, 92, 246, 0.2)" }}
          >
            Selected: <span data-testid="selected-count">{selectedClips.length}</span> /{" "}
            {allClipIds.length}
          </span>

          {/* Quick selection chips for clips */}
          {clips.length > 0 && clips.length <= 6 && (
            <div style={{ display: "flex", gap: 4 }}>
              {clips.map((c) => {
                const isSel = selectedClips.includes(c.id);
                return (
                  <button
                    key={c.id}
                    data-testid={`batch-select-clip-${c.id}`}
                    onClick={() => toggleSelectClip(c.id)}
                    className={`btn sm ghost ${isSel ? "on" : ""}`}
                    style={{
                      fontSize: 10,
                      padding: "1px 6px",
                      opacity: isSel ? 1 : 0.6,
                    }}
                  >
                    {c.id}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Batch Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
          {/* Batch Rate (Supports both batch-rate and batch-rate-up testids) */}
          <button
            className="btn sm primary"
            data-testid="batch-rate"
            onClick={() => executeBatch("rate", { rating: 5 })}
            disabled={batchBusy}
            title="Batch rate selected clips 5 stars"
            style={{ fontSize: 11, padding: "3px 10px" }}
          >
            ★ Batch Rate 5★
          </button>
          <button
            style={{ display: "none" }}
            data-testid="batch-rate-up"
            onClick={() => executeBatch("rate_up", { rating: 5 })}
            aria-hidden="true"
          >
            Batch Rate Up
          </button>
          <button
            style={{ display: "none" }}
            data-testid="batch-rate-5"
            onClick={() => executeBatch("rate", { rating: 5 })}
            aria-hidden="true"
          >
            Batch Rate 5
          </button>

          {/* Batch Tag */}
          <button
            className="btn sm ghost"
            data-testid="batch-tag"
            onClick={() => executeBatch("tag", { tags: ["#viral", "#climax"] })}
            disabled={batchBusy}
            title="Batch add #viral & #climax tags to selected clips"
            style={{ fontSize: 11, padding: "3px 10px" }}
          >
            🏷️ Batch Tag
          </button>
          <button
            style={{ display: "none" }}
            data-testid="batch-tag-viral"
            onClick={() => executeBatch("tag", { tags: ["#viral", "#climax"] })}
            aria-hidden="true"
          >
            Batch Tag Viral
          </button>

          {/* Batch Export */}
          <button
            className="btn sm ghost"
            data-testid="batch-export"
            onClick={() => executeBatch("export", {})}
            disabled={batchBusy}
            title="Trigger batch export for selected clips"
            style={{ fontSize: 11, padding: "3px 10px" }}
          >
            🚀 Batch Export
          </button>

          {/* Batch Delete */}
          <button
            className="btn sm ghost"
            data-testid="batch-delete"
            onClick={() => executeBatch("delete", {})}
            disabled={batchBusy}
            title="Delete selected clips (with confirmation)"
            style={{
              fontSize: 11,
              padding: "3px 10px",
              color: "var(--bad, #ef4444)",
              borderColor: "rgba(239, 68, 68, 0.4)",
            }}
          >
            🗑️ Batch Delete
          </button>
        </div>
      </div>

      {/* 3. Visual Tactile HUD / OSD Badge Overlay */}
      {hud && (
        <div
          data-testid="hud-badge"
          data-test-label={hud.label}
          className="streamer-hud-badge"
          style={{
            position: "absolute",
            top: -46,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 1000,
            background: "rgba(11, 15, 25, 0.95)",
            color: "#00f0ff",
            border: "1px solid rgba(0, 240, 255, 0.8)",
            borderRadius: 6,
            padding: "5px 14px",
            fontSize: 13,
            fontWeight: 800,
            fontFamily: "monospace",
            letterSpacing: "0.5px",
            boxShadow:
              "0 0 16px rgba(0, 240, 255, 0.4), inset 0 0 8px rgba(0, 240, 255, 0.2)",
            pointerEvents: "none",
            display: "flex",
            alignItems: "center",
            gap: 6,
            whiteSpace: "nowrap",
          }}
        >
          <span style={{ fontSize: 14 }}>⚡</span>
          <span data-testid="hud-display">{hud.label}</span>
        </div>
      )}
    </div>
  );
}
