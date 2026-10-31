import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import type { Rect, CaptionWord, StyleTemplate } from "../lib/types";

// ============================================================================
// Types & Contracts
// ============================================================================

export type ResizeHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export interface DragState {
  type: "facecam-move" | "facecam-resize" | "crop" | "caption";
  handle?: ResizeHandle;
  startX: number;
  startY: number;
  initialRect?: Rect;
  initialCx?: number;
  initialYFrac?: number;
}

export interface InteractiveCanvasProps {
  /** Optional video or background elements rendered under controls */
  children?: React.ReactNode;
  videoRef?: React.RefObject<HTMLVideoElement>;

  /** Dimensions used for coordinate math in test environments where getBoundingClientRect is 0 */
  canvasWidth?: number;
  canvasHeight?: number;

  /** Facecam bounding box in normalized [0, 1] coordinates */
  facecam?: Rect | null;
  onFacecamChange?: (rect: Rect) => void;
  showFacecam?: boolean;
  aspectLockRatio?: number;

  /** Gameplay 9:16 crop window horizontal center [0, 1] */
  cropCx?: number | null;
  onCropCxChange?: (cx: number | null) => void;
  showCropOverlay?: boolean;
  cropWidthFrac?: number; // Defaults to (81 / 256) ≈ 0.3164 (9:16 inside 16:9)

  /** Live WYSIWYG Captions */
  captionWords?: CaptionWord[];
  onCaptionWordsChange?: (words: CaptionWord[]) => void;
  captionYFrac?: number;
  onCaptionYFracChange?: (yFrac: number) => void;
  showCaptions?: boolean;
  currentTime?: number;
  activeStyle?: StyleTemplate | null;

  /** Platform Safe Zone Guides */
  showSafeGuides?: boolean;

  /** Undo / Redo integration */
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;

  /** Optional container style / class */
  className?: string;
  style?: React.CSSProperties;
}

// ============================================================================
// Pure Coordinate & Snapping Algorithms
// ============================================================================

export function clampRect(r: Rect): Rect {
  const minSize = 0.05;
  const w = Math.max(minSize, Math.min(1.0, r.w));
  const h = Math.max(minSize, Math.min(1.0, r.h));
  const x = Math.max(0.0, Math.min(1.0 - w, r.x));
  const y = Math.max(0.0, Math.min(1.0 - h, r.y));
  return { x, y, w, h };
}

export function applyHandleResize(
  initial: Rect,
  handle: ResizeHandle,
  dx: number,
  dy: number,
  aspectLockRatio?: number
): Rect {
  let { x, y, w, h } = initial;

  if (handle.includes("e")) w += dx;
  if (handle.includes("w")) {
    x += dx;
    w -= dx;
  }
  if (handle.includes("s")) h += dy;
  if (handle.includes("n")) {
    y += dy;
    h -= dy;
  }

  // Handle minimum size constraint while maintaining anchor
  if (w < 0.05) {
    if (handle.includes("w")) x = initial.x + initial.w - 0.05;
    w = 0.05;
  }
  if (h < 0.05) {
    if (handle.includes("n")) y = initial.y + initial.h - 0.05;
    h = 0.05;
  }

  if (aspectLockRatio && aspectLockRatio > 0) {
    if (Math.abs(dx) > Math.abs(dy)) {
      h = w / aspectLockRatio;
    } else {
      w = h * aspectLockRatio;
    }
  }

  return clampRect({ x, y, w, h });
}

export function applyMagneticSnap(
  target: number,
  snapPoints: number[] = [0.5],
  threshold = 0.02
): { value: number; snapped: boolean; snapPoint?: number } {
  let closestDist = Infinity;
  let closestPoint: number | null = null;
  for (const p of snapPoints) {
    const dist = Math.abs(target - p);
    if (dist < closestDist) {
      closestDist = dist;
      closestPoint = p;
    }
  }
  if (closestPoint !== null && closestDist <= threshold + 1e-6) {
    return { value: closestPoint, snapped: true, snapPoint: closestPoint };
  }
  return { value: target, snapped: false };
}

// ============================================================================
// InteractiveCanvas Component
// ============================================================================

export default function InteractiveCanvas({
  children,
  canvasWidth = 800,
  canvasHeight = 450,
  facecam,
  onFacecamChange,
  showFacecam = true,
  aspectLockRatio,
  cropCx,
  onCropCxChange,
  showCropOverlay = false,
  cropWidthFrac = 81 / 256,
  captionWords,
  onCaptionWordsChange,
  captionYFrac = 0.78,
  onCaptionYFracChange,
  showCaptions = true,
  currentTime,
  activeStyle,
  showSafeGuides = false,
  onUndo,
  onRedo,
  className = "",
  style = {},
}: InteractiveCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);

  // Snapping feedback state
  const [cropSnapped, setCropSnapped] = useState(false);
  const [captionSnapped, setCaptionSnapped] = useState<number | null>(null);

  // Inline word editing state
  const [editingWordIndex, setEditingWordIndex] = useState<number | null>(null);
  const [editingText, setEditingText] = useState("");

  // Helper to read current container dimensions
  const getDimensions = useCallback(() => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        return { width: rect.width, height: rect.height };
      }
    }
    return { width: canvasWidth, height: canvasHeight };
  }, [canvasWidth, canvasHeight]);

  // Pointer down starter
  const startDrag = (
    e: React.PointerEvent,
    type: DragState["type"],
    handle?: ResizeHandle
  ) => {
    e.preventDefault();
    e.stopPropagation();

    try {
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      // In synthetic tests or unsupported environments, setPointerCapture is a no-op
    }

    setDragState({
      type,
      handle,
      startX: e.clientX,
      startY: e.clientY,
      initialRect: facecam ? { ...facecam } : undefined,
      initialCx: cropCx ?? 0.5,
      initialYFrac: captionYFrac ?? 0.78,
    });
  };

  // Pointer move handler
  const handlePointerMove = useCallback(
    (e: React.PointerEvent | PointerEvent) => {
      if (!dragState) return;

      const { width, height } = getDimensions();
      const dx = (e.clientX - dragState.startX) / width;
      const dy = (e.clientY - dragState.startY) / height;

      if (dragState.type === "facecam-move" && dragState.initialRect) {
        const next = clampRect({
          ...dragState.initialRect,
          x: dragState.initialRect.x + dx,
          y: dragState.initialRect.y + dy,
        });
        onFacecamChange?.(next);
      } else if (
        dragState.type === "facecam-resize" &&
        dragState.initialRect &&
        dragState.handle
      ) {
        const next = applyHandleResize(
          dragState.initialRect,
          dragState.handle,
          dx,
          dy,
          aspectLockRatio
        );
        onFacecamChange?.(next);
      } else if (dragState.type === "crop" && dragState.initialCx !== undefined) {
        const rawCx = dragState.initialCx + dx;
        const cropHalf = cropWidthFrac / 2;
        const clampedCx = Math.max(cropHalf, Math.min(1 - cropHalf, rawCx));
        const snap = applyMagneticSnap(clampedCx, [0.5], 0.02);
        setCropSnapped(snap.snapped);
        onCropCxChange?.(snap.value);
      } else if (
        dragState.type === "caption" &&
        dragState.initialYFrac !== undefined
      ) {
        const rawY = dragState.initialYFrac + dy;
        const clampedY = Math.max(0.05, Math.min(0.95, rawY));
        const snap = applyMagneticSnap(clampedY, [0.70, 0.78, 0.85], 0.02);
        setCaptionSnapped(snap.snapped ? snap.value : null);
        onCaptionYFracChange?.(snap.value);
      }
    },
    [
      dragState,
      getDimensions,
      onFacecamChange,
      aspectLockRatio,
      cropWidthFrac,
      onCropCxChange,
      onCaptionYFracChange,
    ]
  );

  // Pointer up handler
  const handlePointerUp = useCallback(
    (e: React.PointerEvent | PointerEvent) => {
      if (dragState) {
        try {
          (e.target as HTMLElement)?.releasePointerCapture?.((e as any).pointerId);
        } catch {
          // No-op
        }
        setDragState(null);
      }
    },
    [dragState]
  );

  // Global window listeners while dragging to guarantee smooth capture
  useEffect(() => {
    if (!dragState) return;
    const onMove = (e: PointerEvent) => handlePointerMove(e);
    const onUp = (e: PointerEvent) => handlePointerUp(e);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [dragState, handlePointerMove, handlePointerUp]);

  // Active/visible caption words
  const visibleWords = useMemo(() => {
    if (!captionWords || captionWords.length === 0) return [];
    if (currentTime !== undefined && currentTime > 0) {
      const matchIdx = captionWords.findIndex(
        (w) => currentTime >= w.t && currentTime <= w.t + w.d + 0.4
      );
      if (matchIdx !== -1) {
        const s = Math.max(0, matchIdx - 2);
        const e = Math.min(captionWords.length, matchIdx + 4);
        return captionWords.slice(s, e).map((w, i) => ({ word: w, index: s + i }));
      }
    }
    // Default / stopped preview: first 8 words
    return captionWords.slice(0, 8).map((w, i) => ({ word: w, index: i }));
  }, [captionWords, currentTime]);

  // Commit inline word edit
  const commitWordEdit = useCallback(() => {
    if (editingWordIndex !== null && captionWords) {
      const updated = [...captionWords];
      if (updated[editingWordIndex]) {
        updated[editingWordIndex] = {
          ...updated[editingWordIndex],
          text: editingText,
        };
        onCaptionWordsChange?.(updated);
      }
      setEditingWordIndex(null);
    }
  }, [editingWordIndex, editingText, captionWords, onCaptionWordsChange]);

  // Keyboard shortcut listener for canvas level (undo / redo)
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return; // Do not intercept while actively typing in input
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) {
        onRedo?.();
      } else {
        onUndo?.();
      }
    }
  };

  // Dimensions & layout calculations
  const effectiveCx = cropCx ?? 0.5;
  const cropLeftPct = (effectiveCx - cropWidthFrac / 2) * 100;
  const cropWidthPct = cropWidthFrac * 100;
  const effectiveCaptionY = captionYFrac ?? 0.78;

  const handles: ResizeHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

  return (
    <div
      ref={containerRef}
      data-testid="interactive-canvas"
      className={`interactive-canvas ${className}`}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        userSelect: "none",
        ...style,
      }}
    >
      {/* Background / Video Children */}
      {children}

      {/* ================================================================== */}
      {/* 1. Gameplay 9:16 Crop Window Overlay                               */}
      {/* ================================================================== */}
      {showCropOverlay && (
        <div data-testid="crop-overlay-container" style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
          {/* Left Scrim */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              bottom: 0,
              width: `${Math.max(0, cropLeftPct)}%`,
              background: "rgba(7, 9, 14, 0.65)",
              backdropFilter: "blur(2px)",
            }}
          />

          {/* Right Scrim */}
          <div
            style={{
              position: "absolute",
              left: `${cropLeftPct + cropWidthPct}%`,
              top: 0,
              bottom: 0,
              right: 0,
              background: "rgba(7, 9, 14, 0.65)",
              backdropFilter: "blur(2px)",
            }}
          />

          {/* 9:16 Viewport Box */}
          <div
            data-testid="crop-window"
            data-cx={effectiveCx}
            style={{
              position: "absolute",
              left: `${cropLeftPct}%`,
              width: `${cropWidthPct}%`,
              top: 0,
              bottom: 0,
              border: "2px solid var(--accent-cyan, #06b6d4)",
              boxShadow: "0 0 15px rgba(6, 182, 212, 0.35)",
              pointerEvents: "auto",
            }}
          >
            {/* Rule of Thirds Guides */}
            <div
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage:
                  "linear-gradient(to right, rgba(6, 182, 212, 0.15) 1px, transparent 1px), linear-gradient(to bottom, rgba(6, 182, 212, 0.15) 1px, transparent 1px)",
                backgroundSize: "33.333% 33.333%",
                pointerEvents: "none",
              }}
            />

            {/* Horizontal Drag Handle */}
            <div
              data-testid="crop-drag-handle"
              onPointerDown={(e) => startDrag(e, "crop")}
              style={{
                position: "absolute",
                top: 8,
                left: "50%",
                transform: "translateX(-50%)",
                background: "rgba(6, 182, 212, 0.9)",
                color: "#07090e",
                padding: "3px 10px",
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 800,
                letterSpacing: "0.5px",
                cursor: "ew-resize",
                display: "flex",
                alignItems: "center",
                gap: 6,
                boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
                whiteSpace: "nowrap",
                zIndex: 10,
              }}
              title="Drag horizontally to reposition 9:16 crop window"
            >
              <span>↔</span> 9:16 CROP {(effectiveCx * 100).toFixed(0)}%
            </div>

            {/* Magnetic Center Snap Indicator */}
            {cropSnapped && (
              <div
                data-testid="crop-snap-indicator"
                style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  bottom: 0,
                  width: 2,
                  background: "#06b6d4",
                  boxShadow: "0 0 10px #06b6d4",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span
                  style={{
                    background: "#06b6d4",
                    color: "#07090e",
                    fontSize: 9,
                    fontWeight: 900,
                    padding: "2px 6px",
                    borderRadius: 3,
                    whiteSpace: "nowrap",
                  }}
                >
                  CENTER 50%
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ================================================================== */}
      {/* 2. Interactive Facecam Bounding Box with 8-Point Handles          */}
      {/* ================================================================== */}
      {showFacecam && facecam && (
        <div
          data-testid="facecam-box"
          style={{
            position: "absolute",
            left: `${facecam.x * 100}%`,
            top: `${facecam.y * 100}%`,
            width: `${facecam.w * 100}%`,
            height: `${facecam.h * 100}%`,
            border: "2px solid var(--accent, #8b5cf6)",
            background: "rgba(139, 92, 246, 0.15)",
            boxShadow: "0 0 12px rgba(139, 92, 246, 0.4)",
            zIndex: 15,
            boxSizing: "border-box",
          }}
        >
          {/* Draggable Interior Area */}
          <div
            data-testid="facecam-drag-interior"
            onPointerDown={(e) => startDrag(e, "facecam-move")}
            style={{
              width: "100%",
              height: "100%",
              cursor: "move",
              display: "flex",
              alignItems: "flex-start",
              justifyContent: "flex-start",
              padding: 4,
            }}
            title="Drag facecam box to reposition"
          >
            <span
              style={{
                background: "rgba(139, 92, 246, 0.9)",
                color: "#fff",
                fontSize: 10,
                fontWeight: 800,
                padding: "1px 5px",
                borderRadius: 3,
                pointerEvents: "none",
                letterSpacing: "0.5px",
              }}
            >
              FACECAM
            </span>
          </div>

          {/* 8-Point Resize Handles */}
          {handles.map((h) => {
            const getCursor = () => {
              if (h === "nw" || h === "se") return "nwse-resize";
              if (h === "ne" || h === "sw") return "nesw-resize";
              if (h === "n" || h === "s") return "ns-resize";
              return "ew-resize";
            };

            const getPosition = (): React.CSSProperties => {
              const half = -5;
              const center = "calc(50% - 5px)";
              switch (h) {
                case "nw":
                  return { top: half, left: half };
                case "n":
                  return { top: half, left: center };
                case "ne":
                  return { top: half, right: half };
                case "e":
                  return { top: center, right: half };
                case "se":
                  return { bottom: half, right: half };
                case "s":
                  return { bottom: half, left: center };
                case "sw":
                  return { bottom: half, left: half };
                case "w":
                  return { top: center, left: half };
              }
            };

            return (
              <div
                key={h}
                data-testid={`facecam-handle-${h}`}
                onPointerDown={(e) => startDrag(e, "facecam-resize", h)}
                style={{
                  position: "absolute",
                  width: 10,
                  height: 10,
                  background: "#ffffff",
                  border: "2px solid var(--accent, #8b5cf6)",
                  borderRadius: 2,
                  boxShadow: "0 0 6px rgba(0, 0, 0, 0.8)",
                  cursor: getCursor(),
                  zIndex: 20,
                  ...getPosition(),
                }}
                title={`Resize ${h.toUpperCase()}`}
              />
            );
          })}
        </div>
      )}

      {/* ================================================================== */}
      {/* 3. Live WYSIWYG Captions Overlay with Vertical Drag & Snapping     */}
      {/* ================================================================== */}
      {showCaptions && (
        <div
          data-testid="caption-overlay"
          style={{
            position: "absolute",
            left: "5%",
            right: "5%",
            top: `${effectiveCaptionY * 100}%`,
            transform: "translateY(-50%)",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            zIndex: 18,
            pointerEvents: "none",
          }}
        >
          {/* Vertical Position Drag Bar */}
          <div
            data-testid="caption-drag-handle"
            onPointerDown={(e) => startDrag(e, "caption")}
            style={{
              cursor: "ns-resize",
              pointerEvents: "auto",
              background: "rgba(17, 22, 36, 0.85)",
              border: "1px solid rgba(236, 72, 153, 0.6)",
              borderRadius: 4,
              padding: "2px 10px",
              marginBottom: 6,
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 10,
              fontWeight: 700,
              color: "var(--accent-2, #ec4899)",
              boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
            }}
            title="Drag vertically to reposition captions"
          >
            <span>↕</span> CAPTIONS {(effectiveCaptionY * 100).toFixed(0)}%
          </div>

          {/* Safe Zone Snap Guide Badge */}
          {captionSnapped !== null && (
            <div
              data-testid="caption-snap-guide"
              style={{
                position: "absolute",
                top: -24,
                background: "var(--accent-2, #ec4899)",
                color: "#ffffff",
                fontSize: 9,
                fontWeight: 900,
                padding: "2px 8px",
                borderRadius: 4,
                boxShadow: "0 0 8px rgba(236, 72, 153, 0.6)",
                whiteSpace: "nowrap",
              }}
            >
              SAFE ZONE: {(captionSnapped * 100).toFixed(0)}%
            </div>
          )}

          {/* WYSIWYG Rendered Caption Words */}
          <div
            data-testid="caption-words-container"
            style={{
              display: "flex",
              flexWrap: "wrap",
              justifyContent: "center",
              gap: 6,
              padding: "6px 12px",
              borderRadius: 6,
              background: "rgba(7, 9, 14, 0.4)",
              backdropFilter: "blur(4px)",
              pointerEvents: "auto",
              fontFamily: activeStyle?.font || "sans-serif",
              fontSize: activeStyle?.font_size ? `${Math.max(14, activeStyle.font_size / 2)}px` : "16px",
              fontWeight: 800,
              textTransform: activeStyle?.uppercase ? "uppercase" : "none",
            }}
          >
            {visibleWords.length === 0 ? (
              <span className="tiny muted" style={{ color: "#aaa" }}>
                No captions at this timestamp
              </span>
            ) : (
              visibleWords.map(({ word, index }) => {
                const isSpoken =
                  currentTime !== undefined
                    ? currentTime >= word.t && currentTime < word.t + word.d
                    : false;

                const isEditing = editingWordIndex === index;

                return (
                  <span
                    key={index}
                    data-testid={`caption-word-item-${index}`}
                    style={{ display: "inline-flex", alignItems: "center" }}
                  >
                    {isEditing ? (
                      <input
                        data-testid={`caption-word-input-${index}`}
                        className="caption-word-input"
                        value={editingText}
                        autoFocus
                        onChange={(e) => setEditingText(e.target.value)}
                        onBlur={commitWordEdit}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            commitWordEdit();
                          } else if (e.key === "Escape") {
                            e.preventDefault();
                            setEditingWordIndex(null);
                          } else if (e.key === "Tab") {
                            e.preventDefault();
                            commitWordEdit();
                            const next = e.shiftKey ? index - 1 : index + 1;
                            if (captionWords && next >= 0 && next < captionWords.length) {
                              setEditingWordIndex(next);
                              setEditingText(captionWords[next].text);
                            }
                          }
                        }}
                        style={{
                          background: "rgba(0, 0, 0, 0.8)",
                          color: activeStyle ? `#${activeStyle.highlight || activeStyle.primary}` : "#FFCC00",
                          border: "1px solid var(--accent, #8b5cf6)",
                          borderRadius: 4,
                          padding: "2px 6px",
                          font: "inherit",
                          fontWeight: "inherit",
                          textTransform: "inherit",
                          outline: "none",
                          boxShadow: "0 0 8px rgba(139, 92, 246, 0.7)",
                        }}
                      />
                    ) : (
                      <span
                        data-testid={`caption-word-${index}`}
                        data-word-index={index}
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingWordIndex(index);
                          setEditingText(word.text);
                        }}
                        style={{
                          cursor: "pointer",
                          color: isSpoken
                            ? activeStyle ? `#${activeStyle.highlight}` : "#FFCC00"
                            : activeStyle ? `#${activeStyle.primary}` : "#FFFFFF",
                          textShadow: "0 2px 4px rgba(0,0,0,0.95)",
                          WebkitTextStroke: activeStyle?.outline_w
                            ? `${activeStyle.outline_w}px #${activeStyle.outline}`
                            : "1px #000",
                          padding: "2px 4px",
                          borderRadius: 3,
                          transition: "transform 0.1s ease",
                          transform: isSpoken ? "scale(1.08)" : "none",
                        }}
                        title="Click to edit word on canvas"
                      >
                        {word.text}
                      </span>
                    )}
                  </span>
                );
              })
            )}
          </div>
        </div>
      )}

      {/* ================================================================== */}
      {/* 4. Safe Zone Guidelines Overlay                                    */}
      {/* ================================================================== */}
      {showSafeGuides && (
        <div
          data-testid="safe-zone-overlay"
          style={{
            position: "absolute",
            inset: "5%",
            border: "2px dashed rgba(236, 72, 153, 0.6)",
            pointerEvents: "none",
            zIndex: 25,
            borderRadius: 8,
          }}
        >
          <span
            style={{
              position: "absolute",
              top: 8,
              right: 8,
              background: "rgba(236, 72, 153, 0.85)",
              color: "#fff",
              fontSize: 9,
              fontWeight: 800,
              padding: "2px 6px",
              borderRadius: 4,
              textTransform: "uppercase",
            }}
          >
            Safe Zone
          </span>
        </div>
      )}
    </div>
  );
}
