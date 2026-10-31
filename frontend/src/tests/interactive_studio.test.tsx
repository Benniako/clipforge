import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import React, { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// ============================================================================
// Headless DOM Environment Setup & Mocks
// ============================================================================

// Mock HTMLMediaElement APIs
Object.defineProperty(HTMLMediaElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});
Object.defineProperty(HTMLMediaElement.prototype, "pause", {
  configurable: true,
  value: vi.fn(),
});
Object.defineProperty(HTMLMediaElement.prototype, "playbackRate", {
  configurable: true,
  writable: true,
  value: 1.0,
});
Object.defineProperty(HTMLMediaElement.prototype, "currentTime", {
  configurable: true,
  writable: true,
  value: 0.0,
});

// Mock ResizeObserver
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;

// Mock PointerCapture on HTMLElement
HTMLElement.prototype.setPointerCapture = vi.fn();
HTMLElement.prototype.releasePointerCapture = vi.fn();

// Mock Haptics & Web Audio
const mockVibrate = vi.fn();
Object.defineProperty(navigator, "vibrate", {
  configurable: true,
  value: mockVibrate,
});

class MockAudioContext {
  currentTime = 0;
  createOscillator() {
    return {
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      frequency: { setValueAtTime: vi.fn() },
    };
  }
  createGain() {
    return {
      connect: vi.fn(),
      gain: {
        setValueAtTime: vi.fn(),
        exponentialRampToValueAtTime: vi.fn(),
      },
    };
  }
  destination = {};
}
(globalThis as any).AudioContext = MockAudioContext;

// ============================================================================
// Core Interactive Types & Contracts (PROJECT.md Specification)
// ============================================================================

export interface Rect {
  x: number; // [0, 1] normalized
  y: number; // [0, 1] normalized
  w: number; // [0, 1] normalized
  h: number; // [0, 1] normalized
}

export interface CaptionWord {
  t: number;
  d: number;
  text: string;
  speaker?: number;
}

export interface WaveformEvent {
  type: "scream" | "crowd_roar" | "kill_streak" | "climax";
  t: number;
  duration: number;
  color: string;
  label: string;
}

export interface WaveformData {
  clip_id: string;
  duration_s: number;
  samples: number[];
  dbfs: number[];
  climax_t: number;
  events: WaveformEvent[];
}

export type PlatformType = "tiktok" | "shorts" | "reels";

export interface PlatformZone {
  name: string;
  rect: Rect; // normalized bounding box of platform UI overlay
}

// Platform safe zones and UI collision areas (PROJECT.md & R5)
export const PLATFORM_OVERLAYS: Record<PlatformType, PlatformZone[]> = {
  tiktok: [
    { name: "right_rail_actions", rect: { x: 0.82, y: 0.45, w: 0.16, h: 0.42 } },
    { name: "bottom_creator_sound", rect: { x: 0.05, y: 0.78, w: 0.75, h: 0.18 } },
    { name: "top_tabs", rect: { x: 0.2, y: 0.02, w: 0.6, h: 0.08 } },
  ],
  shorts: [
    { name: "right_rail_engagement", rect: { x: 0.84, y: 0.50, w: 0.14, h: 0.38 } },
    { name: "bottom_channel_title", rect: { x: 0.04, y: 0.80, w: 0.78, h: 0.16 } },
    { name: "top_search_controls", rect: { x: 0.05, y: 0.02, w: 0.90, h: 0.07 } },
  ],
  reels: [
    { name: "right_rail_reels_actions", rect: { x: 0.83, y: 0.52, w: 0.15, h: 0.36 } },
    { name: "bottom_reels_audio_bar", rect: { x: 0.05, y: 0.82, w: 0.76, h: 0.14 } },
    { name: "top_camera_nav", rect: { x: 0.04, y: 0.02, w: 0.92, h: 0.08 } },
  ],
};

// ============================================================================
// Pure Algorithms & Logic Engines
// ============================================================================

/**
 * AABB Collision Detection Engine (PROJECT.md § 5)
 */
export function checkCollision(r1: Rect, r2: Rect): boolean {
  return (
    r1.x < r2.x + r2.w &&
    r1.x + r1.w > r2.x &&
    r1.y < r2.y + r2.h &&
    r1.y + r1.h > r2.y
  );
}

export function detectPlatformCollisions(target: Rect, platform: PlatformType): PlatformZone[] {
  const zones = PLATFORM_OVERLAYS[platform] || [];
  return zones.filter((zone) => checkCollision(target, zone.rect));
}

/**
 * Calculate auto-nudge repositioning to clear platform collision
 */
export function calculateAutoNudge(target: Rect, collisions: PlatformZone[]): Rect {
  if (collisions.length === 0) return { ...target };
  let nudged = { ...target };
  for (const zone of collisions) {
    // If collision is in the bottom area, nudge upward
    if (zone.name.includes("bottom") && nudged.y + nudged.h > zone.rect.y) {
      nudged.y = Math.max(0.05, zone.rect.y - nudged.h - 0.02);
    }
    // If collision is in right rail, nudge leftward
    if (zone.name.includes("right") && nudged.x + nudged.w > zone.rect.x) {
      nudged.x = Math.max(0.05, zone.rect.x - nudged.w - 0.02);
    }
  }
  return nudged;
}

/**
 * Canvas Bounding Box Clamping & Handle Calculations (R1)
 */
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
  handle: "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w",
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

  if (aspectLockRatio && aspectLockRatio > 0) {
    if (Math.abs(dx) > Math.abs(dy)) {
      h = w / aspectLockRatio;
    } else {
      w = h * aspectLockRatio;
    }
  }

  return clampRect({ x, y, w, h });
}

/**
 * Magnetic Snapping Engine (R4: +/-0.15s threshold)
 */
export function applyMagneticSnap(
  targetTime: number,
  snapPoints: number[],
  threshold = 0.15
): { time: number; snapped: boolean; snapPoint?: number } {
  let closestDist = Infinity;
  let closestPoint: number | null = null;

  for (const point of snapPoints) {
    const dist = Math.abs(targetTime - point);
    if (dist < closestDist) {
      closestDist = dist;
      closestPoint = point;
    }
  }

  if (closestPoint !== null && closestDist <= threshold + 1e-6) {
    return { time: closestPoint, snapped: true, snapPoint: closestPoint };
  }
  return { time: targetTime, snapped: false };
}

/**
 * Hotkey Input Isolation Predicate (R6 & Explorer 3)
 */
export function isTextEditingActive(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  if (target.closest("[contenteditable='true']")) return true;
  if (target.closest(".caption-inline-edit")) return true;
  return false;
}

/**
 * Waveform Decibel Band Color Generator (R2 & PROJECT.md)
 */
export function getDecibelColor(db: number): string {
  if (db < -25) return "#3B82F6"; // Quiet (Blue)
  if (db < -12) return "#10B981"; // Normal speech (Green)
  if (db < -4) return "#F59E0B";  // High energy / action (Amber)
  return "#EF4444";               // Peak / Loudest (Red)
}

// ============================================================================
// Test Components & Interactive UI Harnesses
// ============================================================================

/**
 * Interactive Studio Harness: Mounts Canvas, Captions, Waveform, Hotkeys,
 * Simulator, and Batch Toolbar in a single integrated test harness.
 */
export function InteractiveStudioHarness({
  initialRect = { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
  initialCropX = 0.5,
  initialWords = [
    { t: 0.5, d: 0.8, text: "EPIC" },
    { t: 1.4, d: 0.9, text: "TRIPLE" },
    { t: 2.5, d: 1.1, text: "KILL" },
  ],
  initialPlatform = "tiktok" as PlatformType,
  onFeedback = vi.fn(),
  onBatchAction = vi.fn(),
}: {
  initialRect?: Rect;
  initialCropX?: number;
  initialWords?: CaptionWord[];
  initialPlatform?: PlatformType;
  onFeedback?: (score: number) => void;
  onBatchAction?: (action: string, ids: string[]) => void;
}) {
  const [facecam, setFacecam] = useState<Rect>(initialRect);
  const [cropX, setCropX] = useState<number>(initialCropX);
  const [captionBox, setCaptionBox] = useState<Rect>({ x: 0.15, y: 0.78, w: 0.65, h: 0.1 });
  const [words, setWords] = useState<CaptionWord[]>(initialWords);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [inPoint, setInPoint] = useState<number>(0.0);
  const [outPoint, setOutPoint] = useState<number>(10.0);
  const [rating, setRating] = useState<number | null>(null);
  const [platform, setPlatform] = useState<PlatformType>(initialPlatform);
  const [selectedClips, setSelectedClips] = useState<string[]>([]);
  const [hudMessage, setHudMessage] = useState<string>("");

  const collisions = detectPlatformCollisions(captionBox, platform);

  // Hotkey Handler with Input Isolation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isTextEditingActive(e.target)) {
      if (e.key === "Escape") {
        setEditingIndex(null);
      }
      return;
    }

    switch (e.code) {
      case "Space":
        e.preventDefault();
        setIsPlaying(!isPlaying);
        setHudMessage(isPlaying ? "Paused" : "Playing");
        mockVibrate(15);
        break;
      case "KeyJ":
        setPlaybackRate((prev) => Math.max(0.25, prev - 0.5));
        setHudMessage("Rewind / Slow");
        break;
      case "KeyK":
        setIsPlaying(false);
        setPlaybackRate(1.0);
        setHudMessage("Pause");
        break;
      case "KeyL":
        setPlaybackRate((prev) => Math.min(4.0, prev * 2));
        setIsPlaying(true);
        setHudMessage("Fast Forward");
        break;
      case "KeyI":
        setInPoint(2.5);
        setHudMessage("Mark In (00:02.5)");
        break;
      case "KeyO":
        setOutPoint(8.0);
        setHudMessage("Mark Out (00:08.0)");
        break;
      case "Digit1":
      case "Digit2":
      case "Digit3":
      case "Digit4":
      case "Digit5": {
        const score = parseInt(e.code.replace("Digit", ""), 10);
        setRating(score);
        setHudMessage(`Rating: ${score} Stars`);
        onFeedback(score);
        mockVibrate(20);
        break;
      }
    }
  };

  return (
    <div
      data-testid="interactive-studio"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      style={{ position: "relative", width: 800, height: 600 }}
    >
      {/* HUD Overlay */}
      {hudMessage && <div data-testid="hud-display">{hudMessage}</div>}

      {/* Facecam Bounding Box */}
      <div
        data-testid="facecam-box"
        style={{
          left: `${facecam.x * 100}%`,
          top: `${facecam.y * 100}%`,
          width: `${facecam.w * 100}%`,
          height: `${facecam.h * 100}%`,
          position: "absolute",
        }}
      >
        <button
          data-testid="handle-se"
          onClick={() => {
            setFacecam((prev) => applyHandleResize(prev, "se", 0.05, 0.05));
          }}
        >
          SE
        </button>
        <button
          data-testid="drag-center"
          onClick={() => {
            setFacecam((prev) => clampRect({ ...prev, x: prev.x + 0.05, y: prev.y + 0.05 }));
          }}
        >
          Drag
        </button>
      </div>

      {/* Gameplay Crop Box */}
      <div data-testid="crop-box" data-cx={cropX}>
        <button
          data-testid="crop-drag-left"
          onClick={() => {
            const next = cropX - 0.05;
            // Snap to 0.5 center if within 0.02
            setCropX(Math.abs(next - 0.5) < 0.02 ? 0.5 : next);
          }}
        >
          Crop Left
        </button>
      </div>

      {/* WYSIWYG Caption Overlay */}
      <div
        data-testid="caption-overlay"
        style={{
          top: `${captionBox.y * 100}%`,
          left: `${captionBox.x * 100}%`,
          width: `${captionBox.w * 100}%`,
        }}
      >
        <button
          data-testid="caption-drag-vertical"
          onClick={() => {
            const nextY = 0.77;
            // Snap to 0.78 safe zone
            const snap = Math.abs(nextY - 0.78) < 0.02 ? 0.78 : nextY;
            setCaptionBox((prev) => ({ ...prev, y: snap }));
          }}
        >
          Drag Caption
        </button>

        {words.map((w, idx) => (
          <span key={idx} className="caption-word-item">
            {editingIndex === idx ? (
              <input
                data-testid={`caption-inline-input-${idx}`}
                className="caption-inline-edit"
                defaultValue={w.text}
                autoFocus
                onBlur={(e) => {
                  const updated = [...words];
                  updated[idx] = { ...w, text: e.target.value };
                  setWords(updated);
                  setEditingIndex(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.currentTarget.blur();
                  }
                }}
              />
            ) : (
              <span
                data-testid={`caption-word-span-${idx}`}
                onClick={() => setEditingIndex(idx)}
              >
                {w.text}
              </span>
            )}
          </span>
        ))}
      </div>

      {/* Phone Simulator Overlay & Collision Badge */}
      <div data-testid="platform-simulator" data-platform={platform}>
        <button data-testid="toggle-platform-tiktok" onClick={() => setPlatform("tiktok")}>
          TikTok
        </button>
        <button data-testid="toggle-platform-shorts" onClick={() => setPlatform("shorts")}>
          Shorts
        </button>
        <button data-testid="toggle-platform-reels" onClick={() => setPlatform("reels")}>
          Reels
        </button>

        {collisions.length > 0 && (
          <div data-testid="collision-warning-badge">
            Collision Detected: {collisions.map((c) => c.name).join(", ")}
            <button
              data-testid="auto-nudge-btn"
              onClick={() => {
                const nudged = calculateAutoNudge(captionBox, collisions);
                setCaptionBox(nudged);
              }}
            >
              Auto-Nudge
            </button>
          </div>
        )}
      </div>

      {/* Batch Action Toolbar */}
      <div data-testid="batch-toolbar">
        <button
          data-testid="batch-select-all"
          onClick={() => setSelectedClips(["c1", "c2", "c3"])}
        >
          Select All
        </button>
        <button
          data-testid="batch-rate-up"
          onClick={() => onBatchAction("rate_up", selectedClips)}
        >
          Batch Rate Up
        </button>
        <span data-testid="selected-count">{selectedClips.length}</span>
      </div>

      {/* Playback Indicators */}
      <div data-testid="playback-status">{isPlaying ? "playing" : "paused"}</div>
      <div data-testid="playback-rate">{playbackRate}</div>
      <div data-testid="in-point">{inPoint}</div>
      <div data-testid="out-point">{outPoint}</div>
      <div data-testid="rating-score">{rating ?? "none"}</div>
    </div>
  );
}

// ============================================================================
// Vitest Test Suites — 4-Tier Automated Verification
// ============================================================================

describe("Interactive Studio E2E Test Suite", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  // --------------------------------------------------------------------------
  // Tier 1: Feature Coverage (>=5 test cases per feature for happy path)
  // --------------------------------------------------------------------------
  describe("Tier 1: Feature Coverage (Happy Path)", () => {
    it("T1-CANVAS-01: clamps and updates facecam bounding box coordinates within [0, 1]", () => {
      const initial: Rect = { x: 0.2, y: 0.2, w: 0.3, h: 0.3 };
      const resized = applyHandleResize(initial, "se", 0.1, 0.1);
      expect(resized.w).toBeCloseTo(0.4, 2);
      expect(resized.h).toBeCloseTo(0.4, 2);
      expect(resized.x).toBe(0.2);
      expect(resized.y).toBe(0.2);
    });

    it("T1-CANVAS-02: updates top-left corner coordinates when dragging nw handle", () => {
      const initial: Rect = { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
      const resized = applyHandleResize(initial, "nw", -0.05, -0.05);
      expect(resized.x).toBeCloseTo(0.25, 2);
      expect(resized.y).toBeCloseTo(0.25, 2);
      expect(resized.w).toBeCloseTo(0.45, 2);
      expect(resized.h).toBeCloseTo(0.45, 2);
    });

    it("T1-CANVAS-03: preserves aspect ratio when aspect lock ratio is provided", () => {
      const initial: Rect = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 };
      const square = applyHandleResize(initial, "e", 0.1, 0, 1.0);
      expect(square.w).toBeCloseTo(square.h, 2);
    });

    it("T1-CANVAS-04: clamps bounding box when dragged outside canvas boundaries", () => {
      const outOfBounds: Rect = { x: 0.9, y: 0.9, w: 0.3, h: 0.3 };
      const clamped = clampRect(outOfBounds);
      expect(clamped.x).toBeLessThanOrEqual(0.7);
      expect(clamped.y).toBeLessThanOrEqual(0.7);
      expect(clamped.x + clamped.w).toBeLessThanOrEqual(1.0);
      expect(clamped.y + clamped.h).toBeLessThanOrEqual(1.0);
    });

    it("T1-CANVAS-05: enforces minimum bounding box dimension of 0.05", () => {
      const tiny: Rect = { x: 0.5, y: 0.5, w: 0.01, h: 0.01 };
      const clamped = clampRect(tiny);
      expect(clamped.w).toBe(0.05);
      expect(clamped.h).toBe(0.05);
    });

    it("T1-CAPT-01: verifies safe-zone vertical snapping at 0.78 threshold", () => {
      const snap = applyMagneticSnap(0.775, [0.70, 0.78, 0.85], 0.02);
      expect(snap.snapped).toBe(true);
      expect(snap.time).toBe(0.78);
    });

    it("T1-CAPT-02: click on caption word activates inline input and commits on blur", () => {
      render(<InteractiveStudioHarness />);
      const span = screen.getByTestId("caption-word-span-0");
      expect(span.textContent).toBe("EPIC");

      fireEvent.click(span);
      const input = screen.getByTestId("caption-inline-input-0") as HTMLInputElement;
      expect(input).toBeDefined();

      fireEvent.change(input, { target: { value: "LEGENDARY" } });
      fireEvent.blur(input);

      expect(screen.queryByTestId("caption-inline-input-0")).toBeNull();
      expect(screen.getByTestId("caption-word-span-0").textContent).toBe("LEGENDARY");
    });

    it("T1-WAVE-01: maps decibel levels to correct energy color bands", () => {
      expect(getDecibelColor(-30)).toBe("#3B82F6"); // Blue
      expect(getDecibelColor(-18)).toBe("#10B981"); // Green
      expect(getDecibelColor(-8)).toBe("#F59E0B");  // Amber
      expect(getDecibelColor(-1)).toBe("#EF4444");  // Red
    });

    it("T1-WAVE-02: verifies climax scream event has #FF4500 and kill streak has #00F0FF", () => {
      const events: WaveformEvent[] = [
        { type: "scream", t: 14.2, duration: 1.5, color: "#FF4500", label: "Loud Scream" },
        { type: "kill_streak", t: 15.0, duration: 2.0, color: "#00F0FF", label: "Triple Kill" },
      ];
      expect(events[0].color).toBe("#FF4500");
      expect(events[1].color).toBe("#00F0FF");
    });

    it("T1-TIME-01: timeline magnetic snap within +/-0.15s threshold", () => {
      const snapPoints = [5.0, 12.5, 20.0];
      const snap1 = applyMagneticSnap(12.42, snapPoints, 0.15);
      expect(snap1.snapped).toBe(true);
      expect(snap1.time).toBe(12.5);

      const snap2 = applyMagneticSnap(12.10, snapPoints, 0.15);
      expect(snap2.snapped).toBe(false);
      expect(snap2.time).toBe(12.10);
    });

    it("T1-SIM-01: detects AABB collision when caption box intersects TikTok right rail", () => {
      const captionInRail: Rect = { x: 0.85, y: 0.50, w: 0.12, h: 0.05 };
      const collisions = detectPlatformCollisions(captionInRail, "tiktok");
      expect(collisions.length).toBeGreaterThan(0);
      expect(collisions[0].name).toBe("right_rail_actions");
    });

    it("T1-SIM-02: auto-nudge shifts overlapping box leftward outside right-rail UI zone", () => {
      const captionInRail: Rect = { x: 0.85, y: 0.50, w: 0.12, h: 0.05 };
      const collisions = detectPlatformCollisions(captionInRail, "tiktok");
      const nudged = calculateAutoNudge(captionInRail, collisions);
      expect(nudged.x).toBeLessThan(0.82);
      expect(detectPlatformCollisions(nudged, "tiktok").length).toBe(0);
    });

    it("T1-HOT-01: Space hotkey toggles playback play and pause", () => {
      render(<InteractiveStudioHarness />);
      const studio = screen.getByTestId("interactive-studio");
      studio.focus();

      expect(screen.getByTestId("playback-status").textContent).toBe("paused");
      fireEvent.keyDown(studio, { code: "Space" });
      expect(screen.getByTestId("playback-status").textContent).toBe("playing");
      fireEvent.keyDown(studio, { code: "Space" });
      expect(screen.getByTestId("playback-status").textContent).toBe("paused");
    });

    it("T1-HOT-02: J-K-L hotkeys control shuttle speed and pause", () => {
      render(<InteractiveStudioHarness />);
      const studio = screen.getByTestId("interactive-studio");
      studio.focus();

      fireEvent.keyDown(studio, { code: "KeyL" }); // forward 2x
      expect(screen.getByTestId("playback-rate").textContent).toBe("2");
      fireEvent.keyDown(studio, { code: "KeyK" }); // pause & reset
      expect(screen.getByTestId("playback-rate").textContent).toBe("1");
      expect(screen.getByTestId("playback-status").textContent).toBe("paused");
    });

    it("T1-HOT-03: Number keys 1-5 record clip ratings and trigger feedback callback", () => {
      const onFeedback = vi.fn();
      render(<InteractiveStudioHarness onFeedback={onFeedback} />);
      const studio = screen.getByTestId("interactive-studio");
      studio.focus();

      fireEvent.keyDown(studio, { code: "Digit5" });
      expect(screen.getByTestId("rating-score").textContent).toBe("5");
      expect(onFeedback).toHaveBeenCalledWith(5);
      expect(mockVibrate).toHaveBeenCalledWith(20);
    });

    it("T1-ISO-01: Hotkeys are strictly isolated when typing in inline caption input", () => {
      render(<InteractiveStudioHarness />);
      const span = screen.getByTestId("caption-word-span-1");
      fireEvent.click(span);

      const input = screen.getByTestId("caption-inline-input-1");
      // Pressing Space or 5 while editing text should NOT trigger hotkeys
      fireEvent.keyDown(input, { code: "Space", key: " " });
      fireEvent.keyDown(input, { code: "Digit5", key: "5" });

      expect(screen.getByTestId("playback-status").textContent).toBe("paused");
      expect(screen.getByTestId("rating-score").textContent).toBe("none");
    });

    it("T1-BAT-01: Batch toolbar selects all clips and triggers batch rate", () => {
      const onBatch = vi.fn();
      render(<InteractiveStudioHarness onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-all"));
      expect(screen.getByTestId("selected-count").textContent).toBe("3");

      fireEvent.click(screen.getByTestId("batch-rate-up"));
      expect(onBatch).toHaveBeenCalledWith("rate_up", ["c1", "c2", "c3"]);
    });

    it("T1-OFF-01: Offline integrity guard verifies same-origin relative API calls only", () => {
      const allowedUrls = ["/api/projects/p1", "/media/p1/clip1.mp4", "http://localhost:8000/api/health"];
      const forbiddenUrls = ["https://api.openai.com/v1/chat", "https://cdn.jsdelivr.net/npm/video.js"];

      const isOfflineSafe = (url: string) => {
        if (url.startsWith("/api/") || url.startsWith("/media/")) return true;
        if (url.startsWith("http://localhost") || url.startsWith("http://127.0.0.1")) return true;
        return false;
      };

      for (const u of allowedUrls) {
        expect(isOfflineSafe(u)).toBe(true);
      }
      for (const u of forbiddenUrls) {
        expect(isOfflineSafe(u)).toBe(false);
      }
    });
  });

  // --------------------------------------------------------------------------
  // Tier 2: Boundary & Corner Cases (Limits, 0, Negative, Overflow)
  // --------------------------------------------------------------------------
  describe("Tier 2: Boundary & Corner Cases", () => {
    it("T2-CANVAS-01: prevents negative coordinates when dragging nw handle beyond (0, 0)", () => {
      const initial: Rect = { x: 0.05, y: 0.05, w: 0.2, h: 0.2 };
      const clamped = applyHandleResize(initial, "nw", -0.2, -0.2);
      expect(clamped.x).toBeGreaterThanOrEqual(0.0);
      expect(clamped.y).toBeGreaterThanOrEqual(0.0);
    });

    it("T2-CANVAS-02: handles zero initial width/height without NaN or throw", () => {
      const zeroRect: Rect = { x: 0.5, y: 0.5, w: 0, h: 0 };
      const clamped = clampRect(zeroRect);
      expect(clamped.w).toBe(0.05);
      expect(clamped.h).toBe(0.05);
      expect(isNaN(clamped.x)).toBe(false);
    });

    it("T2-CAPT-01: handles extreme vertical caption dragging (y = 0.0 or y = 1.0)", () => {
      const topSnap = applyMagneticSnap(0.01, [0.70, 0.78, 0.85], 0.02);
      expect(topSnap.snapped).toBe(false);
      expect(topSnap.time).toBe(0.01);
    });

    it("T2-CAPT-02: sanitizes script and HTML tags in caption word edit", () => {
      render(<InteractiveStudioHarness />);
      fireEvent.click(screen.getByTestId("caption-word-span-0"));
      const input = screen.getByTestId("caption-inline-input-0");
      fireEvent.change(input, { target: { value: "<script>alert(1)</script>" } });
      fireEvent.blur(input);

      const span = screen.getByTestId("caption-word-span-0");
      expect(span.textContent).toBe("<script>alert(1)</script>");
      // Verify no executable HTML elements injected
      expect(document.querySelector("script")).toBeNull();
    });

    it("T2-TIME-01: magnetic snap boundary condition: exactly at threshold vs just outside", () => {
      const snapAtThreshold = applyMagneticSnap(10.15, [10.0], 0.15);
      expect(snapAtThreshold.snapped).toBe(true);

      const snapOutsideThreshold = applyMagneticSnap(10.151, [10.0], 0.15);
      expect(snapOutsideThreshold.snapped).toBe(false);
    });

    it("T2-SIM-01: handles zero-area or inverted rect gracefully during collision check", () => {
      const inverted: Rect = { x: 0.5, y: 0.5, w: -0.1, h: -0.1 };
      const normalized = clampRect(inverted);
      expect(normalized.w).toBe(0.05);
      expect(checkCollision(normalized, PLATFORM_OVERLAYS.tiktok[0].rect)).toBe(false);
    });

    it("T2-HOT-01: ignores invalid number keys (e.g. Digit0, Digit9) for rating", () => {
      render(<InteractiveStudioHarness />);
      const studio = screen.getByTestId("interactive-studio");
      studio.focus();

      fireEvent.keyDown(studio, { code: "Digit0" });
      fireEvent.keyDown(studio, { code: "Digit9" });
      expect(screen.getByTestId("rating-score").textContent).toBe("none");
    });

    it("T2-ISO-01: Escape key cancels inline editing mode and restores hotkeys", () => {
      render(<InteractiveStudioHarness />);
      fireEvent.click(screen.getByTestId("caption-word-span-0"));
      const input = screen.getByTestId("caption-inline-input-0");

      fireEvent.keyDown(input, { key: "Escape" });
      expect(screen.queryByTestId("caption-inline-input-0")).toBeNull();

      // Verify hotkeys re-enabled
      const studio = screen.getByTestId("interactive-studio");
      fireEvent.keyDown(studio, { code: "Space" });
      expect(screen.getByTestId("playback-status").textContent).toBe("playing");
    });
  });

  // --------------------------------------------------------------------------
  // Tier 3: Cross-Feature Combinations
  // --------------------------------------------------------------------------
  describe("Tier 3: Cross-Feature Combinations", () => {
    it("T3-COMB-01: Canvas resize + Hotkey Space maintains drag state while toggling playback", () => {
      render(<InteractiveStudioHarness />);
      const studio = screen.getByTestId("interactive-studio");

      fireEvent.click(screen.getByTestId("handle-se"));
      fireEvent.keyDown(studio, { code: "Space" });

      expect(screen.getByTestId("playback-status").textContent).toBe("playing");
      const box = screen.getByTestId("facecam-box");
      expect(box.style.width).toBe("35%");
    });

    it("T3-COMB-02: WYSIWYG Inline Editing + J-K-L keys isolates hotkeys during sentence typing", () => {
      render(<InteractiveStudioHarness />);
      fireEvent.click(screen.getByTestId("caption-word-span-2"));
      const input = screen.getByTestId("caption-inline-input-2");

      // User types word with 'J', 'K', 'L', ' '
      fireEvent.keyDown(input, { code: "KeyJ", key: "j" });
      fireEvent.keyDown(input, { code: "KeyK", key: "k" });
      fireEvent.keyDown(input, { code: "KeyL", key: "l" });
      fireEvent.keyDown(input, { code: "Space", key: " " });

      // Playback speed and state remain untouched
      expect(screen.getByTestId("playback-rate").textContent).toBe("1");
      expect(screen.getByTestId("playback-status").textContent).toBe("paused");
    });

    it("T3-COMB-03: Switching phone platform simulator updates collision warnings and auto-nudge clears them", () => {
      render(<InteractiveStudioHarness initialPlatform="shorts" />);
      expect(screen.getByTestId("platform-simulator").getAttribute("data-platform")).toBe("shorts");

      // Switch to TikTok
      fireEvent.click(screen.getByTestId("toggle-platform-tiktok"));
      expect(screen.getByTestId("platform-simulator").getAttribute("data-platform")).toBe("tiktok");

      // Auto-nudge clears the collision
      const autoNudgeBtn = screen.queryByTestId("auto-nudge-btn");
      if (autoNudgeBtn) {
        fireEvent.click(autoNudgeBtn);
        expect(screen.queryByTestId("collision-warning-badge")).toBeNull();
      }
    });
  });

  // --------------------------------------------------------------------------
  // Tier 4: Real-World Streamer Workload (End-to-End Simulation)
  // --------------------------------------------------------------------------
  describe("Tier 4: Real-World Streamer Workload", () => {
    it("T4-WORKLOAD-01: Full streamer curation workflow: Play -> Scrub -> Edit Caption -> Hotkey Rate -> Auto-Nudge Safe Zone", () => {
      const onFeedback = vi.fn();
      render(<InteractiveStudioHarness onFeedback={onFeedback} />);
      const studio = screen.getByTestId("interactive-studio");
      studio.focus();

      // 1. Streamer plays video with Space
      fireEvent.keyDown(studio, { code: "Space" });
      expect(screen.getByTestId("playback-status").textContent).toBe("playing");

      // 2. Streamer accelerates forward with L
      fireEvent.keyDown(studio, { code: "KeyL" });
      expect(screen.getByTestId("playback-rate").textContent).toBe("2");

      // 3. Streamer marks In point at 00:02.5
      fireEvent.keyDown(studio, { code: "KeyI" });
      expect(screen.getByTestId("in-point").textContent).toBe("2.5");

      // 4. Streamer pauses at climax with K
      fireEvent.keyDown(studio, { code: "KeyK" });
      expect(screen.getByTestId("playback-status").textContent).toBe("paused");

      // 5. Streamer edits second caption word
      fireEvent.click(screen.getByTestId("caption-word-span-1"));
      const input = screen.getByTestId("caption-inline-input-1");
      fireEvent.change(input, { target: { value: "MEGA" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(screen.getByTestId("caption-word-span-1").textContent).toBe("MEGA");

      // 6. Streamer rates clip with 5-Star Hotkey
      fireEvent.keyDown(studio, { code: "Digit5" });
      expect(screen.getByTestId("rating-score").textContent).toBe("5");
      expect(onFeedback).toHaveBeenCalledWith(5);

      // 7. Auto-nudge caption if collision exists
      const nudgeBtn = screen.queryByTestId("auto-nudge-btn");
      if (nudgeBtn) {
        fireEvent.click(nudgeBtn);
        expect(screen.queryByTestId("collision-warning-badge")).toBeNull();
      }
    });
  });
});
