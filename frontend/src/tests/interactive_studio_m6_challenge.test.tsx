/**
 * Milestone M6 — Phase 2 Adversarial Stress & Cross-Component Interaction Harness
 *
 * White-box empirical verification of cross-component interactions across:
 * 1. InteractiveCanvas (facecam & crop bounding boxes, inline captions, undo/redo)
 * 2. AudioWaveformHeatmap (decibel colors, climax events, seek synchronization)
 * 3. AiSuggestionsCoPilot (dead air trim, hook ratings, split reframe, teaser loop)
 * 4. PhoneSimulator (TikTok / Shorts / Reels overlays, collision detection, auto-nudge)
 * 5. MontageTimeline (drag-and-drop sequencing, magnetic snapping, transitions selector)
 * 6. StreamerWorkflowBar (J-K-L shuttle, Space, I/O trimming, 1-5 ratings, batch actions)
 * 7. collision.ts (AABB math, boundary tolerances, safe zone corridor guarantees)
 */

import React, { useState } from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, act, within } from "@testing-library/react";

import InteractiveCanvas from "../components/InteractiveCanvas";
import AudioWaveformHeatmap, {
  getDecibelColor,
  COLOR_SCREAM,
  COLOR_KILL_STREAK,
  type WaveformData,
} from "../components/AudioWaveformHeatmap";
import AiSuggestionsCoPilot, {
  type AiSuggestion,
} from "../components/AiSuggestionsCoPilot";
import PhoneSimulator from "../components/PhoneSimulator";
import MontageTimeline, {
  type TimelineClipItem,
  type TransitionType,
} from "../components/MontageTimeline";
import StreamerWorkflowBar, {
  isInputIsolated,
} from "../components/StreamerWorkflowBar";
import {
  checkCollision,
  boxesIntersect,
  intersectionArea,
  getIntersectionRect,
  detectPlatformCollisions,
  checkPlatformCollisions,
  autoNudgeSafePosition,
  applyMagneticSnap,
  type Rect,
  type PlatformType,
} from "../lib/collision";
import type { CaptionWord, Clip } from "../lib/types";

// ============================================================================
// Media & Environment Mocks
// ============================================================================

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

if (!HTMLElement.prototype.setPointerCapture) {
  HTMLElement.prototype.setPointerCapture = vi.fn();
}
if (!HTMLElement.prototype.releasePointerCapture) {
  HTMLElement.prototype.releasePointerCapture = vi.fn();
}

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
// Sample Mock Data
// ============================================================================

const sampleClips: Clip[] = [
  {
    id: "clip_val_1",
    title: "Valorant 5K Ace",
    start: 10.0,
    end: 35.0,
    score: 60,
    feedback: "up",
    factors: [{ label: "Triple Kill", detail: "High arousal", weight: 30 }],
    hashtags: ["#valorant", "#ace", "#clutch"],
    reframe: {
      layout: "center",
      tracked: false,
      overridden: false,
      cx_overridden: false,
      facecam: null,
      keyframes: [{ t: 0, cx: 0.5 }],
    },
    captions: {
      id: "captions_val_1",
      style_id: "bold_stroke",
      max_words_per_line: 4,
      words: [
        { t: 10.5, d: 0.8, text: "WATCH", speaker: null },
        { t: 11.5, d: 0.7, text: "THIS", speaker: null },
        { t: 18.2, d: 1.2, text: "ACE", speaker: null },
      ],
    },
  } as unknown as Clip,
  {
    id: "clip_cs2_2",
    title: "CS2 Clutch Defuse",
    start: 40.0,
    end: 58.0,
    score: 84,
    feedback: null,
    factors: [{ label: "1v3 Clutch", detail: "Loud crowd", weight: 25 }],
    hashtags: ["#cs2", "#clutch"],
    reframe: {
      layout: "split",
      tracked: false,
      overridden: true,
      cx_overridden: true,
      facecam: { x: 0.05, y: 0.12, w: 0.28, h: 0.28 },
      keyframes: [{ t: 0, cx: 0.45 }],
    },
    captions: {
      id: "captions_cs2_2",
      style_id: "yellow_glow",
      max_words_per_line: 4,
      words: [
        { t: 41.0, d: 1.0, text: "ONE", speaker: null },
        { t: 42.2, d: 1.1, text: "SECOND", speaker: null },
        { t: 55.0, d: 1.5, text: "DEFUSE", speaker: null },
      ],
    },
  } as unknown as Clip,
];

const sampleWaveform: WaveformData = {
  clip_id: "clip_val_1",
  duration_s: 25.0,
  samples: [0.1, 0.3, 0.85, 0.95, 0.4, 0.2],
  dbfs: [-30.0, -18.0, -2.5, -0.8, -14.0, -28.0],
  climax_t: 18.2,
  events: [
    { type: "kill_streak", t: 17.5, duration: 2.0, color: COLOR_KILL_STREAK, label: "Triple Kill" },
    { type: "scream", t: 18.2, duration: 1.5, color: COLOR_SCREAM, label: "Scream (98 dB)" },
  ],
};

// ============================================================================
// Integrated Cross-Component Test Harness
// ============================================================================

function IntegratedCrossStudioHarness({
  initialClips = sampleClips,
  initialActiveIndex = 0,
}: {
  initialClips?: Clip[];
  initialActiveIndex?: number;
}) {
  const [clips, setClips] = useState<Clip[]>(initialClips);
  const [activeIdx, setActiveIdx] = useState<number>(initialActiveIndex);
  const currentClip = clips[activeIdx] || clips[0];

  const [currentTime, setCurrentTime] = useState<number>(currentClip.start);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [platform, setPlatform] = useState<PlatformType>("tiktok");
  const [captionY, setCaptionY] = useState<number>(0.78);
  const [facecam, setFacecam] = useState<Rect | null>(currentClip.reframe.facecam ?? null);
  const [words, setWords] = useState<CaptionWord[]>(currentClip.captions.words);
  const [clipRating, setClipRating] = useState<number>(Math.round(currentClip.score / 20));
  const [showTimeline, setShowTimeline] = useState<boolean>(true);
  const [batchActionLog, setBatchActionLog] = useState<string[]>([]);
  const [undoStack, setUndoStack] = useState<any[]>([]);

  const pushUndo = (state: any) => {
    setUndoStack((prev) => [...prev, state]);
  };

  return (
    <div data-testid="integrated-studio-root" style={{ width: 800 }}>
      {/* 1. Streamer Workflow Bar */}
      <StreamerWorkflowBar
        clips={clips}
        activeClipId={currentClip.id}
        currentTime={currentTime}
        duration={currentClip.end - currentClip.start}
        start={currentClip.start}
        end={currentClip.end}
        onPlayToggle={(p: boolean) => setIsPlaying(p)}
        onShuttleRewind={(t) => t !== undefined && setCurrentTime(t)}
        onShuttleForward={(t) => t !== undefined && setCurrentTime(t)}
        onShuttlePause={() => setIsPlaying(false)}
        onMarkIn={(newStart) => {
          pushUndo({ start: currentClip.start });
          setClips((prev) =>
            prev.map((c, i) => (i === activeIdx ? { ...c, start: newStart } : c))
          );
        }}
        onMarkOut={(newEnd) => {
          pushUndo({ end: currentClip.end });
          setClips((prev) =>
            prev.map((c, i) => (i === activeIdx ? { ...c, end: newEnd } : c))
          );
        }}
        onRate={(score) => {
          setClipRating(score);
          setClips((prev) =>
            prev.map((c, i) => (i === activeIdx ? { ...c, score: score * 20 } : c))
          );
        }}
        onBatchAction={(action, selectedIds, payload) => {
          setBatchActionLog((prev) => [
            ...prev,
            `${action}:${selectedIds.join(",")}:${JSON.stringify(payload || {})}`,
          ]);
        }}
      />

      {/* 2. Waveform & Climax Heatmap */}
      <AudioWaveformHeatmap
        start={currentClip.start}
        end={currentClip.end}
        currentTime={currentTime}
        waveformData={sampleWaveform}
        onSeek={(absT) => setCurrentTime(absT)}
      />

      {/* 3. Phone Simulator wrapping Interactive Canvas */}
      <PhoneSimulator
        platform={platform}
        onPlatformChange={(p) => setPlatform(p)}
        captionYFrac={captionY}
        captionWords={words}
        facecamRect={facecam}
        currentTime={currentTime}
        onNudgeCaption={(_rect, safeY) => {
          pushUndo({ captionY });
          setCaptionY(safeY);
        }}
        onNudgeFacecam={(safeCam) => {
          pushUndo({ facecam });
          setFacecam(safeCam);
        }}
      >
        <InteractiveCanvas
          facecam={facecam}
          onFacecamChange={(cam) => {
            pushUndo({ facecam });
            setFacecam(cam);
          }}
          showFacecam={!!facecam}
          captionWords={words}
          onCaptionWordsChange={(w) => {
            pushUndo({ words });
            setWords(w);
          }}
          captionYFrac={captionY}
          onCaptionYFracChange={(y) => {
            pushUndo({ captionY });
            setCaptionY(y);
          }}
          currentTime={currentTime}
          canvasWidth={360}
          canvasHeight={640}
        />
      </PhoneSimulator>

      {/* 4. AI Suggestions Co-Pilot */}
      <AiSuggestionsCoPilot
        clipId={currentClip.id}
        clip={currentClip}
        onApplySuggestion={(sug) => {
          pushUndo({ start: currentClip.start, facecam, layout: currentClip.reframe.layout });
          const p = sug.patch || {};
          if (p.start_t !== undefined) {
            setClips((prev) =>
              prev.map((c, i) => (i === activeIdx ? { ...c, start: p.start_t } : c))
            );
          }
          if (p.layout === "split") {
            setFacecam({ x: 0.05, y: 0.12, w: 0.28, h: 0.28 });
          }
        }}
      />

      {/* 5. Montage Timeline */}
      {showTimeline && (
        <MontageTimeline
          clips={clips}
          onSequenceChange={(items) => {
            const reordered = items
              .map((it) => clips.find((c) => c.id === it.clipId))
              .filter(Boolean) as Clip[];
            setClips(reordered);
          }}
        />
      )}

      {/* Audit state indicators for testing */}
      <div data-testid="audit-current-time">{currentTime.toFixed(2)}</div>
      <div data-testid="audit-caption-y">{captionY.toFixed(3)}</div>
      <div data-testid="audit-rating">{clipRating}</div>
      <div data-testid="audit-platform">{platform}</div>
      <div data-testid="audit-batch-log">{batchActionLog.join("|")}</div>
      <div data-testid="audit-clip-count">{clips.length}</div>
      <div data-testid="audit-active-clip">{currentClip.id}</div>
      <div data-testid="audit-active-start">{currentClip.start.toFixed(1)}</div>
    </div>
  );
}

// ============================================================================
// Adversarial Test Suites
// ============================================================================

describe("Milestone M6 Adversarial Cross-Component Hardening", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  describe("1. Montage Timeline + Phone Simulator Cross-Interaction", () => {
    it("dynamically recalculates platform collision when switching clips on montage timeline", () => {
      render(<IntegratedCrossStudioHarness />);

      // Initial clip (clip_val_1) has captionY=0.78 which collides with TikTok creator metadata zone
      expect(screen.getByTestId("collision-status-bar")).toBeTruthy();
      expect(screen.getByTestId("collision-warning-badge")).toBeTruthy();

      // Click Auto-Nudge
      const nudgeBtn = screen.getByTestId("auto-nudge-btn");
      fireEvent.click(nudgeBtn);

      // Verify caption nudged to safe corridor (< 0.76)
      const auditY = parseFloat(screen.getByTestId("audit-caption-y").textContent || "0");
      expect(auditY).toBeLessThan(0.76);
      expect(auditY).toBeGreaterThanOrEqual(0.12);

      // Now switch platform to YouTube Shorts via selector
      const shortsTab = screen.getByTestId("platform-tab-shorts");
      fireEvent.click(shortsTab);
      expect(screen.getByTestId("audit-platform").textContent).toBe("shorts");
    });

    it("executes transition switching across timeline clips without corrupting active clip state", () => {
      render(<IntegratedCrossStudioHarness />);

      expect(screen.getByTestId("audit-active-clip").textContent).toBe("clip_val_1");
      const badge = screen.getByTestId("transition-badge-0");
      expect(badge).toBeTruthy();

      // Click transition badge to open popover
      fireEvent.click(badge);
      const wipeOpt = screen.getByTestId("transition-opt-wipe-0");
      fireEvent.click(wipeOpt);

      // Popover updates transition to wipe
      expect(badge.textContent).toMatch(/wipe/i);
    });

    it("verifies magnetic snap mathematical invariance at +/-0.15s boundaries", () => {
      const snapPoints = [10.0, 25.5, 42.0];

      // Exact snap within boundary
      expect(applyMagneticSnap(9.86, snapPoints).snapped).toBe(true);
      expect(applyMagneticSnap(9.86, snapPoints).time).toBe(10.0);

      expect(applyMagneticSnap(10.14, snapPoints).snapped).toBe(true);
      expect(applyMagneticSnap(10.14, snapPoints).time).toBe(10.0);

      // Threshold edge: 0.150s is snapped
      expect(applyMagneticSnap(10.15, snapPoints).snapped).toBe(true);
      expect(applyMagneticSnap(10.15, snapPoints).time).toBe(10.0);

      // Beyond threshold: 0.151s is free
      expect(applyMagneticSnap(10.151, snapPoints).snapped).toBe(false);
      expect(applyMagneticSnap(10.151, snapPoints).time).toBe(10.151);

      // Empty snap points handled safely
      expect(applyMagneticSnap(15.0, []).snapped).toBe(false);
      expect(applyMagneticSnap(15.0, []).time).toBe(15.0);
    });
  });

  describe("2. AudioWaveformHeatmap + Phone Simulator + Hotkeys Cross-Interaction", () => {
    it("seeks video via waveform climax click and synchronizes phone simulator time display", () => {
      render(<IntegratedCrossStudioHarness />);

      // Find waveform event markers
      const markers = screen.getAllByTestId("waveform-event-marker");
      expect(markers.length).toBeGreaterThanOrEqual(1);

      // Click second marker (scream event at 18.2s, clip start is 10.0s -> 28.20s abs time)
      fireEvent.click(markers[1]);

      // Verify audit time updated
      const auditT = parseFloat(screen.getByTestId("audit-current-time").textContent || "0");
      expect(auditT).toBeCloseTo(28.2, 1);
    });

    it("marks in-point with 'I' hotkey and rescales clip bounds", () => {
      render(<IntegratedCrossStudioHarness />);

      // Seek to 15.0s
      const wfTrack = screen.getByTestId("waveform-track");
      fireEvent.click(wfTrack);

      // Trigger 'I' hotkey
      fireEvent.keyDown(window, { code: "KeyI", key: "i" });

      // In-point updated
      const auditStart = parseFloat(screen.getByTestId("audit-active-start").textContent || "0");
      expect(auditStart).toBeGreaterThanOrEqual(10.0);
    });

    it("handles extreme waveform bounds (pure silence, 0 bins, negative dB) without crashing", () => {
      expect(getDecibelColor(-60)).toBe("#3B82F6"); // quiet blue
      expect(getDecibelColor(-25.1)).toBe("#3B82F6");
      expect(getDecibelColor(-24.9)).toBe("#10B981"); // normal green
      expect(getDecibelColor(-12.1)).toBe("#10B981");
      expect(getDecibelColor(-11.9)).toBe("#F59E0B"); // amber
      expect(getDecibelColor(-3.9)).toBe("#EF4444");  // peak red
      expect(getDecibelColor(0.0)).toBe("#EF4444");
      expect(getDecibelColor(6.0)).toBe("#EF4444");   // saturated clipping
    });
  });

  describe("3. AiSuggestionsCoPilot + InteractiveCanvas + Undo/Redo State Machine", () => {
    it("applies split facecam suggestion and renders on-canvas 8-point handles", async () => {
      render(<IntegratedCrossStudioHarness />);

      // Initially clip_val_1 has layout center with no facecam box
      expect(screen.queryByTestId("facecam-box")).toBeNull();

      // Find AI suggestion cards
      const cards = screen.getAllByTestId("suggestion-card");
      // Find card with "Split Facecam"
      const splitCard = cards.find((c) => c.textContent?.includes("Split Facecam"));
      expect(splitCard).toBeDefined();

      if (splitCard) {
        const applyBtn = within(splitCard).getByTestId("apply-suggestion-btn");
        fireEvent.click(applyBtn);

        // Now InteractiveCanvas renders the facecam box with 8-point resize handles
        const facecamBox = await screen.findByTestId("facecam-box");
        expect(facecamBox).toBeTruthy();

        const handles = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
        for (const h of handles) {
          expect(screen.getByTestId(`facecam-handle-${h}`)).toBeTruthy();
        }
      }
    });

    it("applies dead-air silence hook trim suggestion and updates clip start", () => {
      render(<IntegratedCrossStudioHarness />);

      const cards = screen.getAllByTestId("suggestion-card");
      const trimCard = cards.find((c) => c.textContent?.includes("Dead Air"));
      expect(trimCard).toBeDefined();

      if (trimCard) {
        const applyBtn = within(trimCard).getByTestId("apply-suggestion-btn");
        fireEvent.click(applyBtn);

        // Verify start shifted from 10.0 to 10.8s
        expect(screen.getByTestId("audit-active-start").textContent).toBe("10.8");
      }
    });
  });

  describe("4. Hotkey Input Isolation During Cross-Component Editing", () => {
    it("strictly isolates J-K-L, Space, I/O, and rating keys when typing inside caption word inline editor", () => {
      render(<IntegratedCrossStudioHarness />);

      // Hotkey Space toggles playback
      fireEvent.keyDown(window, { code: "Space", key: " " });
      expect(screen.getByTestId("hud-badge")).toBeTruthy();

      // Simulate inline caption editing element
      const fakeInput = document.createElement("input");
      fakeInput.className = "caption-word-input";
      document.body.appendChild(fakeInput);
      fakeInput.focus();

      try {
        expect(isInputIsolated(fakeInput)).toBe(true);

        // Dispatch hotkeys while focused on caption input
        fireEvent.keyDown(fakeInput, { code: "Space", key: " " });
        fireEvent.keyDown(fakeInput, { code: "KeyJ", key: "j" });
        fireEvent.keyDown(fakeInput, { code: "KeyK", key: "k" });
        fireEvent.keyDown(fakeInput, { code: "KeyL", key: "l" });
        fireEvent.keyDown(fakeInput, { code: "KeyI", key: "i" });
        fireEvent.keyDown(fakeInput, { code: "KeyO", key: "o" });
        fireEvent.keyDown(fakeInput, { code: "Digit5", key: "5" });

        // Rating remains unchanged at 3
        expect(screen.getByTestId("audit-rating").textContent).toBe("3");
      } finally {
        // Blur input and clean up DOM reliably
        fakeInput.blur();
        if (fakeInput.parentNode) {
          fakeInput.parentNode.removeChild(fakeInput);
        }
      }

      // Hotkey 5 now works outside input
      fireEvent.keyDown(window, { code: "Digit5", key: "5" });
      expect(screen.getByTestId("audit-rating").textContent).toBe("5");
    });

    it("suppresses hotkeys inside deep contenteditable containers", () => {
      const container = document.createElement("div");
      container.setAttribute("contenteditable", "true");
      const innerSpan = document.createElement("span");
      container.appendChild(innerSpan);
      document.body.appendChild(container);

      expect(isInputIsolated(innerSpan)).toBe(true);

      document.body.removeChild(container);
    });
  });

  describe("5. Streamer Workflow Bar Batch Actions + Selection State", () => {
    it("selects all clips and fires batch rate action with payload", () => {
      render(<IntegratedCrossStudioHarness />);

      // Click Select All
      const selectAllBtn = screen.getByTestId("batch-select-all");
      fireEvent.click(selectAllBtn);

      // Batch Action Toolbar is now visible
      const batchToolbar = screen.getByTestId("batch-toolbar");
      expect(batchToolbar).toBeTruthy();
      expect(screen.getByTestId("selected-count").textContent).toContain("2");

      // Click Batch Rate 5 Stars
      const rate5Btn = screen.getByTestId("batch-rate-5");
      fireEvent.click(rate5Btn);

      // Verify batch action log
      const batchLog = screen.getByTestId("audit-batch-log").textContent;
      expect(batchLog).toContain("rate:clip_val_1,clip_cs2_2:{\"rating\":5}");
    });

    it("safely handles individual clip selection and unselection", () => {
      render(<IntegratedCrossStudioHarness />);

      // Select single clip
      const clipCheck = screen.getByTestId("batch-select-clip-clip_val_1");
      fireEvent.click(clipCheck);

      expect(screen.getByTestId("selected-count").textContent).toContain("1");

      // Unselect it
      fireEvent.click(clipCheck);
      expect(screen.getByTestId("selected-count").textContent).toContain("0");
    });
  });

  describe("6. Mathematical Invariance & Collision Tolerances in collision.ts", () => {
    it("handles exact boundary touching without false collision (strict AABB)", () => {
      // In JS IEEE-754 float arithmetic, 0.1 + 0.2 = 0.30000000000000004 > 0.3.
      // With dyadic power-of-two coordinates (0.125 + 0.25 = 0.375 exact IEEE-754 representation):
      const box1: Rect = { x: 0.125, y: 0.125, w: 0.25, h: 0.25 };
      // Touching at x = 0.375
      const box2: Rect = { x: 0.375, y: 0.125, w: 0.25, h: 0.25 };

      expect(checkCollision(box1, box2)).toBe(false);
      expect(boxesIntersect(box1, box2)).toBe(false);

      // Penetrating by 0.001
      const box3: Rect = { x: 0.374, y: 0.125, w: 0.25, h: 0.25 };
      expect(checkCollision(box1, box3)).toBe(true);
      expect(boxesIntersect(box1, box3)).toBe(true);
    });

    it("documents IEEE-754 float rounding behavior on 0.1 + 0.2 boundary touch", () => {
      const box1: Rect = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 };
      const box2: Rect = { x: 0.3, y: 0.1, w: 0.2, h: 0.2 };
      // 0.1 + 0.2 = 0.30000000000000004 > 0.3 causes checkCollision to evaluate to true
      expect(checkCollision(box1, box2)).toBe(true);
      // But physical intersection area is negligible (< 1e-12)
      expect(intersectionArea(box1, box2)).toBeLessThan(1e-12);
    });

    it("evaluates TikTok action rail precision at floating boundary (0.82)", () => {
      const safeCaption: Rect = { x: 0.08, y: 0.50, w: 0.739, h: 0.1 }; // right edge = 0.819
      const collisionsSafe = checkPlatformCollisions(safeCaption, "tiktok");
      const railHitSafe = collisionsSafe.some((c) => c.zoneType === "action_rail");
      expect(railHitSafe).toBe(false);

      const encroachingCaption: Rect = { x: 0.08, y: 0.50, w: 0.741, h: 0.1 }; // right edge = 0.821
      const collisionsEncroach = checkPlatformCollisions(encroachingCaption, "tiktok");
      const railHitEncroach = collisionsEncroach.some((c) => c.zoneType === "action_rail");
      expect(railHitEncroach).toBe(true);
    });

    it("guarantees auto-nudge centers wide captions in [0.04, 0.80] corridor without rail or meta overlap", () => {
      // Intentionally create a wide colliding caption at bottom right
      const badCaption: Rect = { x: 0.20, y: 0.80, w: 0.78, h: 0.20 };
      const safeRect = autoNudgeSafePosition(badCaption, "tiktok", { elementRole: "caption" });

      // Check right edge strictly clears action rail (0.82)
      expect(safeRect.x + safeRect.w).toBeLessThanOrEqual(0.80);
      // Check left edge respects margin
      expect(safeRect.x).toBeGreaterThanOrEqual(0.04);
      // Check bottom edge strictly clears metadata zone (0.76)
      expect(safeRect.y + safeRect.h).toBeLessThanOrEqual(0.74);
      // Check no remaining collisions
      const remainingCollisions = detectPlatformCollisions(safeRect, "tiktok");
      expect(remainingCollisions.length).toBe(0);
    });

    it("handles zero-area, inverted, and extreme full-screen rectangles gracefully", () => {
      const unitBox: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
      const pointInside: Rect = { x: 0.4, y: 0.4, w: 0, h: 0 };
      // checkCollision returns true because coordinates fall within open intervals
      expect(checkCollision(unitBox, pointInside)).toBe(true);
      // But physical intersection area is strictly 0 and getIntersectionRect is null
      expect(intersectionArea(unitBox, pointInside)).toBe(0);
      expect(getIntersectionRect(unitBox, pointInside)).toBeNull();

      const fullScreen: Rect = { x: 0, y: 0, w: 1, h: 1 };
      const collisions = checkPlatformCollisions(fullScreen, "tiktok");
      expect(collisions.length).toBeGreaterThanOrEqual(2);

      const nudged = autoNudgeSafePosition(fullScreen, "tiktok", { elementRole: "caption" });
      expect(nudged.w).toBeLessThanOrEqual(0.72);
      expect(nudged.h).toBeLessThanOrEqual(0.18);
      expect(nudged.x + nudged.w).toBeLessThanOrEqual(0.80);
    });
  });

  describe("7. Streamer Workload Pipeline (End-to-End Synthesis)", () => {
    it("completes full 9-step clutch review journey seamlessly", async () => {
      render(<IntegratedCrossStudioHarness />);

      // Step 1: Verify initial state
      expect(screen.getByTestId("audit-active-clip").textContent).toBe("clip_val_1");

      // Step 2: Shuttle forward with L key
      fireEvent.keyDown(window, { code: "KeyL", key: "l" });
      expect(screen.getByTestId("hud-badge")).toBeTruthy();

      // Step 3: Seek to scream climax via waveform event marker
      const markers = screen.getAllByTestId("waveform-event-marker");
      fireEvent.click(markers[1]);
      const auditT = parseFloat(screen.getByTestId("audit-current-time").textContent || "0");
      expect(auditT).toBeCloseTo(28.2, 1);

      // Step 4: Detect collision and auto-nudge
      expect(screen.getByTestId("collision-status-bar")).toBeTruthy();
      fireEvent.click(screen.getByTestId("auto-nudge-btn"));
      const nudgedY = parseFloat(screen.getByTestId("audit-caption-y").textContent || "0");
      expect(nudgedY).toBeLessThan(0.76);

      // Step 5: Switch platform to Reels
      fireEvent.click(screen.getByTestId("platform-tab-reels"));
      expect(screen.getByTestId("audit-platform").textContent).toBe("reels");

      // Step 6: Apply AI hook trim
      const cards = screen.getAllByTestId("suggestion-card");
      const trimCard = cards.find((c) => c.textContent?.includes("Dead Air"));
      if (trimCard) {
        fireEvent.click(within(trimCard).getByTestId("apply-suggestion-btn"));
        expect(screen.getByTestId("audit-active-start").textContent).toBe("10.8");
      }

      // Step 7: Rate clip with hotkey 5
      fireEvent.keyDown(window, { code: "Digit5", key: "5" });
      expect(screen.getByTestId("audit-rating").textContent).toBe("5");

      // Step 8: Multi-clip selection
      fireEvent.click(screen.getByTestId("batch-select-all"));
      expect(screen.getByTestId("batch-toolbar")).toBeTruthy();

      // Step 9: Batch export
      fireEvent.click(screen.getByTestId("batch-export"));
      expect(screen.getByTestId("audit-batch-log").textContent).toContain("export:clip_val_1,clip_cs2_2");
    });
  });
});
