/**
 * Empirical Challenge & Stress-Test Suite for Milestone M4:
 * Smart Montage Timeline & Multi-Platform Phone Canvas Simulator (R4 & R5)
 *
 * Adversarial challenge suite covering:
 * 1. Collision Math Edge Cases: exact boundary touching, IEEE 754 float sensitivity, zero-area rects,
 *    negative coordinates, multi-zone simultaneous overlap, auto-nudge convergence & bounds clamping.
 * 2. Magnetic Snapping Bounds: precise +/- 0.15s threshold boundaries (0.149s, 0.150s, 0.151s).
 * 3. Timeline Reordering Stress: single-clip timelines, first-to-last, last-to-first,
 *    rapid drag-drops, and sequence index integrity.
 * 4. Transition Selector & Multi-Clip Preview Engine: 5 transition types (cut, crossfade, wipe, zoom, dip_black),
 *    duration switching, continuous playback sequence advancement, and boundary playhead behavior.
 * 5. Phone Simulator Stress: rapid platform switching, multi-collision alerts, and auto-nudge HUD feedback.
 */

import React from "react";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, cleanup, act } from "@testing-library/react";

import {
  checkCollision,
  boxesIntersect,
  intersectionArea,
  getIntersectionRect,
  detectPlatformCollisions,
  checkPlatformCollisions,
  calculateAutoNudge,
  autoNudgeSafePosition,
  applyMagneticSnap,
  PLATFORM_UI_ZONES,
  type Rect,
  type PlatformType,
} from "../lib/collision";

import PhoneSimulator from "../components/PhoneSimulator";
import MontageTimeline, {
  type TimelineClipItem,
  type TransitionType,
} from "../components/MontageTimeline";

// ============================================================================
// Media & DOM Mocks
// ============================================================================

Object.defineProperty(HTMLMediaElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});
Object.defineProperty(HTMLMediaElement.prototype, "pause", {
  configurable: true,
  value: vi.fn(),
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

describe("Empirical Challenge Suite: Milestone M4 (Montage Timeline & Phone Simulator)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // SECTION 1: Collision Math Edge Cases & Stress Tests
  // ==========================================================================
  describe("Section 1: Collision Math & Boundary Edge Cases", () => {
    it("evaluates exact boundary touching with binary-exact coordinates (x1 + w1 == x2): non-colliding", () => {
      // Using power-of-two coordinates (0.25, 0.5, 0.75) where 0.25 + 0.5 === 0.75 exactly in IEEE 754
      const exactBox: Rect = { x: 0.25, y: 0.25, w: 0.5, h: 0.5 };

      // Box touching right edge exactly at x=0.75
      const touchRight: Rect = { x: 0.75, y: 0.25, w: 0.25, h: 0.5 };
      expect(checkCollision(exactBox, touchRight)).toBe(false);
      expect(intersectionArea(exactBox, touchRight)).toBe(0);
      expect(getIntersectionRect(exactBox, touchRight)).toBeNull();

      // Box touching left edge exactly at x=0.25
      const touchLeft: Rect = { x: 0.0, y: 0.25, w: 0.25, h: 0.5 };
      expect(checkCollision(exactBox, touchLeft)).toBe(false);
      expect(intersectionArea(exactBox, touchLeft)).toBe(0);

      // Box touching bottom edge exactly at y=0.75
      const touchBottom: Rect = { x: 0.25, y: 0.75, w: 0.5, h: 0.25 };
      expect(checkCollision(exactBox, touchBottom)).toBe(false);
      expect(intersectionArea(exactBox, touchBottom)).toBe(0);

      // Box touching top edge exactly at y=0.25
      const touchTop: Rect = { x: 0.25, y: 0.0, w: 0.5, h: 0.25 };
      expect(checkCollision(exactBox, touchTop)).toBe(false);
      expect(intersectionArea(exactBox, touchTop)).toBe(0);

      // Box touching exact corner vertex
      const touchCorner: Rect = { x: 0.75, y: 0.75, w: 0.25, h: 0.25 };
      expect(checkCollision(exactBox, touchCorner)).toBe(false);
      expect(intersectionArea(exactBox, touchCorner)).toBe(0);
    });

    it("empirical finding: IEEE 754 float rounding causes false positive on 0.2 + 0.4 boundary touch", () => {
      // In JS, 0.2 + 0.4 = 0.6000000000000001 > 0.6
      const boxA: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
      const boxB: Rect = { x: 0.6, y: 0.2, w: 0.2, h: 0.4 };

      // Demonstrates lack of epsilon threshold in checkCollision:
      // a.x + a.w (0.6000000000000001) > b.x (0.6) evaluates to TRUE
      const floatCollision = checkCollision(boxA, boxB);
      expect(floatCollision).toBe(true);

      // However, getIntersectionRect handles this gracefully because overlap is negligible:
      const interRect = getIntersectionRect(boxA, boxB);
      // Width is 0.000000000000000111...
      expect(interRect?.w).toBeLessThan(1e-12);
      expect(intersectionArea(boxA, boxB)).toBeLessThan(1e-12);
    });

    it("evaluates zero-area rectangles behavior and intersection area consistency", () => {
      const unitBox: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
      // Degenerate zero-width, zero-height point inside the box
      const pointInside: Rect = { x: 0.4, y: 0.4, w: 0, h: 0 };
      // checkCollision returns true because 0.4 < 0.6 and 0.4+0 > 0.2
      const pointCollides = checkCollision(unitBox, pointInside);
      expect(pointCollides).toBe(true);
      // But physical intersection area is strictly 0
      const pointArea = intersectionArea(unitBox, pointInside);
      expect(pointArea).toBe(0);
      // And intersection rectangle is null
      expect(getIntersectionRect(unitBox, pointInside)).toBeNull();

      // Degenerate zero-area point outside the box
      const pointOutside: Rect = { x: 0.1, y: 0.1, w: 0, h: 0 };
      expect(checkCollision(unitBox, pointOutside)).toBe(false);
      expect(intersectionArea(unitBox, pointOutside)).toBe(0);
      expect(getIntersectionRect(unitBox, pointOutside)).toBeNull();
    });

    it("evaluates negative coordinates and out-of-frame rects", () => {
      const unitBox: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
      // Entirely negative coords (off canvas top-left)
      const offCanvasTopLeft: Rect = { x: -0.5, y: -0.5, w: 0.3, h: 0.3 };
      expect(checkCollision(unitBox, offCanvasTopLeft)).toBe(false);
      expect(detectPlatformCollisions(offCanvasTopLeft, "tiktok").length).toBe(0);
      expect(detectPlatformCollisions(offCanvasTopLeft, "shorts").length).toBe(0);
      expect(detectPlatformCollisions(offCanvasTopLeft, "reels").length).toBe(0);

      // Entirely beyond 1.0 (off canvas bottom-right)
      const offCanvasBottomRight: Rect = { x: 1.5, y: 1.5, w: 0.4, h: 0.4 };
      expect(checkCollision(unitBox, offCanvasBottomRight)).toBe(false);
      expect(detectPlatformCollisions(offCanvasBottomRight, "tiktok").length).toBe(0);

      // Partially out of frame, but straddling a platform zone
      // Straddles left boundary into bottom meta
      const straddlingBottom: Rect = { x: -0.05, y: 0.80, w: 0.20, h: 0.10 };
      const collisions = detectPlatformCollisions(straddlingBottom, "tiktok");
      expect(collisions.length).toBeGreaterThan(0);
      expect(collisions.some((z) => z.type === "bottom_meta")).toBe(true);
    });

    it("evaluates simultaneous multi-zone overlap and collision severity ordering", () => {
      // Element that overlaps top nav, right rail, and bottom meta simultaneously
      const massiveBox: Rect = { x: 0.10, y: 0.05, w: 0.85, h: 0.85 };
      const results = checkPlatformCollisions(massiveBox, "tiktok", "MassiveOverlay");

      expect(results.length).toBeGreaterThanOrEqual(3);
      // Critical collisions should be sorted before warning/info
      const severities = results.map((r) => r.severity);
      const firstWarningIdx = severities.indexOf("warning");
      const lastCriticalIdx = severities.lastIndexOf("critical");
      if (firstWarningIdx !== -1 && lastCriticalIdx !== -1) {
        expect(lastCriticalIdx).toBeLessThan(firstWarningIdx);
      }
    });

    it("evaluates auto-nudge safe position bounds clamping on colliding elements", () => {
      // Elements that collide with platform zones get nudged and clamped
      const collidingBoxes: Rect[] = [
        { x: 0.88, y: 0.50, w: 0.20, h: 0.10 }, // collides with right rail
        { x: 0.10, y: 0.85, w: 0.70, h: 0.15 }, // collides with bottom meta
        { x: 0.90, y: 0.90, w: 0.20, h: 0.20 }, // collides with bottom rail
      ];

      for (const box of collidingBoxes) {
        const nudged = autoNudgeSafePosition(box, "tiktok");
        expect(nudged.x).toBeGreaterThanOrEqual(0.02);
        expect(nudged.y).toBeGreaterThanOrEqual(0.02);
        expect(nudged.x + nudged.w).toBeLessThanOrEqual(0.99); // within normalized frame
        expect(nudged.y + nudged.h).toBeLessThanOrEqual(0.99);
      }
    });

    it("empirical finding: non-colliding out-of-bounds rects are passed through by autoNudgeSafePosition", () => {
      // If a rect has no platform zone collision (e.g. negative coords off top-left),
      // autoNudgeSafePosition returns early at line 337 without running bounds clamping
      const offCanvas: Rect = { x: -0.2, y: -0.2, w: 0.3, h: 0.2 };
      expect(detectPlatformCollisions(offCanvas, "tiktok").length).toBe(0);

      const returned = autoNudgeSafePosition(offCanvas, "tiktok");
      // Documents that autoNudgeSafePosition only triggers on UI zone collisions
      expect(returned.x).toBe(-0.2);
      expect(returned.y).toBe(-0.2);
    });

    it("empirical convergence generator: tests auto-nudge on standard caption and facecam placements", () => {
      // Test 50 distinct positions along bottom metadata zone for captions
      let resolvedCount = 0;
      for (let i = 0; i < 50; i++) {
        const testY = 0.76 + (i / 50) * 0.18; // y between 0.76 and 0.94
        const captionBox: Rect = { x: 0.08, y: testY, w: 0.65, h: 0.06 };
        expect(detectPlatformCollisions(captionBox, "tiktok").length).toBeGreaterThan(0);

        const nudged = autoNudgeSafePosition(captionBox, "tiktok", { elementRole: "caption" });
        const remaining = detectPlatformCollisions(nudged, "tiktok");
        if (remaining.length === 0) {
          resolvedCount++;
        }
      }
      // Standard caption boxes along bottom meta should all successfully resolve
      expect(resolvedCount).toBe(50);

      // Test 30 distinct positions along right rail for facecam
      let facecamResolved = 0;
      for (let i = 0; i < 30; i++) {
        const testY = 0.45 + (i / 30) * 0.35;
        const camBox: Rect = { x: 0.84, y: testY, w: 0.14, h: 0.14 };
        expect(detectPlatformCollisions(camBox, "tiktok").length).toBeGreaterThan(0);

        const nudged = autoNudgeSafePosition(camBox, "tiktok", { elementRole: "facecam" });
        const remaining = detectPlatformCollisions(nudged, "tiktok");
        if (remaining.length === 0) {
          facecamResolved++;
        }
      }
      expect(facecamResolved).toBe(30);
    });

    it("empirical finding: documents fallback behavior when oversized element cannot fit in safe zone", () => {
      // If an element is wider and taller than safe zone (e.g. 0.9 x 0.9),
      // autoNudgeSafePosition clamps it to safe dimensions (safeW=0.72, safeH=0.20)
      const oversized: Rect = { x: 0.05, y: 0.05, w: 0.90, h: 0.90 };
      const nudged = autoNudgeSafePosition(oversized, "tiktok");

      // Verify that dimensions are clamped
      expect(nudged.w).toBeLessThanOrEqual(0.72);
      expect(nudged.h).toBeLessThanOrEqual(0.20);
    });

    it("wide caption fallback centering: centers within [0.04, 0.80] and never encroaches right rail or bottom meta", () => {
      // Wide caption with w >= 0.78 placed in bottom-right collision area
      const wideBox: Rect = { x: 0.20, y: 0.76, w: 0.80, h: 0.08 };
      const nudged = autoNudgeSafePosition(wideBox, "tiktok", { elementRole: "caption" });

      expect(detectPlatformCollisions(nudged, "tiktok").length).toBe(0);
      expect(nudged.x).toBeGreaterThanOrEqual(0.04);
      expect(nudged.x + nudged.w).toBeLessThanOrEqual(0.80);
      expect(nudged.x + nudged.w).toBeLessThan(0.82); // Action rail safe
      expect(nudged.y + nudged.h).toBeLessThan(0.76); // Bottom metadata safe
    });
  });

  // ==========================================================================
  // SECTION 2: Magnetic Snapping Precision & Threshold Bounds
  // ==========================================================================
  describe("Section 2: Magnetic Snapping Precision & Threshold Bounds", () => {
    const snapPoints = [0.0, 5.0, 10.0, 15.0, 20.0];

    it("precisely tests threshold boundaries at delta = 0.149s, 0.150s, and 0.151s", () => {
      const center = 10.0;

      // 1. Delta = 0.149s (strictly inside threshold) -> MUST SNAP
      const snapLeftInside = applyMagneticSnap(center - 0.149, snapPoints, 0.15);
      expect(snapLeftInside.snapped).toBe(true);
      expect(snapLeftInside.time).toBe(10.0);
      expect(snapLeftInside.delta).toBeCloseTo(0.149, 4);

      const snapRightInside = applyMagneticSnap(center + 0.149, snapPoints, 0.15);
      expect(snapRightInside.snapped).toBe(true);
      expect(snapRightInside.time).toBe(10.0);
      expect(snapRightInside.delta).toBeCloseTo(0.149, 4);

      // 2. Delta = 0.150s (exact threshold boundary) -> MUST SNAP
      const snapExactBoundary = applyMagneticSnap(center - 0.150, snapPoints, 0.15);
      expect(snapExactBoundary.snapped).toBe(true);
      expect(snapExactBoundary.time).toBe(10.0);
      expect(snapExactBoundary.delta).toBeCloseTo(0.15, 4);

      // 3. Delta = 0.151s (strictly outside threshold) -> MUST NOT SNAP
      const snapLeftOutside = applyMagneticSnap(center - 0.151, snapPoints, 0.15);
      expect(snapLeftOutside.snapped).toBe(false);
      expect(snapLeftOutside.time).toBeCloseTo(9.849, 3);
      expect(snapLeftOutside.snapPoint).toBeUndefined();

      const snapRightOutside = applyMagneticSnap(center + 0.151, snapPoints, 0.15);
      expect(snapRightOutside.snapped).toBe(false);
      expect(snapRightOutside.time).toBeCloseTo(10.151, 3);
    });

    it("handles zero threshold (exact match only)", () => {
      const snapExact = applyMagneticSnap(10.0, snapPoints, 0.0);
      expect(snapExact.snapped).toBe(true);
      expect(snapExact.time).toBe(10.0);

      const snapOffByTiny = applyMagneticSnap(10.001, snapPoints, 0.0);
      expect(snapOffByTiny.snapped).toBe(false);
      expect(snapOffByTiny.time).toBe(10.001);
    });

    it("handles empty snap points array gracefully", () => {
      const snapEmpty = applyMagneticSnap(7.5, [], 0.15);
      expect(snapEmpty.snapped).toBe(false);
      expect(snapEmpty.time).toBe(7.5);
    });

    it("snaps to closest point when between two nearby cut points", () => {
      const closePoints = [9.9, 10.1];
      // Target at 9.96: distance to 9.9 is 0.06, distance to 10.1 is 0.14
      const snap = applyMagneticSnap(9.96, closePoints, 0.15);
      expect(snap.snapped).toBe(true);
      expect(snap.time).toBe(9.9);
    });
  });

  // ==========================================================================
  // SECTION 3: Timeline Reordering Stress Tests
  // ==========================================================================
  describe("Section 3: Timeline Reordering Stress Tests", () => {
    const generateClips = (count: number): TimelineClipItem[] => {
      return Array.from({ length: count }, (_, i) => ({
        id: `clip_${i}`,
        clipId: `c_${i}`,
        title: `Clip Title ${i}`,
        duration: 5.0 + i,
        score: 80 + i,
      }));
    };

    it("tests dragging first clip to last position", () => {
      const clips = generateClips(4); // [0, 1, 2, 3]
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={clips} onSequenceChange={onSequenceChange} />);

      const clip0 = screen.getByTestId("timeline-clip-0");
      const clip3 = screen.getByTestId("timeline-clip-3");

      const mockDataTransfer = {
        setData: vi.fn(),
        getData: vi.fn().mockReturnValue("0"),
        effectAllowed: "",
        dropEffect: "",
      };

      fireEvent.dragStart(clip0, { dataTransfer: mockDataTransfer });
      fireEvent.dragOver(clip3, { dataTransfer: mockDataTransfer });
      fireEvent.drop(clip3, { dataTransfer: mockDataTransfer });

      expect(onSequenceChange).toHaveBeenCalled();
      const updated = onSequenceChange.mock.calls[0][0] as TimelineClipItem[];
      expect(updated.map((c) => c.title)).toEqual([
        "Clip Title 1",
        "Clip Title 2",
        "Clip Title 3",
        "Clip Title 0",
      ]);
      expect(updated.length).toBe(4);
    });

    it("tests dragging last clip to first position", () => {
      const clips = generateClips(4); // [0, 1, 2, 3]
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={clips} onSequenceChange={onSequenceChange} />);

      const clip0 = screen.getByTestId("timeline-clip-0");
      const clip3 = screen.getByTestId("timeline-clip-3");

      const mockDataTransfer = {
        setData: vi.fn(),
        getData: vi.fn().mockReturnValue("3"),
        effectAllowed: "",
        dropEffect: "",
      };

      fireEvent.dragStart(clip3, { dataTransfer: mockDataTransfer });
      fireEvent.dragOver(clip0, { dataTransfer: mockDataTransfer });
      fireEvent.drop(clip0, { dataTransfer: mockDataTransfer });

      expect(onSequenceChange).toHaveBeenCalled();
      const updated = onSequenceChange.mock.calls[0][0] as TimelineClipItem[];
      expect(updated.map((c) => c.title)).toEqual([
        "Clip Title 3",
        "Clip Title 0",
        "Clip Title 1",
        "Clip Title 2",
      ]);
    });

    it("tests dropping clip on itself produces no reorder or duplicate mutation", () => {
      const clips = generateClips(3);
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={clips} onSequenceChange={onSequenceChange} />);

      const clip1 = screen.getByTestId("timeline-clip-1");
      const mockDataTransfer = {
        setData: vi.fn(),
        getData: vi.fn().mockReturnValue("1"),
        effectAllowed: "",
        dropEffect: "",
      };

      fireEvent.dragStart(clip1, { dataTransfer: mockDataTransfer });
      fireEvent.drop(clip1, { dataTransfer: mockDataTransfer });

      expect(onSequenceChange).not.toHaveBeenCalled();
    });

    it("stress tests rapid successive accessible reordering across 5 clips", () => {
      const clips = generateClips(5);
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={clips} onSequenceChange={onSequenceChange} />);

      // Move clip 1 to the left (becomes index 0)
      fireEvent.click(screen.getByTestId("move-left-1"));
      expect(onSequenceChange).toHaveBeenCalledTimes(1);

      // Now move clip 4 to the left (becomes index 3)
      fireEvent.click(screen.getByTestId("move-left-4"));
      expect(onSequenceChange).toHaveBeenCalledTimes(2);

      // Verify sequence integrity
      const finalSequence = onSequenceChange.mock.calls[1][0] as TimelineClipItem[];
      expect(finalSequence.length).toBe(5);
      expect(new Set(finalSequence.map((c) => c.id)).size).toBe(5);
    });

    it("verifies single-clip timeline preserves duration and has no reorder handles enabled", () => {
      const single: TimelineClipItem[] = [
        { id: "lone_clip", clipId: "c_lone", title: "Single Highlight", duration: 18.2, score: 99 },
      ];
      render(<MontageTimeline initialClips={single} />);

      expect(screen.getByTestId("timeline-clip-0")).toBeDefined();
      expect((screen.getByTestId("move-left-0") as HTMLButtonElement).disabled).toBe(true);
      expect((screen.getByTestId("move-right-0") as HTMLButtonElement).disabled).toBe(true);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("18.2s");
    });
  });

  // ==========================================================================
  // SECTION 4: Transitions & Multi-Clip Preview Engine
  // ==========================================================================
  describe("Section 4: Transitions & Multi-Clip Preview Engine", () => {
    const mockClips: TimelineClipItem[] = [
      { id: "c1", clipId: "clip_1", title: "Kill Climax", duration: 6.0 },
      { id: "c2", clipId: "clip_2", title: "Funny Reaction", duration: 4.0 },
      { id: "c3", clipId: "clip_3", title: "End Game", duration: 8.0 },
    ];

    it("tests transition switching across all 5 transition types: cut, crossfade, wipe, zoom, dip_black", () => {
      const onTransitionsChange = vi.fn();
      render(<MontageTimeline initialClips={mockClips} onTransitionsChange={onTransitionsChange} />);

      const types: TransitionType[] = ["cut", "crossfade", "wipe", "zoom", "dip_black"];
      const badge = screen.getByTestId("transition-badge-0");

      types.forEach((type, idx) => {
        fireEvent.click(badge);
        const optBtn = screen.getByTestId(`transition-opt-${type}-0`);
        fireEvent.click(optBtn);

        expect(onTransitionsChange).toHaveBeenCalledTimes(idx + 1);
        const latest = onTransitionsChange.mock.calls[idx][0];
        expect(latest[0].type).toBe(type);
      });
    });

    it("tests transition duration switching between 0.3s, 0.5s, and 0.8s", () => {
      const onTransitionsChange = vi.fn();
      render(<MontageTimeline initialClips={mockClips} onTransitionsChange={onTransitionsChange} />);

      const badge = screen.getByTestId("transition-badge-0");
      // Open popover once
      fireEvent.click(badge);
      expect(screen.getByTestId("transition-popover-0")).toBeDefined();

      const durations = [0.3, 0.5, 0.8];
      durations.forEach((dur, idx) => {
        const durBtn = screen.getByTestId(`transition-duration-${dur}-0`);
        fireEvent.click(durBtn);

        expect(onTransitionsChange).toHaveBeenCalledTimes(idx + 1);
        const latest = onTransitionsChange.mock.calls[idx][0];
        expect(latest[0].duration).toBe(dur);
      });
    });

    it("tests multi-clip continuous preview: sequential playback advancement through all clips", () => {
      vi.useFakeTimers();
      render(<MontageTimeline initialClips={mockClips} />);

      const playBtn = screen.getByTestId("montage-play-pause");
      fireEvent.click(playBtn); // Start playback

      // At t=0s -> Clip 1 of 3
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 1 of 3");

      // Advance by 6.5s -> into Clip 2 (starts at 6.0s, ends at 10.0s)
      act(() => {
        vi.advanceTimersByTime(6500);
      });
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 2 of 3");

      // Advance by 4.0s (total 10.5s) -> into Clip 3 (starts at 10.0s, ends at 18.0s)
      act(() => {
        vi.advanceTimersByTime(4000);
      });
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 3 of 3");

      // Advance past totalDuration (18.0s) -> should stop playback and reset
      act(() => {
        vi.advanceTimersByTime(8000);
      });
      expect(playBtn.textContent).toContain("Play");

      vi.useRealTimers();
    });

    it("tests transition effect overlay HUD rendering during transition window", () => {
      render(<MontageTimeline initialClips={mockClips} />);
      const scrubber = screen.getByTestId("montage-scrubber");

      // Clip 0 duration is 6.0s. Default transition is crossfade with duration 0.5s.
      // Transition window is [5.5s, 6.0s].
      // Scrub to 5.8s -> transition overlay HUD must be visible!
      fireEvent.change(scrubber, { target: { value: 5.8 } });

      const transitionOverlay = screen.queryByTestId("transition-effect-overlay");
      expect(transitionOverlay).not.toBeNull();
      expect(transitionOverlay?.textContent).toContain("crossfade");

      // Scrub to 3.0s (outside transition window) -> transition overlay HUD must be null
      fireEvent.change(scrubber, { target: { value: 3.0 } });
      expect(screen.queryByTestId("transition-effect-overlay")).toBeNull();
    });

    it("tests Next Clip and Prev Clip jump navigation bounds at edges", () => {
      render(<MontageTimeline initialClips={mockClips} />);

      const nextBtn = screen.getByTestId("montage-next-clip");
      const prevBtn = screen.getByTestId("montage-prev-clip");

      // Clicking Prev Clip when at start (0s) stays at 0s
      fireEvent.click(prevBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("0.0s");

      // Jump to clip 2 (6.0s)
      fireEvent.click(nextBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("6.0s");

      // Jump to clip 3 (10.0s)
      fireEvent.click(nextBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("10.0s");

      // Clicking Next Clip when already at last clip stays at clip 3 (10.0s)
      fireEvent.click(nextBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("10.0s");
    });
  });

  // ==========================================================================
  // SECTION 5: Phone Simulator Component Stress Tests
  // ==========================================================================
  describe("Section 5: Phone Simulator Component Stress Tests", () => {
    it("stress tests rapid platform switching without state corruption", () => {
      const onPlatformChange = vi.fn();
      render(<PhoneSimulator onPlatformChange={onPlatformChange} captionYFrac={0.65} />);

      const platforms: PlatformType[] = ["tiktok", "shorts", "reels", "tiktok", "reels", "shorts"];
      for (const p of platforms) {
        fireEvent.click(screen.getByTestId(`platform-tab-${p}`));
        expect(onPlatformChange).toHaveBeenLastCalledWith(p);
      }
    });

    it("renders pixel-accurate overlays for all 3 platforms via platform tab buttons", () => {
      render(<PhoneSimulator initialPlatform="tiktok" captionYFrac={0.65} />);
      expect(screen.getByTestId("top-tabs-tiktok")).toBeDefined();
      expect(screen.getByTestId("action-rail-tiktok")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-tiktok")).toBeDefined();

      fireEvent.click(screen.getByTestId("platform-tab-shorts"));
      expect(screen.getByTestId("top-nav-shorts")).toBeDefined();
      expect(screen.getByTestId("action-rail-shorts")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-shorts")).toBeDefined();

      fireEvent.click(screen.getByTestId("platform-tab-reels"));
      expect(screen.getByTestId("top-nav-reels")).toBeDefined();
      expect(screen.getByTestId("action-rail-reels")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-reels")).toBeDefined();
    });

    it("verifies simultaneous caption and facecam auto-nudge with feedback message", () => {
      const onNudgeCaption = vi.fn();
      const onNudgeFacecam = vi.fn();

      render(
        <PhoneSimulator
          initialPlatform="tiktok"
          captionYFrac={0.88}
          facecamRect={{ x: 0.85, y: 0.50, w: 0.12, h: 0.10 }}
          onNudgeCaption={onNudgeCaption}
          onNudgeFacecam={onNudgeFacecam}
        />
      );

      const autoNudgeBtn = screen.getByTestId("auto-nudge-btn");
      fireEvent.click(autoNudgeBtn);

      expect(onNudgeCaption).toHaveBeenCalled();
      expect(onNudgeFacecam).toHaveBeenCalled();
      expect(screen.getByTestId("nudge-feedback")).toBeDefined();
    });
  });
});
