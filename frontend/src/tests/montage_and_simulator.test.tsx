/**
 * Automated Test Suite for Milestone M4:
 * Smart Montage Timeline & Phone Platform Simulator (R4 & R5)
 *
 * Verifies:
 * 1. AABB collision engine & platform exclusion zones for TikTok, Shorts, and Reels.
 * 2. Auto-nudge safe repositioning algorithm.
 * 3. Magnetic snapping (+/- 0.15s threshold).
 * 4. PhoneSimulator component platform toggling, UI chrome overlays, real-time collision warnings, and auto-nudge button.
 * 5. MontageTimeline component drag-and-drop reordering, transition selector node [ ⧗ ], and multi-clip sequential preview engine.
 * 6. Edge cases: multi-zone collisions, single-clip sequences, and boundary clamping.
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
// Setup DOM & Media Mocks for Happy-DOM Environment
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

describe("Milestone M4: Montage Timeline & Phone Platform Simulator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // 1. Collision Detection Engine & Safe Zone Math (collision.ts)
  // ==========================================================================
  describe("Collision Detection Engine & Math", () => {
    it("checkCollision: correctly detects 2D AABB overlapping rectangles", () => {
      const boxA: Rect = { x: 0.1, y: 0.1, w: 0.3, h: 0.3 };
      const boxB: Rect = { x: 0.2, y: 0.2, w: 0.3, h: 0.3 };
      const disjoint: Rect = { x: 0.6, y: 0.6, w: 0.2, h: 0.2 };

      expect(checkCollision(boxA, boxB)).toBe(true);
      expect(boxesIntersect(boxA, boxB)).toBe(true);
      expect(checkCollision(boxA, disjoint)).toBe(false);
      expect(boxesIntersect(boxA, disjoint)).toBe(false);
    });

    it("intersectionArea: calculates exact normalized overlap area", () => {
      const boxA: Rect = { x: 0.1, y: 0.1, w: 0.3, h: 0.3 };
      const boxB: Rect = { x: 0.2, y: 0.2, w: 0.3, h: 0.3 };
      // Overlap: x from 0.2 to 0.4 (w=0.2), y from 0.2 to 0.4 (h=0.2) -> area = 0.04
      expect(intersectionArea(boxA, boxB)).toBeCloseTo(0.04, 4);

      const disjoint: Rect = { x: 0.6, y: 0.6, w: 0.2, h: 0.2 };
      expect(intersectionArea(boxA, disjoint)).toBe(0);
    });

    it("getIntersectionRect: returns the correct intersection rectangle or null", () => {
      const boxA: Rect = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
      const boxB: Rect = { x: 0.4, y: 0.3, w: 0.3, h: 0.2 };

      const inter = getIntersectionRect(boxA, boxB);
      expect(inter).not.toBeNull();
      expect(inter?.x).toBeCloseTo(0.4, 2);
      expect(inter?.y).toBeCloseTo(0.3, 2);
      expect(inter?.w).toBeCloseTo(0.2, 2);
      expect(inter?.h).toBeCloseTo(0.2, 2);

      const nonOverlapping = getIntersectionRect(boxA, { x: 0.8, y: 0.8, w: 0.1, h: 0.1 });
      expect(nonOverlapping).toBeNull();
    });

    it("detectPlatformCollisions: identifies TikTok action rail collision", () => {
      const captionInRail: Rect = { x: 0.85, y: 0.50, w: 0.12, h: 0.05 };
      const collisions = detectPlatformCollisions(captionInRail, "tiktok");

      expect(collisions.length).toBeGreaterThan(0);
      expect(collisions.some((z) => z.name === "right_rail_actions")).toBe(true);
      expect(collisions.some((z) => z.type === "action_rail")).toBe(true);
    });

    it("detectPlatformCollisions: identifies TikTok bottom metadata collision", () => {
      const captionInBottom: Rect = { x: 0.10, y: 0.82, w: 0.70, h: 0.10 };
      const collisions = detectPlatformCollisions(captionInBottom, "tiktok");

      expect(collisions.length).toBeGreaterThan(0);
      expect(collisions.some((z) => z.name === "bottom_creator_sound")).toBe(true);
      expect(collisions.some((z) => z.type === "bottom_meta")).toBe(true);
    });

    it("detectPlatformCollisions: identifies YouTube Shorts engagement rail & metadata collisions", () => {
      const inShortsRail: Rect = { x: 0.84, y: 0.50, w: 0.14, h: 0.06 };
      const railCollisions = detectPlatformCollisions(inShortsRail, "shorts");
      expect(railCollisions.some((z) => z.name === "right_rail_engagement")).toBe(true);

      const inShortsBottom: Rect = { x: 0.10, y: 0.82, w: 0.70, h: 0.06 };
      const bottomCollisions = detectPlatformCollisions(inShortsBottom, "shorts");
      expect(bottomCollisions.some((z) => z.name === "bottom_channel_title")).toBe(true);
    });

    it("detectPlatformCollisions: identifies Instagram Reels action rail & bottom audio bar collisions", () => {
      const inReelsRail: Rect = { x: 0.83, y: 0.52, w: 0.15, h: 0.06 };
      const railCollisions = detectPlatformCollisions(inReelsRail, "reels");
      expect(railCollisions.some((z) => z.name === "right_rail_reels_actions")).toBe(true);

      const inReelsBottom: Rect = { x: 0.10, y: 0.82, w: 0.70, h: 0.06 };
      const bottomCollisions = detectPlatformCollisions(inReelsBottom, "reels");
      expect(bottomCollisions.some((z) => z.name === "bottom_reels_audio_bar")).toBe(true);
    });

    it("detectPlatformCollisions: returns empty array when element is in platform safe zone", () => {
      // Safe vertical placement for captions in 9:16 space is centered around y=0.64
      const safeCaption: Rect = { x: 0.10, y: 0.64, w: 0.70, h: 0.06 };

      expect(detectPlatformCollisions(safeCaption, "tiktok").length).toBe(0);
      expect(detectPlatformCollisions(safeCaption, "shorts").length).toBe(0);
      expect(detectPlatformCollisions(safeCaption, "reels").length).toBe(0);
    });

    it("checkPlatformCollisions: returns formatted CollisionResult with severity and message", () => {
      const colliding: Rect = { x: 0.85, y: 0.50, w: 0.12, h: 0.05 };
      const results = checkPlatformCollisions(colliding, "tiktok", "Facecam");

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].severity).toBe("critical");
      expect(results[0].zoneType).toBe("action_rail");
      expect(results[0].message).toContain("Facecam overlaps");
      expect(results[0].overlapArea).toBeGreaterThan(0);
    });

    it("autoNudgeSafePosition: shifts colliding right-rail element leftward to safe position", () => {
      const inRail: Rect = { x: 0.85, y: 0.50, w: 0.12, h: 0.05 };
      const nudged = autoNudgeSafePosition(inRail, "tiktok");

      expect(nudged.x).toBeLessThan(0.82);
      expect(detectPlatformCollisions(nudged, "tiktok").length).toBe(0);
    });

    it("autoNudgeSafePosition: shifts colliding bottom-metadata element upward to safe position", () => {
      const inBottom: Rect = { x: 0.10, y: 0.82, w: 0.70, h: 0.08 };
      const nudged = autoNudgeSafePosition(inBottom, "tiktok", { elementRole: "caption" });

      expect(nudged.y).toBeLessThan(0.76);
      expect(detectPlatformCollisions(nudged, "tiktok").length).toBe(0);
    });

    it("autoNudgeSafePosition: leaves already-safe elements unmodified", () => {
      const safeRect: Rect = { x: 0.15, y: 0.62, w: 0.65, h: 0.06 };
      const nudged = autoNudgeSafePosition(safeRect, "tiktok");

      expect(nudged.x).toBe(safeRect.x);
      expect(nudged.y).toBe(safeRect.y);
      expect(nudged.w).toBe(safeRect.w);
      expect(nudged.h).toBe(safeRect.h);
    });

    it("applyMagneticSnap: snaps within +/-0.15s threshold and ignores points outside", () => {
      const snapPoints = [5.0, 12.5, 20.0];

      // Inside threshold: 12.42 is 0.08s from 12.5 -> snap
      const snap1 = applyMagneticSnap(12.42, snapPoints, 0.15);
      expect(snap1.snapped).toBe(true);
      expect(snap1.time).toBe(12.5);

      // Boundary test: 12.36 is 0.14s from 12.5 -> snap
      const snap2 = applyMagneticSnap(12.36, snapPoints, 0.15);
      expect(snap2.snapped).toBe(true);
      expect(snap2.time).toBe(12.5);

      // Outside threshold: 12.30 is 0.20s from 12.5 -> no snap
      const snap3 = applyMagneticSnap(12.30, snapPoints, 0.15);
      expect(snap3.snapped).toBe(false);
      expect(snap3.time).toBe(12.30);
    });
  });

  // ==========================================================================
  // 2. Phone Simulator Component (PhoneSimulator.tsx)
  // ==========================================================================
  describe("PhoneSimulator Component", () => {
    it("renders phone mockup frame with 9:16 vertical viewport and dynamic island", () => {
      render(
        <PhoneSimulator captionYFrac={0.65}>
          <div data-testid="test-video-content">Video Canvas Content</div>
        </PhoneSimulator>
      );

      expect(screen.getByTestId("phone-simulator")).toBeDefined();
      expect(screen.getByTestId("phone-screen")).toBeDefined();
      expect(screen.getByTestId("phone-dynamic-island")).toBeDefined();
      expect(screen.getByTestId("test-video-content")).toBeDefined();
    });

    it("toggles between TikTok, YouTube Shorts, and Instagram Reels overlays", () => {
      const onPlatformChange = vi.fn();
      render(
        <PhoneSimulator
          initialPlatform="tiktok"
          onPlatformChange={onPlatformChange}
          captionYFrac={0.65}
        />
      );

      // Starts on TikTok
      expect(screen.getByTestId("top-tabs-tiktok")).toBeDefined();
      expect(screen.getByTestId("action-rail-tiktok")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-tiktok")).toBeDefined();

      // Switch to YouTube Shorts
      fireEvent.click(screen.getByTestId("platform-tab-shorts"));
      expect(onPlatformChange).toHaveBeenCalledWith("shorts");
      expect(screen.getByTestId("top-nav-shorts")).toBeDefined();
      expect(screen.getByTestId("action-rail-shorts")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-shorts")).toBeDefined();

      // Switch to Instagram Reels
      fireEvent.click(screen.getByTestId("platform-tab-reels"));
      expect(onPlatformChange).toHaveBeenCalledWith("reels");
      expect(screen.getByTestId("top-nav-reels")).toBeDefined();
      expect(screen.getByTestId("action-rail-reels")).toBeDefined();
      expect(screen.getByTestId("bottom-meta-reels")).toBeDefined();

      // Switch back to TikTok
      fireEvent.click(screen.getByTestId("platform-tab-tiktok"));
      expect(onPlatformChange).toHaveBeenCalledWith("tiktok");
      expect(screen.getByTestId("top-tabs-tiktok")).toBeDefined();
    });

    it("toggles platform UI overlay visibility via toggle button", () => {
      render(<PhoneSimulator initialPlatform="tiktok" captionYFrac={0.65} />);

      const toggleBtn = screen.getByTestId("toggle-overlay");
      expect(screen.getByTestId("platform-overlay")).toBeDefined();

      fireEvent.click(toggleBtn);
      expect(screen.queryByTestId("platform-overlay")).toBeNull();

      fireEvent.click(toggleBtn);
      expect(screen.getByTestId("platform-overlay")).toBeDefined();
    });

    it("detects real-time collision when caption vertical fraction overlaps bottom meta and shows warning", () => {
      // Y=0.82 overlaps bottom creator metadata on TikTok (y=0.76 to 0.96)
      render(<PhoneSimulator initialPlatform="tiktok" captionYFrac={0.82} />);

      const warningBadge = screen.getByTestId("collision-warning-badge");
      expect(warningBadge).toBeDefined();
      expect(warningBadge.textContent).toContain("Collision");

      const autoNudgeBtn = screen.getByTestId("auto-nudge-btn");
      expect(autoNudgeBtn).toBeDefined();
    });

    it("shows safe status badge when caption is positioned in platform safe zone", () => {
      // Y=0.64 is in safe zone between top tabs and bottom metadata
      render(<PhoneSimulator initialPlatform="tiktok" captionYFrac={0.64} />);

      expect(screen.queryByTestId("collision-warning-badge")).toBeNull();
      expect(screen.getByTestId("safe-status-badge")).toBeDefined();
      expect(screen.getByTestId("safe-status-badge").textContent).toContain("Safe");
    });

    it("auto-nudge button repositions colliding caption to safe zone and calls onNudgeCaption", () => {
      const onNudgeCaption = vi.fn();
      render(
        <PhoneSimulator
          initialPlatform="tiktok"
          captionYFrac={0.85}
          onNudgeCaption={onNudgeCaption}
        />
      );

      const autoNudgeBtn = screen.getByTestId("auto-nudge-btn");
      fireEvent.click(autoNudgeBtn);

      expect(onNudgeCaption).toHaveBeenCalled();
      const [safeRect, safeYFrac] = onNudgeCaption.mock.calls[0];
      expect(safeYFrac).toBeLessThan(0.76);
      expect(safeRect.y).toBeLessThan(0.74);
      expect(screen.getByTestId("nudge-feedback")).toBeDefined();
    });

    it("detects collision when facecam overlaps action rail and triggers facecam auto-nudge", () => {
      const onNudgeFacecam = vi.fn();
      // Facecam located in right action rail
      const collidingCam: Rect = { x: 0.84, y: 0.50, w: 0.14, h: 0.20 };

      render(
        <PhoneSimulator
          initialPlatform="tiktok"
          captionYFrac={0.64}
          facecamRect={collidingCam}
          onNudgeFacecam={onNudgeFacecam}
        />
      );

      expect(screen.getByTestId("collision-warning-badge")).toBeDefined();
      const autoNudgeBtn = screen.getByTestId("auto-nudge-btn");
      fireEvent.click(autoNudgeBtn);

      expect(onNudgeFacecam).toHaveBeenCalled();
      const [safeCam] = onNudgeFacecam.mock.calls[0];
      expect(safeCam.x).toBeLessThan(0.82);
    });
  });

  // ==========================================================================
  // 3. Montage Timeline Component (MontageTimeline.tsx)
  // ==========================================================================
  describe("MontageTimeline Component", () => {
    const mockClips: TimelineClipItem[] = [
      { id: "c1", clipId: "clip_1", title: "Sniper Ace", duration: 10.0, score: 95 },
      { id: "c2", clipId: "clip_2", title: "Clutch Defuse", duration: 8.0, score: 88 },
      { id: "c3", clipId: "clip_3", title: "Victory Celebration", duration: 12.0, score: 91 },
    ];

    it("renders sequenced clip cards with titles, scores, and formatted durations", () => {
      render(<MontageTimeline initialClips={mockClips} />);

      expect(screen.getByTestId("timeline-clip-0")).toBeDefined();
      expect(screen.getByTestId("timeline-clip-1")).toBeDefined();
      expect(screen.getByTestId("timeline-clip-2")).toBeDefined();

      expect(screen.getAllByText(/Sniper Ace/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Clutch Defuse/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Victory Celebration/).length).toBeGreaterThanOrEqual(1);

      // Total duration = 10.0 + 8.0 + 12.0 = 30.0s
      expect(screen.getByTestId("montage-time-display").textContent).toContain("30.0s");
    });

    it("reorders clips using accessible move controls and fires onSequenceChange", () => {
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={mockClips} onSequenceChange={onSequenceChange} />);

      // Click "move-right" on clip 0 (Sniper Ace)
      const moveRightBtn = screen.getByTestId("move-right-0");
      fireEvent.click(moveRightBtn);

      expect(onSequenceChange).toHaveBeenCalled();
      const reordered = onSequenceChange.mock.calls[0][0] as TimelineClipItem[];
      expect(reordered[0].title).toBe("Clutch Defuse");
      expect(reordered[1].title).toBe("Sniper Ace");
      expect(reordered[2].title).toBe("Victory Celebration");
    });

    it("reorders clips via HTML5 drag-and-drop events and displays drop indicator", () => {
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={mockClips} onSequenceChange={onSequenceChange} />);

      const clip0 = screen.getByTestId("timeline-clip-0");
      const clip2 = screen.getByTestId("timeline-clip-2");

      const mockDataTransfer = {
        setData: vi.fn(),
        getData: vi.fn().mockReturnValue("0"),
        effectAllowed: "",
        dropEffect: "",
      };

      fireEvent.dragStart(clip0, { dataTransfer: mockDataTransfer });
      fireEvent.dragOver(clip2, { dataTransfer: mockDataTransfer });

      expect(screen.getByTestId("drop-indicator")).toBeDefined();

      fireEvent.drop(clip2, { dataTransfer: mockDataTransfer });

      expect(onSequenceChange).toHaveBeenCalled();
      const updated = onSequenceChange.mock.calls[0][0] as TimelineClipItem[];
      expect(updated[0].title).toBe("Clutch Defuse");
      expect(updated[1].title).toBe("Victory Celebration");
      expect(updated[2].title).toBe("Sniper Ace");
    });

    it("magnetic snapping: playhead snaps to clip cut points within +/-0.15s and displays guide", () => {
      render(<MontageTimeline initialClips={mockClips} />);
      const scrubber = screen.getByTestId("montage-scrubber");

      // Cut points are at 0s, 10.0s, 18.0s, 30.0s.
      // Scrub to 10.08s (within 0.15s of 10.0s) -> should snap to 10.0s
      fireEvent.change(scrubber, { target: { value: 10.08 } });

      expect(screen.getByTestId("magnetic-snap-guide")).toBeDefined();
      expect(screen.getByTestId("magnetic-snap-guide").textContent).toContain("10.00s");
      expect(screen.getByTestId("montage-time-display").textContent).toContain("10.0s");

      // Scrub far from cut point (14.5s) -> should not snap
      fireEvent.change(scrubber, { target: { value: 14.5 } });
      expect(screen.queryByTestId("magnetic-snap-guide")).toBeNull();
      expect(screen.getByTestId("montage-time-display").textContent).toContain("14.5s");
    });

    it("transitions selector: clicking transition badge opens popover to select transition types and durations", () => {
      const onTransitionsChange = vi.fn();
      render(<MontageTimeline initialClips={mockClips} onTransitionsChange={onTransitionsChange} />);

      // Between clip 0 and clip 1 there is transition-badge-0
      const badge = screen.getByTestId("transition-badge-0");
      expect(badge).toBeDefined();

      // Click to open popover
      fireEvent.click(badge);
      expect(screen.getByTestId("transition-popover-0")).toBeDefined();

      // Select "wipe" transition
      const wipeOpt = screen.getByTestId("transition-opt-wipe-0");
      fireEvent.click(wipeOpt);

      expect(onTransitionsChange).toHaveBeenCalled();
      const transitions = onTransitionsChange.mock.calls[0][0];
      expect(transitions[0].type).toBe("wipe");

      // Popover closes after selection
      expect(screen.queryByTestId("transition-popover-0")).toBeNull();

      // Re-open and select duration
      fireEvent.click(badge);
      const dur08 = screen.getByTestId("transition-duration-0.8-0");
      fireEvent.click(dur08);

      expect(onTransitionsChange).toHaveBeenCalledTimes(2);
      const updatedTransitions = onTransitionsChange.mock.calls[1][0];
      expect(updatedTransitions[0].duration).toBe(0.8);
    });

    it("multi-clip sequential playback: play/pause toggles and advances sequential playback", () => {
      vi.useFakeTimers();
      render(<MontageTimeline initialClips={mockClips} />);

      const playBtn = screen.getByTestId("montage-play-pause");
      expect(playBtn.textContent).toContain("Play");

      // Start playback
      fireEvent.click(playBtn);
      expect(playBtn.textContent).toContain("Pause");

      // Advance timer by 500ms
      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(screen.getByTestId("montage-time-display").textContent).toContain("0.5s");

      // Fast forward past clip 0 (10.0s) to 11.0s (into clip 1)
      act(() => {
        vi.advanceTimersByTime(10500);
      });

      // Active clip index should have advanced to clip 1
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 2 of 3");

      vi.useRealTimers();
    });

    it("navigation controls: Next Clip and Prev Clip buttons jump playhead between clips", () => {
      render(<MontageTimeline initialClips={mockClips} />);

      const nextBtn = screen.getByTestId("montage-next-clip");
      const prevBtn = screen.getByTestId("montage-prev-clip");

      // Click Next Clip -> jumps to 10.0s (start of clip 1)
      fireEvent.click(nextBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("10.0s");
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 2 of 3");

      // Click Next Clip again -> jumps to 18.0s (start of clip 2)
      fireEvent.click(nextBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("18.0s");
      expect(screen.getByTestId("active-clip-preview").textContent).toContain("Clip 3 of 3");

      // Click Prev Clip -> jumps back to 10.0s (clip 1)
      fireEvent.click(prevBtn);
      expect(screen.getByTestId("montage-time-display").textContent).toContain("10.0s");
    });
  });

  // ==========================================================================
  // 4. Edge Cases & Boundary Conditions
  // ==========================================================================
  describe("Edge Cases & Boundary Conditions", () => {
    it("handles single-clip timeline gracefully without transition badges", () => {
      const singleClip: TimelineClipItem[] = [
        { id: "single", clipId: "c1", title: "Solo Highlight", duration: 15.0 },
      ];
      render(<MontageTimeline initialClips={singleClip} />);

      expect(screen.getByTestId("timeline-clip-0")).toBeDefined();
      expect(screen.queryByTestId("transition-badge-0")).toBeNull();
      expect(screen.getByTestId("montage-time-display").textContent).toContain("15.0s");
    });

    it("multi-zone collision: element overlapping both action rail and bottom metadata is handled", () => {
      // Element in bottom right corner overlapping both bottom meta and action rail
      const cornerRect: Rect = { x: 0.80, y: 0.78, w: 0.12, h: 0.08 };
      const collisions = detectPlatformCollisions(cornerRect, "tiktok");

      expect(collisions.length).toBeGreaterThanOrEqual(2);
      const nudged = autoNudgeSafePosition(cornerRect, "tiktok");

      // After auto-nudge, it must be cleared of both collisions
      expect(detectPlatformCollisions(nudged, "tiktok").length).toBe(0);
      expect(nudged.x + nudged.w).toBeLessThanOrEqual(0.82);
      expect(nudged.y + nudged.h).toBeLessThanOrEqual(0.76);
    });

    it("auto-nudge boundary clamping: never nudges elements outside normalized [0, 1] frame", () => {
      const hugeBox: Rect = { x: 0.9, y: 0.9, w: 0.5, h: 0.5 };
      const nudged = autoNudgeSafePosition(hugeBox, "tiktok");

      expect(nudged.x).toBeGreaterThanOrEqual(0.02);
      expect(nudged.y).toBeGreaterThanOrEqual(0.02);
      expect(nudged.x + nudged.w).toBeLessThanOrEqual(1.0);
      expect(nudged.y + nudged.h).toBeLessThanOrEqual(1.0);
    });

    it("auto-nudge works reliably on YouTube Shorts and Instagram Reels", () => {
      // Shorts bottom metadata collision
      const shortsColliding: Rect = { x: 0.10, y: 0.85, w: 0.70, h: 0.08 };
      expect(detectPlatformCollisions(shortsColliding, "shorts").length).toBeGreaterThan(0);
      const shortsNudged = autoNudgeSafePosition(shortsColliding, "shorts", { elementRole: "caption" });
      expect(detectPlatformCollisions(shortsNudged, "shorts").length).toBe(0);

      // Reels action rail collision
      const reelsColliding: Rect = { x: 0.85, y: 0.55, w: 0.12, h: 0.08 };
      expect(detectPlatformCollisions(reelsColliding, "reels").length).toBeGreaterThan(0);
      const reelsNudged = autoNudgeSafePosition(reelsColliding, "reels", { elementRole: "caption" });
      expect(detectPlatformCollisions(reelsNudged, "reels").length).toBe(0);
    });

    it("supports all 5 transition types: cut, crossfade, wipe, zoom, dip_black", () => {
      const mockItems: TimelineClipItem[] = [
        { id: "c1", clipId: "clip_1", title: "Intro", duration: 5.0 },
        { id: "c2", clipId: "clip_2", title: "Outro", duration: 5.0 },
      ];
      const onTransitionsChange = vi.fn();
      render(<MontageTimeline initialClips={mockItems} onTransitionsChange={onTransitionsChange} />);

      const types: TransitionType[] = ["cut", "crossfade", "wipe", "zoom", "dip_black"];
      const badge = screen.getByTestId("transition-badge-0");

      for (let i = 0; i < types.length; i++) {
        const tType = types[i];
        fireEvent.click(badge);
        const opt = screen.getByTestId(`transition-opt-${tType}-0`);
        fireEvent.click(opt);
        expect(onTransitionsChange).toHaveBeenCalled();
        const emitted = onTransitionsChange.mock.calls[onTransitionsChange.mock.calls.length - 1][0];
        expect(emitted[0].type).toBe(tType);
      }
    });

    it("respects boundary controls: cannot move first clip left or last clip right", () => {
      const mockItems: TimelineClipItem[] = [
        { id: "c1", clipId: "clip_1", title: "Clip 1", duration: 5.0 },
        { id: "c2", clipId: "clip_2", title: "Clip 2", duration: 5.0 },
      ];
      const onSequenceChange = vi.fn();
      render(<MontageTimeline initialClips={mockItems} onSequenceChange={onSequenceChange} />);

      const left0 = screen.getByTestId("move-left-0") as HTMLButtonElement;
      expect(left0.disabled).toBe(true);

      const right1 = screen.getByTestId("move-right-1") as HTMLButtonElement;
      expect(right1.disabled).toBe(true);

      fireEvent.click(left0);
      expect(onSequenceChange).not.toHaveBeenCalled();
    });

    it("simultaneously auto-nudges both colliding caption and colliding facecam in PhoneSimulator", () => {
      const onNudgeCaption = vi.fn();
      const onNudgeFacecam = vi.fn();

      render(
        <PhoneSimulator
          initialPlatform="tiktok"
          captionYFrac={0.88} // Collides with bottom metadata
          facecamRect={{ x: 0.85, y: 0.50, w: 0.12, h: 0.10 }} // Collides with action rail
          onNudgeCaption={onNudgeCaption}
          onNudgeFacecam={onNudgeFacecam}
        />
      );

      const badge = screen.getByTestId("collision-warning-badge");
      expect(badge).toBeDefined();

      const autoNudgeBtn = screen.getByTestId("auto-nudge-btn");
      fireEvent.click(autoNudgeBtn);

      expect(onNudgeCaption).toHaveBeenCalled();
      expect(onNudgeFacecam).toHaveBeenCalled();
    });

    it("fires onActiveClipChange and onPlayheadChange during scrubbing", () => {
      const onActiveClip = vi.fn();
      const onPlayhead = vi.fn();
      const mockItems: TimelineClipItem[] = [
        { id: "c1", clipId: "clip_1", title: "Clip A", duration: 10.0 },
        { id: "c2", clipId: "clip_2", title: "Clip B", duration: 10.0 },
      ];

      render(
        <MontageTimeline
          initialClips={mockItems}
          onActiveClipChange={onActiveClip}
          onPlayheadChange={onPlayhead}
        />
      );

      expect(onActiveClip).toHaveBeenCalledWith(expect.objectContaining({ title: "Clip A" }));

      const scrubber = screen.getByTestId("montage-scrubber");
      // Scrub to 15.0s (in Clip B)
      fireEvent.change(scrubber, { target: { value: 15.0 } });

      expect(onActiveClip).toHaveBeenCalledWith(expect.objectContaining({ title: "Clip B" }));
      expect(onPlayhead).toHaveBeenCalledWith(15.0, 20.0);
    });
  });
});
