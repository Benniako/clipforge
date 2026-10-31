import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import React, { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import InteractiveCanvas, {
  applyHandleResize,
  applyMagneticSnap,
  clampRect,
  type ResizeHandle,
} from "../components/InteractiveCanvas";
import type { CaptionWord, Rect, StyleTemplate } from "../lib/types";

// Setup Happy-DOM pointer capture stubs
if (!HTMLElement.prototype.setPointerCapture) {
  HTMLElement.prototype.setPointerCapture = vi.fn();
}
if (!HTMLElement.prototype.releasePointerCapture) {
  HTMLElement.prototype.releasePointerCapture = vi.fn();
}

describe("InteractiveCanvas Adversarial & Stress Testing Suite (Milestone M2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // Objective 1: Bounding Box Clamping & Extreme Offscreen Drags
  // ==========================================================================
  describe("Objective 1: Extreme Coordinate Clamping & Drag Bounds [0.0, 1.0]", () => {
    it("pure clampRect: rigidly clamps absurdly out-of-range coordinates", () => {
      // 1. Extreme negative values
      const negRect: Rect = { x: -1000.5, y: -250.7, w: 0.2, h: 0.3 };
      const clampedNeg = clampRect(negRect);
      expect(clampedNeg.x).toBe(0.0);
      expect(clampedNeg.y).toBe(0.0);
      expect(clampedNeg.w).toBe(0.2);
      expect(clampedNeg.h).toBe(0.3);

      // 2. Extreme positive coordinates
      const posRect: Rect = { x: 500.0, y: 888.8, w: 0.2, h: 0.2 };
      const clampedPos = clampRect(posRect);
      expect(clampedPos.x).toBeCloseTo(0.8, 4);
      expect(clampedPos.y).toBeCloseTo(0.8, 4);
      expect(clampedPos.x + clampedPos.w).toBeLessThanOrEqual(1.0);
      expect(clampedPos.y + clampedPos.h).toBeLessThanOrEqual(1.0);

      // 3. Oversized width and height (> 1.0)
      const giantRect: Rect = { x: -0.5, y: -0.5, w: 5.0, h: 10.0 };
      const clampedGiant = clampRect(giantRect);
      expect(clampedGiant.w).toBe(1.0);
      expect(clampedGiant.h).toBe(1.0);
      expect(clampedGiant.x).toBe(0.0);
      expect(clampedGiant.y).toBe(0.0);
      expect(clampedGiant.x + clampedGiant.w).toBe(1.0);
      expect(clampedGiant.y + clampedGiant.h).toBe(1.0);

      // 4. Zero and negative dimensions forced to minimum bounds (>= 0.05)
      const subZeroRect: Rect = { x: 0.5, y: 0.5, w: -0.8, h: 0.0 };
      const clampedSubZero = clampRect(subZeroRect);
      expect(clampedSubZero.w).toBe(0.05);
      expect(clampedSubZero.h).toBe(0.05);
      expect(clampedSubZero.x).toBe(0.5);
      expect(clampedSubZero.y).toBe(0.5);
    });

    it("UI pointer drag: cursor dragged extreme offscreen top-left clamps to (0, 0)", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.25, y: 0.25, w: 0.3, h: 0.3 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const interior = screen.getByTestId("facecam-drag-interior");

      // Start drag and fling pointer far beyond top-left screen (-50,000px)
      fireEvent.pointerDown(interior, { clientX: 250, clientY: 250, pointerId: 1 });
      fireEvent.pointerMove(interior, { clientX: -50000, clientY: -50000, pointerId: 1 });
      fireEvent.pointerUp(interior, { clientX: -50000, clientY: -50000, pointerId: 1 });

      expect(onFacecamChange).toHaveBeenCalled();
      const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
      expect(lastCall.x).toBe(0.0);
      expect(lastCall.y).toBe(0.0);
      expect(lastCall.w).toBe(0.3);
      expect(lastCall.h).toBe(0.3);
      expect(lastCall.x + lastCall.w).toBeLessThanOrEqual(1.0);
      expect(lastCall.y + lastCall.h).toBeLessThanOrEqual(1.0);
    });

    it("UI pointer drag: cursor dragged extreme offscreen bottom-right clamps to right/bottom edge", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.2, y: 0.2, w: 0.35, h: 0.25 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const interior = screen.getByTestId("facecam-drag-interior");

      // Start drag and fling pointer far beyond bottom-right (+80,000px)
      fireEvent.pointerDown(interior, { clientX: 200, clientY: 200, pointerId: 1 });
      fireEvent.pointerMove(interior, { clientX: 80000, clientY: 80000, pointerId: 1 });
      fireEvent.pointerUp(interior, { clientX: 80000, clientY: 80000, pointerId: 1 });

      expect(onFacecamChange).toHaveBeenCalled();
      const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
      expect(lastCall.x).toBeCloseTo(1.0 - 0.35, 4); // 0.65
      expect(lastCall.y).toBeCloseTo(1.0 - 0.25, 4); // 0.75
      expect(lastCall.w).toBe(0.35);
      expect(lastCall.h).toBe(0.25);
      expect(lastCall.x + lastCall.w).toBeCloseTo(1.0, 4);
      expect(lastCall.y + lastCall.h).toBeCloseTo(1.0, 4);
    });

    it("UI pointer drag: handles rapid multi-axis zig-zag drags without leaking out-of-bounds", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.4, y: 0.4, w: 0.2, h: 0.2 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const interior = screen.getByTestId("facecam-drag-interior");
      fireEvent.pointerDown(interior, { clientX: 400, clientY: 400, pointerId: 1 });

      const trajectory = [
        { clientX: -500, clientY: 1500 },
        { clientX: 2000, clientY: -300 },
        { clientX: 500, clientY: 500 },
        { clientX: -9999, clientY: -9999 },
        { clientX: 9999, clientY: 9999 },
      ];

      for (const pt of trajectory) {
        fireEvent.pointerMove(interior, { ...pt, pointerId: 1 });
        const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
        expect(lastCall.x).toBeGreaterThanOrEqual(0.0);
        expect(lastCall.y).toBeGreaterThanOrEqual(0.0);
        expect(lastCall.x + lastCall.w).toBeLessThanOrEqual(1.0 + 1e-6);
        expect(lastCall.y + lastCall.h).toBeLessThanOrEqual(1.0 + 1e-6);
      }

      fireEvent.pointerUp(interior, { clientX: 500, clientY: 500, pointerId: 1 });
    });
  });

  // ==========================================================================
  // Objective 2: 8-Point Resize Handles & Inversion Prevention
  // ==========================================================================
  describe("Objective 2: 8-Point Resize Handles & Inversion Stress", () => {
    const allHandles: ResizeHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
    const initialBox: Rect = { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };

    it("all 8 handles exist and have appropriate CSS cursors", () => {
      render(
        <InteractiveCanvas
          facecam={initialBox}
          showFacecam={true}
          canvasWidth={800}
          canvasHeight={450}
        />
      );

      const expectedCursors: Record<ResizeHandle, string> = {
        nw: "nwse-resize",
        se: "nwse-resize",
        ne: "nesw-resize",
        sw: "nesw-resize",
        n: "ns-resize",
        s: "ns-resize",
        e: "ew-resize",
        w: "ew-resize",
      };

      for (const h of allHandles) {
        const handleEl = screen.getByTestId(`facecam-handle-${h}`);
        expect(handleEl).toBeDefined();
        expect(handleEl.style.cursor).toBe(expectedCursors[h]);
      }
    });

    it("inversion stress: dragging SE handle past NW corner maintains min size w, h >= 0.05 and does NOT invert", () => {
      // initial right edge is 0.70, bottom edge is 0.70
      // Dragging SE handle by dx = -0.60, dy = -0.60 (past (0.3, 0.3) down to (-0.3, -0.3))
      const resized = applyHandleResize(initialBox, "se", -0.6, -0.6);
      expect(resized.x).toBe(0.3);
      expect(resized.y).toBe(0.3);
      expect(resized.w).toBe(0.05);
      expect(resized.h).toBe(0.05);
      expect(resized.x + resized.w).toBeLessThanOrEqual(1.0);
      expect(resized.y + resized.h).toBeLessThanOrEqual(1.0);
    });

    it("inversion stress: dragging NW handle past SE corner maintains min size w, h >= 0.05 and keeps SE anchor fixed", () => {
      // initial SE corner is (0.3 + 0.4, 0.3 + 0.4) = (0.7, 0.7)
      // Dragging NW handle by dx = +0.80, dy = +0.80 (past (0.7, 0.7))
      const resized = applyHandleResize(initialBox, "nw", 0.8, 0.8);
      // NW should stop at SE - 0.05 = (0.65, 0.65)
      expect(resized.x).toBeCloseTo(0.65, 4);
      expect(resized.y).toBeCloseTo(0.65, 4);
      expect(resized.w).toBe(0.05);
      expect(resized.h).toBe(0.05);
      expect(resized.x + resized.w).toBeCloseTo(0.7, 4);
      expect(resized.y + resized.h).toBeCloseTo(0.7, 4);
    });

    it("inversion stress: dragging NE handle past SW corner maintains min size and anchors", () => {
      // initial SW corner is x=0.3, y+h=0.7
      // Dragging NE handle left by dx = -0.6 (past x=0.3) and down by dy = +0.6 (past y+h=0.7)
      const resized = applyHandleResize(initialBox, "ne", -0.6, 0.6);
      expect(resized.x).toBe(0.3);
      expect(resized.y).toBeCloseTo(0.65, 4);
      expect(resized.w).toBe(0.05);
      expect(resized.h).toBe(0.05);
    });

    it("inversion stress: dragging SW handle past NE corner maintains min size and anchors", () => {
      // initial NE corner is x+w=0.7, y=0.3
      // Dragging SW handle right by dx = +0.6 (past x+w=0.7) and up by dy = -0.6 (past y=0.3)
      const resized = applyHandleResize(initialBox, "sw", 0.6, -0.6);
      expect(resized.x).toBeCloseTo(0.65, 4);
      expect(resized.y).toBe(0.3);
      expect(resized.w).toBe(0.05);
      expect(resized.h).toBe(0.05);
    });

    it("inversion stress across all 4 cardinal handles (n, s, e, w)", () => {
      // 'n' handle dragged down past bottom edge
      const resN = applyHandleResize(initialBox, "n", 0, 0.8);
      expect(resN.y).toBeCloseTo(0.65, 4);
      expect(resN.h).toBe(0.05);
      expect(resN.w).toBe(0.4);

      // 's' handle dragged up past top edge
      const resS = applyHandleResize(initialBox, "s", 0, -0.8);
      expect(resS.y).toBe(0.3);
      expect(resS.h).toBe(0.05);
      expect(resS.w).toBe(0.4);

      // 'w' handle dragged right past right edge
      const resW = applyHandleResize(initialBox, "w", 0.8, 0);
      expect(resW.x).toBeCloseTo(0.65, 4);
      expect(resW.w).toBe(0.05);
      expect(resW.h).toBe(0.4);

      // 'e' handle dragged left past left edge
      const resE = applyHandleResize(initialBox, "e", -0.8, 0);
      expect(resE.x).toBe(0.3);
      expect(resE.w).toBe(0.05);
      expect(resE.h).toBe(0.4);
    });

    it("aspect lock ratio resize preserves aspect ratio and respects [0, 1] bounds", () => {
      const squareBox: Rect = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 };
      // Aspect ratio 1.0 (square)
      const resLocked = applyHandleResize(squareBox, "se", 0.2, 0.1, 1.0);
      expect(resLocked.w).toBeCloseTo(resLocked.h, 4);
      expect(resLocked.w).toBeGreaterThanOrEqual(0.05);
      expect(resLocked.x + resLocked.w).toBeLessThanOrEqual(1.0);
      expect(resLocked.y + resLocked.h).toBeLessThanOrEqual(1.0);

      // Aspect ratio 16/9
      const res169 = applyHandleResize(squareBox, "se", 0.32, 0.05, 16 / 9);
      expect(res169.w / res169.h).toBeCloseTo(16 / 9, 2);
    });

    it("UI pointer drag on all 8 handles triggers onFacecamChange with clamped dimensions", () => {
      const onFacecamChange = vi.fn();
      render(
        <InteractiveCanvas
          facecam={initialBox}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      for (const h of allHandles) {
        onFacecamChange.mockClear();
        const handleEl = screen.getByTestId(`facecam-handle-${h}`);
        fireEvent.pointerDown(handleEl, { clientX: 500, clientY: 500, pointerId: 1 });
        fireEvent.pointerMove(handleEl, { clientX: 100, clientY: 100, pointerId: 1 });
        fireEvent.pointerUp(handleEl, { clientX: 100, clientY: 100, pointerId: 1 });

        expect(onFacecamChange).toHaveBeenCalled();
        const res = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
        expect(res.w).toBeGreaterThanOrEqual(0.05);
        expect(res.h).toBeGreaterThanOrEqual(0.05);
        expect(res.x).toBeGreaterThanOrEqual(0.0);
        expect(res.y).toBeGreaterThanOrEqual(0.0);
        expect(res.x + res.w).toBeLessThanOrEqual(1.0 + 1e-6);
        expect(res.y + res.h).toBeLessThanOrEqual(1.0 + 1e-6);
      }
    });
  });

  // ==========================================================================
  // Objective 3: Crop Magnetic Snap & Boundary Precision
  // ==========================================================================
  describe("Objective 3: Crop Magnetic Snap & Boundary Conditions", () => {
    it("applyMagneticSnap: rigorous boundary testing at exact threshold limits (0.50 ± 0.02)", () => {
      // Exactly at lower boundary: 0.50 - 0.02 = 0.480000
      const atLowerEdge = applyMagneticSnap(0.48, [0.5], 0.02);
      expect(atLowerEdge.snapped).toBe(true);
      expect(atLowerEdge.value).toBe(0.5);

      // Just inside lower boundary: 0.480001
      const insideLowerEdge = applyMagneticSnap(0.480001, [0.5], 0.02);
      expect(insideLowerEdge.snapped).toBe(true);
      expect(insideLowerEdge.value).toBe(0.5);

      // Just outside lower boundary: 0.479990
      const outsideLowerEdge = applyMagneticSnap(0.47999, [0.5], 0.02);
      expect(outsideLowerEdge.snapped).toBe(false);
      expect(outsideLowerEdge.value).toBeCloseTo(0.47999, 5);

      // Exactly at center: 0.500000
      const atCenter = applyMagneticSnap(0.5, [0.5], 0.02);
      expect(atCenter.snapped).toBe(true);
      expect(atCenter.value).toBe(0.5);

      // Exactly at upper boundary: 0.50 + 0.02 = 0.520000
      const atUpperEdge = applyMagneticSnap(0.52, [0.5], 0.02);
      expect(atUpperEdge.snapped).toBe(true);
      expect(atUpperEdge.value).toBe(0.5);

      // Just outside upper boundary: 0.520002
      const outsideUpperEdge = applyMagneticSnap(0.520002, [0.5], 0.02);
      expect(outsideUpperEdge.snapped).toBe(false);
      expect(outsideUpperEdge.value).toBeCloseTo(0.520002, 5);
    });

    it("crop window: bounds clamping when dragged to extreme left and right", () => {
      const onCropCxChange = vi.fn();
      const cropWidthFrac = 81 / 256; // ~0.316406
      const halfWidth = cropWidthFrac / 2; // ~0.158203

      render(
        <InteractiveCanvas
          cropCx={0.5}
          onCropCxChange={onCropCxChange}
          showCropOverlay={true}
          cropWidthFrac={cropWidthFrac}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const handle = screen.getByTestId("crop-drag-handle");

      // 1. Drag extreme left (e.g. -10,000px)
      fireEvent.pointerDown(handle, { clientX: 500, clientY: 20, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: -10000, clientY: 20, pointerId: 1 });

      const leftCall = onCropCxChange.mock.calls[onCropCxChange.mock.calls.length - 1][0];
      expect(leftCall).toBeCloseTo(halfWidth, 4);

      // 2. Drag extreme right (e.g. +10,000px)
      fireEvent.pointerMove(handle, { clientX: 10000, clientY: 20, pointerId: 1 });
      const rightCall = onCropCxChange.mock.calls[onCropCxChange.mock.calls.length - 1][0];
      expect(rightCall).toBeCloseTo(1.0 - halfWidth, 4);

      fireEvent.pointerUp(handle, { clientX: 10000, clientY: 20, pointerId: 1 });
    });

    it("crop snap indicator toggle: dynamically mounts on snap and unmounts on release/departure", () => {
      const onCropCxChange = vi.fn();
      render(
        <InteractiveCanvas
          cropCx={0.3}
          onCropCxChange={onCropCxChange}
          showCropOverlay={true}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const handle = screen.getByTestId("crop-drag-handle");
      expect(screen.queryByTestId("crop-snap-indicator")).toBeNull();

      // Drag to snap zone (delta +190px => 0.49, near 0.5)
      fireEvent.pointerDown(handle, { clientX: 300, clientY: 20, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: 490, clientY: 20, pointerId: 1 });

      expect(screen.getByTestId("crop-snap-indicator")).toBeDefined();
      expect(screen.getByTestId("crop-snap-indicator").textContent).toContain("CENTER 50%");

      // Drag out of snap zone (delta +350px => 0.65)
      fireEvent.pointerMove(handle, { clientX: 650, clientY: 20, pointerId: 1 });
      expect(screen.queryByTestId("crop-snap-indicator")).toBeNull();

      fireEvent.pointerUp(handle, { clientX: 650, clientY: 20, pointerId: 1 });
    });
  });

  // ==========================================================================
  // Objective 4: Caption Inline Word Editing & Undo/Redo Integration
  // ==========================================================================
  describe("Objective 4: Caption Inline Editing Robustness & Undo/Redo", () => {
    const baseWords: CaptionWord[] = [
      { t: 0.0, d: 0.5, text: "HELLO", speaker: null },
      { t: 0.6, d: 0.5, text: "WORLD", speaker: null },
      { t: 1.2, d: 0.5, text: "CLIPFORGE", speaker: null },
    ];

    it("allows committing empty word input without crashing", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-1"));
      const input = screen.getByTestId("caption-word-input-1") as HTMLInputElement;

      fireEvent.change(input, { target: { value: "" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updated = onCaptionWordsChange.mock.calls[0][0];
      expect(updated[1].text).toBe("");
    });

    it("handles special characters, unicode, emojis, and HTML entities securely", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      const complexPayload = `🔥 100% <script>alert("PWN")</script> & "quotes" 日本語`;

      fireEvent.click(screen.getByTestId("caption-word-0"));
      const input = screen.getByTestId("caption-word-input-0") as HTMLInputElement;

      fireEvent.change(input, { target: { value: complexPayload } });
      fireEvent.blur(input);

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updated = onCaptionWordsChange.mock.calls[0][0];
      expect(updated[0].text).toBe(complexPayload);
    });

    it("handles whitespace-only text edits", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-2"));
      const input = screen.getByTestId("caption-word-input-2") as HTMLInputElement;

      fireEvent.change(input, { target: { value: "   \t   " } });
      fireEvent.blur(input);

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updated = onCaptionWordsChange.mock.calls[0][0];
      expect(updated[2].text).toBe("   \t   ");
    });

    it("handles rapid consecutive clicks switching active word", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      // Click word 0
      fireEvent.click(screen.getByTestId("caption-word-0"));
      const input0 = screen.getByTestId("caption-word-input-0") as HTMLInputElement;
      fireEvent.change(input0, { target: { value: "FIRST_EDIT" } });

      // In browser event flow, clicking outside blurs active input and commits
      fireEvent.blur(input0);
      fireEvent.click(screen.getByTestId("caption-word-1"));

      // Verify word 0 edit was committed via blur
      expect(onCaptionWordsChange).toHaveBeenCalled();
      expect(onCaptionWordsChange.mock.calls[0][0][0].text).toBe("FIRST_EDIT");

      // Verify word 1 is now in edit mode
      const input1 = screen.getByTestId("caption-word-input-1") as HTMLInputElement;
      expect(input1).toBeDefined();
      expect(input1.value).toBe("WORLD");
    });

    it("Tab and Shift+Tab navigate between words while committing edits", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      // Start editing word 0
      fireEvent.click(screen.getByTestId("caption-word-0"));
      const input0 = screen.getByTestId("caption-word-input-0") as HTMLInputElement;
      fireEvent.change(input0, { target: { value: "STEP_1" } });

      // Press Tab -> commits word 0 and moves to word 1
      fireEvent.keyDown(input0, { key: "Tab", shiftKey: false });

      expect(onCaptionWordsChange).toHaveBeenCalled();
      expect(onCaptionWordsChange.mock.calls[0][0][0].text).toBe("STEP_1");

      const input1 = screen.getByTestId("caption-word-input-1") as HTMLInputElement;
      expect(input1).toBeDefined();

      // Press Shift+Tab -> commits word 1 and moves back to word 0
      fireEvent.change(input1, { target: { value: "STEP_2" } });
      fireEvent.keyDown(input1, { key: "Tab", shiftKey: true });

      expect(onCaptionWordsChange).toHaveBeenCalledTimes(2);
      expect(onCaptionWordsChange.mock.calls[1][0][1].text).toBe("STEP_2");

      const input0Again = screen.getByTestId("caption-word-input-0") as HTMLInputElement;
      expect(input0Again).toBeDefined();
    });

    it("Tab on the last word commits and closes editor gracefully", () => {
      const onCaptionWordsChange = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      // Edit last word (index 2)
      fireEvent.click(screen.getByTestId("caption-word-2"));
      const input2 = screen.getByTestId("caption-word-input-2") as HTMLInputElement;
      fireEvent.change(input2, { target: { value: "FINAL" } });

      // Tab on last word
      fireEvent.keyDown(input2, { key: "Tab", shiftKey: false });

      expect(onCaptionWordsChange).toHaveBeenCalled();
      expect(onCaptionWordsChange.mock.calls[0][0][2].text).toBe("FINAL");
      // All inputs closed
      expect(screen.queryByTestId("caption-word-input-0")).toBeNull();
      expect(screen.queryByTestId("caption-word-input-1")).toBeNull();
      expect(screen.queryByTestId("caption-word-input-2")).toBeNull();
    });

    it("does not trigger canvas-level undo while typing inside the text input", () => {
      const onUndo = vi.fn();
      render(
        <InteractiveCanvas
          captionWords={baseWords}
          onUndo={onUndo}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-0"));
      const input0 = screen.getByTestId("caption-word-input-0");

      // Press Ctrl+Z inside input
      fireEvent.keyDown(input0, { key: "z", ctrlKey: true });
      expect(onUndo).not.toHaveBeenCalled();

      // Press Ctrl+Z on canvas outside input
      const canvas = screen.getByTestId("interactive-canvas");
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true });
      expect(onUndo).toHaveBeenCalledTimes(1);
    });

    it("multi-step stateful undo/redo history stress verification", () => {
      function HistoryHarness() {
        const [words, setWords] = useState<CaptionWord[]>(baseWords);
        const [past, setPast] = useState<CaptionWord[][]>([]);
        const [future, setFuture] = useState<CaptionWord[][]>([]);

        const handleChange = (newWords: CaptionWord[]) => {
          setPast((p) => [...p, words]);
          setFuture([]);
          setWords(newWords);
        };

        const handleUndo = () => {
          if (past.length === 0) return;
          const prev = past[past.length - 1];
          setPast((p) => p.slice(0, -1));
          setFuture((f) => [words, ...f]);
          setWords(prev);
        };

        const handleRedo = () => {
          if (future.length === 0) return;
          const next = future[0];
          setFuture((f) => f.slice(1));
          setPast((p) => [...p, words]);
          setWords(next);
        };

        return (
          <InteractiveCanvas
            captionWords={words}
            onCaptionWordsChange={handleChange}
            onUndo={handleUndo}
            onRedo={handleRedo}
            showCaptions={true}
          />
        );
      }

      render(<HistoryHarness />);
      const canvas = screen.getByTestId("interactive-canvas");

      // Step 1: Change word 0
      fireEvent.click(screen.getByTestId("caption-word-0"));
      fireEvent.change(screen.getByTestId("caption-word-input-0"), { target: { value: "EDIT_1" } });
      fireEvent.keyDown(screen.getByTestId("caption-word-input-0"), { key: "Enter" });
      expect(screen.getByTestId("caption-word-0").textContent).toBe("EDIT_1");

      // Step 2: Change word 1
      fireEvent.click(screen.getByTestId("caption-word-1"));
      fireEvent.change(screen.getByTestId("caption-word-input-1"), { target: { value: "EDIT_2" } });
      fireEvent.keyDown(screen.getByTestId("caption-word-input-1"), { key: "Enter" });
      expect(screen.getByTestId("caption-word-1").textContent).toBe("EDIT_2");

      // Step 3: Undo -> word 1 goes back to "WORLD"
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true });
      expect(screen.getByTestId("caption-word-1").textContent).toBe("WORLD");
      expect(screen.getByTestId("caption-word-0").textContent).toBe("EDIT_1");

      // Step 4: Undo again -> word 0 goes back to "HELLO"
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true });
      expect(screen.getByTestId("caption-word-0").textContent).toBe("HELLO");

      // Step 5: Redo -> word 0 restored to "EDIT_1"
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true, shiftKey: true });
      expect(screen.getByTestId("caption-word-0").textContent).toBe("EDIT_1");
      expect(screen.getByTestId("caption-word-1").textContent).toBe("WORLD");

      // Step 6: Redo again -> word 1 restored to "EDIT_2"
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true, shiftKey: true });
      expect(screen.getByTestId("caption-word-1").textContent).toBe("EDIT_2");
    });
  });

  // ==========================================================================
  // Objective 5: Edge Cases, Null Prop States & Pointer Lifecycle
  // ==========================================================================
  describe("Objective 5: Edge Cases, Empty Props & Pointer Lifecycle", () => {
    it("handles null / undefined / empty props without runtime errors", () => {
      const { container } = render(
        <InteractiveCanvas
          facecam={null}
          cropCx={null}
          captionWords={[]}
          showFacecam={false}
          showCropOverlay={false}
          showCaptions={false}
          showSafeGuides={false}
        />
      );

      expect(container).toBeDefined();
      expect(screen.queryByTestId("facecam-box")).toBeNull();
      expect(screen.queryByTestId("crop-window")).toBeNull();
      expect(screen.queryByTestId("caption-overlay")).toBeNull();
    });

    it("handles pointercancel event during drag by resetting active drag state", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.1, y: 0.1, w: 0.2, h: 0.2 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const interior = screen.getByTestId("facecam-drag-interior");

      // Start drag
      fireEvent.pointerDown(interior, { clientX: 100, clientY: 100, pointerId: 1 });
      fireEvent.pointerMove(interior, { clientX: 200, clientY: 200, pointerId: 1 });
      expect(onFacecamChange).toHaveBeenCalled();

      // Trigger pointercancel on window
      fireEvent(window, new Event("pointercancel"));

      // Subsequent pointermove should have NO effect because drag was cancelled
      onFacecamChange.mockClear();
      fireEvent.pointerMove(interior, { clientX: 300, clientY: 300, pointerId: 1 });
      expect(onFacecamChange).not.toHaveBeenCalled();
    });

    it("handles active word detection across empty caption spans and future timestamps", () => {
      const words: CaptionWord[] = [
        { t: 10.0, d: 2.0, text: "LATE_WORD_1", speaker: 1 },
        { t: 15.0, d: 1.0, text: "LATE_WORD_2", speaker: 2 },
      ];

      render(
        <InteractiveCanvas
          captionWords={words}
          currentTime={0.5} // Far before first word
          showCaptions={true}
        />
      );

      // Defaults to slice(0, 8) when no active word at timestamp
      expect(screen.getByTestId("caption-word-0").textContent).toBe("LATE_WORD_1");
    });

    it("property-based fuzzing: 1,000 random handle resizes always preserve [0, 1] bounds and min size bounds", () => {
      const handles: ResizeHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
      let seed = 42;
      const pseudoRandom = () => {
        seed = (seed * 16807) % 2147483647;
        return (seed - 1) / 2147483646;
      };

      for (let i = 0; i < 1000; i++) {
        const initial: Rect = {
          x: pseudoRandom() * 0.8,
          y: pseudoRandom() * 0.8,
          w: 0.05 + pseudoRandom() * 0.5,
          h: 0.05 + pseudoRandom() * 0.5,
        };
        const handle = handles[Math.floor(pseudoRandom() * handles.length)];
        const dx = (pseudoRandom() - 0.5) * 4; // [-2.0, 2.0]
        const dy = (pseudoRandom() - 0.5) * 4;

        const result = applyHandleResize(initial, handle, dx, dy);
        expect(result.w).toBeGreaterThanOrEqual(0.05);
        expect(result.h).toBeGreaterThanOrEqual(0.05);
        expect(result.x).toBeGreaterThanOrEqual(0.0);
        expect(result.y).toBeGreaterThanOrEqual(0.0);
        expect(result.x + result.w).toBeLessThanOrEqual(1.0 + 1e-5);
        expect(result.y + result.h).toBeLessThanOrEqual(1.0 + 1e-5);
        expect(Number.isFinite(result.x)).toBe(true);
        expect(Number.isFinite(result.y)).toBe(true);
        expect(Number.isFinite(result.w)).toBe(true);
        expect(Number.isFinite(result.h)).toBe(true);
      }
    });

    it("sliding window caption editing: editing word at high timestamp correctly updates global array index", () => {
      const onCaptionWordsChange = vi.fn();
      // Generate 25 words spanning 0s to 50s
      const longWords: CaptionWord[] = Array.from({ length: 25 }, (_, i) => ({
        t: i * 2.0,
        d: 1.5,
        text: `WORD_${i}`,
        speaker: (i % 2) + 1,
      }));

      render(
        <InteractiveCanvas
          captionWords={longWords}
          currentTime={30.2} // Word 15 is active (t=30.0, d=1.5)
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      // In sliding window around index 15 (s = 13, e = 19), word 15 should be present
      const word15El = screen.getByTestId("caption-word-15");
      expect(word15El.textContent).toBe("WORD_15");

      // Edit word 15
      fireEvent.click(word15El);
      const input15 = screen.getByTestId("caption-word-input-15") as HTMLInputElement;
      expect(input15.value).toBe("WORD_15");
      fireEvent.change(input15, { target: { value: "EDITED_WORD_15" } });
      fireEvent.keyDown(input15, { key: "Enter" });

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updatedWords = onCaptionWordsChange.mock.calls[0][0];
      expect(updatedWords.length).toBe(25);
      expect(updatedWords[15].text).toBe("EDITED_WORD_15");
      // Other words unmodified
      expect(updatedWords[0].text).toBe("WORD_0");
      expect(updatedWords[14].text).toBe("WORD_14");
      expect(updatedWords[16].text).toBe("WORD_16");
    });
  });
});

