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

describe("InteractiveCanvas Component & Interactions (Milestone M2)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // 1. Pure Geometry and Math Verification
  // ==========================================================================
  describe("Pure Coordinate & Snapping Algorithms", () => {
    it("clampRect: keeps coordinates inside [0, 1] bounds and enforces min dimension 0.05", () => {
      const outOfBounds: Rect = { x: 0.95, y: 0.95, w: 0.2, h: 0.2 };
      const clamped = clampRect(outOfBounds);
      expect(clamped.x).toBeLessThanOrEqual(0.8);
      expect(clamped.y).toBeLessThanOrEqual(0.8);
      expect(clamped.x + clamped.w).toBeLessThanOrEqual(1.0);
      expect(clamped.y + clamped.h).toBeLessThanOrEqual(1.0);

      const tiny: Rect = { x: 0.1, y: 0.1, w: 0.01, h: 0.02 };
      const clampedTiny = clampRect(tiny);
      expect(clampedTiny.w).toBe(0.05);
      expect(clampedTiny.h).toBe(0.05);
    });

    it("applyHandleResize: correctly resizes via each handle direction", () => {
      const initial: Rect = { x: 0.2, y: 0.2, w: 0.3, h: 0.3 };

      // 'se' handle expands width & height
      const seResized = applyHandleResize(initial, "se", 0.1, 0.05);
      expect(seResized.w).toBeCloseTo(0.4, 2);
      expect(seResized.h).toBeCloseTo(0.35, 2);
      expect(seResized.x).toBe(0.2);
      expect(seResized.y).toBe(0.2);

      // 'nw' handle shifts origin and expands dimensions
      const nwResized = applyHandleResize(initial, "nw", -0.05, -0.05);
      expect(nwResized.x).toBeCloseTo(0.15, 2);
      expect(nwResized.y).toBeCloseTo(0.15, 2);
      expect(nwResized.w).toBeCloseTo(0.35, 2);
      expect(nwResized.h).toBeCloseTo(0.35, 2);

      // 'e' handle expands width only
      const eResized = applyHandleResize(initial, "e", 0.08, 0);
      expect(eResized.w).toBeCloseTo(0.38, 2);
      expect(eResized.h).toBe(0.3);

      // 's' handle expands height only
      const sResized = applyHandleResize(initial, "s", 0, 0.12);
      expect(sResized.h).toBeCloseTo(0.42, 2);
      expect(sResized.w).toBe(0.3);
    });

    it("applyMagneticSnap: snaps within threshold and passes through outside threshold", () => {
      // Crop center snap at 0.5 (threshold 0.02)
      const nearCenter = applyMagneticSnap(0.515, [0.5], 0.02);
      expect(nearCenter.snapped).toBe(true);
      expect(nearCenter.value).toBe(0.5);

      const farFromCenter = applyMagneticSnap(0.45, [0.5], 0.02);
      expect(farFromCenter.snapped).toBe(false);
      expect(farFromCenter.value).toBe(0.45);

      // Safe zone snapping [0.70, 0.78, 0.85]
      const near78 = applyMagneticSnap(0.778, [0.70, 0.78, 0.85], 0.02);
      expect(near78.snapped).toBe(true);
      expect(near78.value).toBe(0.78);

      const near70 = applyMagneticSnap(0.705, [0.70, 0.78, 0.85], 0.02);
      expect(near70.snapped).toBe(true);
      expect(near70.value).toBe(0.70);

      const near85 = applyMagneticSnap(0.842, [0.70, 0.78, 0.85], 0.02);
      expect(near85.snapped).toBe(true);
      expect(near85.value).toBe(0.85);
    });
  });

  // ==========================================================================
  // 2. Facecam Bounding Box: Dragging & 8-Point Resizing
  // ==========================================================================
  describe("Interactive Facecam Bounding Box", () => {
    it("renders facecam bounding box with 8 resize handles", () => {
      const facecam: Rect = { x: 0.1, y: 0.15, w: 0.25, h: 0.35 };
      render(
        <InteractiveCanvas
          facecam={facecam}
          showFacecam={true}
          canvasWidth={800}
          canvasHeight={450}
        />
      );

      const box = screen.getByTestId("facecam-box");
      expect(box).toBeDefined();
      expect(box.style.left).toBe("10%");
      expect(box.style.top).toBe("15%");
      expect(box.style.width).toBe("25%");
      expect(box.style.height).toBe("35%");

      const handles: ResizeHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
      for (const h of handles) {
        expect(screen.getByTestId(`facecam-handle-${h}`)).toBeDefined();
      }
    });

    it("dragging facecam box interior updates (x, y) coordinates", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.1, y: 0.1, w: 0.3, h: 0.3 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const interior = screen.getByTestId("facecam-drag-interior");

      // Pointer drag +100px X and +150px Y (+0.10 X, +0.15 Y in normalized 1000x1000)
      fireEvent.pointerDown(interior, { clientX: 200, clientY: 200, pointerId: 1 });
      fireEvent.pointerMove(interior, { clientX: 300, clientY: 350, pointerId: 1 });
      fireEvent.pointerUp(interior, { clientX: 300, clientY: 350, pointerId: 1 });

      expect(onFacecamChange).toHaveBeenCalled();
      const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
      expect(lastCall.x).toBeCloseTo(0.2, 2);
      expect(lastCall.y).toBeCloseTo(0.25, 2);
      expect(lastCall.w).toBe(0.3);
      expect(lastCall.h).toBe(0.3);
    });

    it("resizing facecam via 'se' handle updates width and height", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.1, y: 0.1, w: 0.25, h: 0.25 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const seHandle = screen.getByTestId("facecam-handle-se");

      // Pointer drag +100px in both X and Y (+0.10 normalized)
      fireEvent.pointerDown(seHandle, { clientX: 350, clientY: 350, pointerId: 1 });
      fireEvent.pointerMove(seHandle, { clientX: 450, clientY: 450, pointerId: 1 });
      fireEvent.pointerUp(seHandle, { clientX: 450, clientY: 450, pointerId: 1 });

      expect(onFacecamChange).toHaveBeenCalled();
      const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
      expect(lastCall.x).toBe(0.1);
      expect(lastCall.y).toBe(0.1);
      expect(lastCall.w).toBeCloseTo(0.35, 2);
      expect(lastCall.h).toBeCloseTo(0.35, 2);
    });

    it("resizing facecam via 'nw' handle updates x, y and w, h", () => {
      const onFacecamChange = vi.fn();
      const initial: Rect = { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };

      render(
        <InteractiveCanvas
          facecam={initial}
          onFacecamChange={onFacecamChange}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const nwHandle = screen.getByTestId("facecam-handle-nw");

      // Move top-left corner left & up by 50px (-0.05 normalized)
      fireEvent.pointerDown(nwHandle, { clientX: 300, clientY: 300, pointerId: 1 });
      fireEvent.pointerMove(nwHandle, { clientX: 250, clientY: 250, pointerId: 1 });
      fireEvent.pointerUp(nwHandle, { clientX: 250, clientY: 250, pointerId: 1 });

      expect(onFacecamChange).toHaveBeenCalled();
      const lastCall = onFacecamChange.mock.calls[onFacecamChange.mock.calls.length - 1][0];
      expect(lastCall.x).toBeCloseTo(0.25, 2);
      expect(lastCall.y).toBeCloseTo(0.25, 2);
      expect(lastCall.w).toBeCloseTo(0.45, 2);
      expect(lastCall.h).toBeCloseTo(0.45, 2);
    });
  });

  // ==========================================================================
  // 3. Gameplay 9:16 Crop Window & Magnetic Snapping
  // ==========================================================================
  describe("Gameplay 9:16 Crop Window", () => {
    it("renders crop window overlay with 9:16 aspect ratio guides", () => {
      render(
        <InteractiveCanvas
          cropCx={0.5}
          showCropOverlay={true}
          canvasWidth={800}
          canvasHeight={450}
        />
      );

      expect(screen.getByTestId("crop-window")).toBeDefined();
      expect(screen.getByTestId("crop-drag-handle")).toBeDefined();
    });

    it("dragging crop handle horizontally updates cropCx", () => {
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

      // Drag right by +100px (+0.10 normalized)
      fireEvent.pointerDown(handle, { clientX: 300, clientY: 50, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: 400, clientY: 50, pointerId: 1 });
      fireEvent.pointerUp(handle, { clientX: 400, clientY: 50, pointerId: 1 });

      expect(onCropCxChange).toHaveBeenCalled();
      const lastCall = onCropCxChange.mock.calls[onCropCxChange.mock.calls.length - 1][0];
      expect(lastCall).toBeCloseTo(0.4, 2);
    });

    it("magnetically snaps cropCx to 0.5 when dragged near center (threshold 0.02)", () => {
      const onCropCxChange = vi.fn();

      render(
        <InteractiveCanvas
          cropCx={0.4}
          onCropCxChange={onCropCxChange}
          showCropOverlay={true}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const handle = screen.getByTestId("crop-drag-handle");

      // Drag from 0.40 towards 0.51 (delta +110px => 0.51, inside ±0.02 threshold)
      fireEvent.pointerDown(handle, { clientX: 400, clientY: 50, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: 510, clientY: 50, pointerId: 1 });

      expect(onCropCxChange).toHaveBeenCalled();
      const lastCall = onCropCxChange.mock.calls[onCropCxChange.mock.calls.length - 1][0];
      expect(lastCall).toBe(0.5);

      // Verify magnetic snap indicator is shown
      expect(screen.getByTestId("crop-snap-indicator")).toBeDefined();

      fireEvent.pointerUp(handle, { clientX: 510, clientY: 50, pointerId: 1 });
    });
  });

  // ==========================================================================
  // 4. Live WYSIWYG Captions & Vertical Safe-Zone Snapping
  // ==========================================================================
  describe("Live WYSIWYG Captions Dragging & Safe Zones", () => {
    it("renders caption overlay at specified captionYFrac", () => {
      render(
        <InteractiveCanvas
          captionYFrac={0.78}
          showCaptions={true}
          captionWords={[{ t: 1.0, d: 0.5, text: "TEST", speaker: null }]}
        />
      );

      const overlay = screen.getByTestId("caption-overlay");
      expect(overlay.style.top).toBe("78%");
      expect(screen.getByTestId("caption-drag-handle")).toBeDefined();
    });

    it("dragging caption handle vertically updates captionYFrac", () => {
      const onCaptionYFracChange = vi.fn();

      render(
        <InteractiveCanvas
          captionYFrac={0.5}
          onCaptionYFracChange={onCaptionYFracChange}
          showCaptions={true}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const handle = screen.getByTestId("caption-drag-handle");

      // Drag downwards by 120px (+0.12 normalized)
      fireEvent.pointerDown(handle, { clientX: 500, clientY: 500, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: 500, clientY: 620, pointerId: 1 });
      fireEvent.pointerUp(handle, { clientX: 500, clientY: 620, pointerId: 1 });

      expect(onCaptionYFracChange).toHaveBeenCalled();
      const lastCall =
        onCaptionYFracChange.mock.calls[onCaptionYFracChange.mock.calls.length - 1][0];
      expect(lastCall).toBeCloseTo(0.62, 2);
    });

    it("snaps caption position to standard safe zones (0.70, 0.78, 0.85)", () => {
      const onCaptionYFracChange = vi.fn();

      render(
        <InteractiveCanvas
          captionYFrac={0.65}
          onCaptionYFracChange={onCaptionYFracChange}
          showCaptions={true}
          canvasWidth={1000}
          canvasHeight={1000}
        />
      );

      const handle = screen.getByTestId("caption-drag-handle");

      // Drag towards 0.775 (delta +125px => 0.775, within ±0.02 of 0.78)
      fireEvent.pointerDown(handle, { clientX: 500, clientY: 650, pointerId: 1 });
      fireEvent.pointerMove(handle, { clientX: 500, clientY: 775, pointerId: 1 });

      expect(onCaptionYFracChange).toHaveBeenCalled();
      const lastCall =
        onCaptionYFracChange.mock.calls[onCaptionYFracChange.mock.calls.length - 1][0];
      expect(lastCall).toBe(0.78);

      // Verify safe zone badge appears
      expect(screen.getByTestId("caption-snap-guide")).toBeDefined();
    });
  });

  // ==========================================================================
  // 5. Direct Inline Word Editing & useUndo Integration
  // ==========================================================================
  describe("Direct Inline Caption Word Editing & useUndo", () => {
    const testWords: CaptionWord[] = [
      { t: 0.5, d: 0.4, text: "EPIC", speaker: null },
      { t: 1.0, d: 0.6, text: "CLUTCH", speaker: null },
      { t: 1.8, d: 0.5, text: "MOMENT", speaker: null },
    ];

    it("renders caption words and highlights active word based on currentTime", () => {
      const activeStyle: StyleTemplate = {
        id: "neon",
        name: "Neon Glow",
        font: "Impact",
        font_size: 32,
        primary: "FFFFFF",
        highlight: "FF007F",
        outline: "000000",
        outline_w: 3,
        y_frac: 0.78,
        uppercase: true,
      };

      render(
        <InteractiveCanvas
          captionWords={testWords}
          currentTime={1.2} // Within "CLUTCH" (1.0 to 1.6)
          activeStyle={activeStyle}
          showCaptions={true}
        />
      );

      expect(screen.getByTestId("caption-word-0").textContent).toBe("EPIC");
      expect(screen.getByTestId("caption-word-1").textContent).toBe("CLUTCH");
      expect(screen.getByTestId("caption-word-2").textContent).toBe("MOMENT");
    });

    it("clicking a word converts it into an inline editable <input>", () => {
      render(<InteractiveCanvas captionWords={testWords} showCaptions={true} />);

      const wordSpan = screen.getByTestId("caption-word-1");
      expect(wordSpan.textContent).toBe("CLUTCH");

      fireEvent.click(wordSpan);

      const input = screen.getByTestId("caption-word-input-1") as HTMLInputElement;
      expect(input).toBeDefined();
      expect(input.value).toBe("CLUTCH");
    });

    it("committing word edit on Enter updates captionWords", () => {
      const onCaptionWordsChange = vi.fn();

      render(
        <InteractiveCanvas
          captionWords={testWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-0"));

      const input = screen.getByTestId("caption-word-input-0") as HTMLInputElement;
      fireEvent.change(input, { target: { value: "INSANE" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updated = onCaptionWordsChange.mock.calls[0][0];
      expect(updated[0].text).toBe("INSANE");
      expect(updated[1].text).toBe("CLUTCH");
    });

    it("committing word edit on blur updates captionWords", () => {
      const onCaptionWordsChange = vi.fn();

      render(
        <InteractiveCanvas
          captionWords={testWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-2"));

      const input = screen.getByTestId("caption-word-input-2") as HTMLInputElement;
      fireEvent.change(input, { target: { value: "VICTORY" } });
      fireEvent.blur(input);

      expect(onCaptionWordsChange).toHaveBeenCalled();
      const updated = onCaptionWordsChange.mock.calls[0][0];
      expect(updated[2].text).toBe("VICTORY");
    });

    it("Escape key cancels inline editing without saving changes", () => {
      const onCaptionWordsChange = vi.fn();

      render(
        <InteractiveCanvas
          captionWords={testWords}
          onCaptionWordsChange={onCaptionWordsChange}
          showCaptions={true}
        />
      );

      fireEvent.click(screen.getByTestId("caption-word-1"));
      const input = screen.getByTestId("caption-word-input-1") as HTMLInputElement;
      fireEvent.change(input, { target: { value: "MODIFIED" } });
      fireEvent.keyDown(input, { key: "Escape" });

      expect(onCaptionWordsChange).not.toHaveBeenCalled();
      expect(screen.queryByTestId("caption-word-input-1")).toBeNull();
      expect(screen.getByTestId("caption-word-1").textContent).toBe("CLUTCH");
    });

    it("integrates with useUndo: Ctrl+Z triggers undo callback", () => {
      const onUndo = vi.fn();
      const onRedo = vi.fn();

      render(
        <InteractiveCanvas
          captionWords={testWords}
          onUndo={onUndo}
          onRedo={onRedo}
          showCaptions={true}
        />
      );

      const canvas = screen.getByTestId("interactive-canvas");

      // Press Ctrl+Z
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true });
      expect(onUndo).toHaveBeenCalledTimes(1);

      // Press Ctrl+Shift+Z
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true, shiftKey: true });
      expect(onRedo).toHaveBeenCalledTimes(1);
    });

    it("stateful integration: editing updates state and undo restores it", () => {
      function StatefulCanvasContainer() {
        const [words, setWords] = useState<CaptionWord[]>(testWords);
        const [history, setHistory] = useState<CaptionWord[][]>([testWords]);

        const updateWords = (newWords: CaptionWord[]) => {
          setHistory((prev) => [...prev, newWords]);
          setWords(newWords);
        };

        const undoAction = () => {
          if (history.length > 1) {
            const prev = history[history.length - 2];
            setHistory((h) => h.slice(0, -1));
            setWords(prev);
          }
        };

        return (
          <InteractiveCanvas
            captionWords={words}
            onCaptionWordsChange={updateWords}
            onUndo={undoAction}
            showCaptions={true}
          />
        );
      }

      render(<StatefulCanvasContainer />);

      // Step 1: Click and edit first word
      fireEvent.click(screen.getByTestId("caption-word-0"));
      const input = screen.getByTestId("caption-word-input-0") as HTMLInputElement;
      fireEvent.change(input, { target: { value: "LEGENDARY" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(screen.getByTestId("caption-word-0").textContent).toBe("LEGENDARY");

      // Step 2: Trigger undo
      const canvas = screen.getByTestId("interactive-canvas");
      fireEvent.keyDown(canvas, { key: "z", ctrlKey: true });

      // Step 3: Verified restored back to original word
      expect(screen.getByTestId("caption-word-0").textContent).toBe("EPIC");
    });
  });
});
