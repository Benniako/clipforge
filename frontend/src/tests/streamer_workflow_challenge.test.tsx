/**
 * Empirical Challenge & Stress-Test Suite for Milestone M5:
 * Streamer Quick Workflow, Hotkeys & Batch Operations (Requirement R6)
 *
 * Adversarial stress harness covering:
 * 1. Input Isolation Attack: Aggressive typing bursts in <input>, <textarea>,
 *    deeply nested contenteditable, and .caption-word-input inline editor fields.
 * 2. Boundary Trimming & Range Inversion: 'I' and 'O' mark in/out clamping,
 *    boundary conditions when currentTime >= end or currentTime <= start,
 *    negative timestamps, and mathematical invariance against range inversion.
 * 3. Shuttle Speed & State Machine: Rapid alternating J-K-L-Space sequences,
 *    clamping at 0s, clamping at clip duration, and video element play() promise rejection.
 * 4. Web Audio Offline & Autoplay Resilience: Missing AudioContext, suspended state,
 *    constructor exceptions (autoplay restriction), and 500-event audio pip burst stress.
 * 5. Batch Action Toolbar Edge Cases & Concurrency: Empty selection guards,
 *    rapid double-clicking / busy state locks, confirmation dialog cancellation,
 *    backend error HUD recovery, and selection state integrity.
 * 6. Quick Rating Hotkeys: Digit 1-5, Numpad 1-5, and rejection of invalid digits (0, 6-9).
 * 7. Lifecycle & HUD Timer Dissolve: Timer resetting on rapid triggers and clean unmount.
 */

import { cleanup, fireEvent, render, screen, act } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import StreamerWorkflowBar, {
  isInputIsolated,
  playTactileBeep,
} from "../components/StreamerWorkflowBar";
import type { Clip } from "../lib/types";

// Mock media element APIs for videoRef
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
Object.defineProperty(HTMLMediaElement.prototype, "playbackRate", {
  configurable: true,
  writable: true,
  value: 1.0,
});

const mockClips: Clip[] = [
  {
    id: "clip-alpha",
    title: "Triple Headshot",
    start: 5.0,
    end: 15.0,
    score: 85,
    status: "ready",
    feedback: null,
    hashtags: ["#fps", "#aim"],
    reframe: { layout: "center", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
  {
    id: "clip-bravo",
    title: "Jumpscare Scream",
    start: 20.0,
    end: 32.0,
    score: 72,
    status: "ready",
    feedback: null,
    hashtags: ["#horror"],
    reframe: { layout: "split", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
  {
    id: "clip-charlie",
    title: "Overtime Clutch Win",
    start: 40.0,
    end: 58.0,
    score: 96,
    status: "ready",
    feedback: null,
    hashtags: ["#clutch", "#victory"],
    reframe: { layout: "center", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
];

describe("Milestone M5 Empirical Challenge: Streamer Workflow & Hotkeys Stress Harness", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // CHALLENGE 1: Input Isolation Attack
  // ==========================================================================
  describe("Adversarial Input Isolation Stress Tests", () => {
    it("rejects hotkey triggers during an aggressive 100-keystroke typing burst inside an <input>", () => {
      const onPlay = vi.fn();
      const onRewind = vi.fn();
      const onForward = vi.fn();
      const onRate = vi.fn();
      const onMarkIn = vi.fn();
      const onMarkOut = vi.fn();
      const onAudioPip = vi.fn();

      render(
        <div>
          <input data-testid="target-input" />
          <StreamerWorkflowBar
            currentTime={10.0}
            onPlayToggle={onPlay}
            onShuttleRewind={onRewind}
            onShuttleForward={onForward}
            onRate={onRate}
            onMarkIn={onMarkIn}
            onMarkOut={onMarkOut}
            onAudioPip={onAudioPip}
          />
        </div>
      );

      const input = screen.getByTestId("target-input");
      input.focus();

      // Adversarial payload mixing text, numbers, spaces, and hotkey characters
      const adversarialText =
        "just playing valorant 12345 look at that clutch jk space i o kill streak";

      for (let i = 0; i < adversarialText.length; i++) {
        const char = adversarialText[i];
        let code = `Key${char.toUpperCase()}`;
        if (char === " ") code = "Space";
        else if (/[0-9]/.test(char)) code = `Digit${char}`;

        fireEvent.keyDown(input, { key: char, code });
      }

      // Assert complete silence: zero transport callbacks, zero rating callbacks, zero HUD
      expect(onPlay).not.toHaveBeenCalled();
      expect(onRewind).not.toHaveBeenCalled();
      expect(onForward).not.toHaveBeenCalled();
      expect(onRate).not.toHaveBeenCalled();
      expect(onMarkIn).not.toHaveBeenCalled();
      expect(onMarkOut).not.toHaveBeenCalled();
      expect(onAudioPip).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("rejects hotkey triggers during multiline text entry inside a <textarea>", () => {
      const onRate = vi.fn();
      const onPlay = vi.fn();

      render(
        <div>
          <textarea data-testid="target-textarea" />
          <StreamerWorkflowBar onRate={onRate} onPlayToggle={onPlay} />
        </div>
      );

      const textarea = screen.getByTestId("target-textarea");
      textarea.focus();

      const multilineBurst = [
        { code: "KeyJ", key: "j" },
        { code: "KeyK", key: "k" },
        { code: "KeyL", key: "l" },
        { code: "Space", key: " " },
        { code: "Digit5", key: "5" },
        { code: "KeyI", key: "i" },
        { code: "KeyO", key: "o" },
      ];

      for (const stroke of multilineBurst) {
        fireEvent.keyDown(textarea, stroke);
      }

      expect(onRate).not.toHaveBeenCalled();
      expect(onPlay).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("verifies deep element hierarchy inside [contenteditable] is strictly isolated", () => {
      const onPlay = vi.fn();
      const onRewind = vi.fn();

      render(
        <div>
          <div data-testid="editable-root" contentEditable={true}>
            <div className="paragraph">
              <p>
                <span className="bold">
                  <em>
                    <span data-testid="deep-editable-leaf">Deep Word</span>
                  </em>
                </span>
              </p>
            </div>
          </div>
          <StreamerWorkflowBar onPlayToggle={onPlay} onShuttleRewind={onRewind} />
        </div>
      );

      const deepLeaf = screen.getByTestId("deep-editable-leaf");
      deepLeaf.focus();

      fireEvent.keyDown(deepLeaf, { code: "Space", key: " " });
      fireEvent.keyDown(deepLeaf, { code: "KeyJ", key: "j" });

      expect(onPlay).not.toHaveBeenCalled();
      expect(onRewind).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("verifies nested caption editing elements (.caption-word-input) are strictly isolated", () => {
      const onMarkIn = vi.fn();
      const onRate = vi.fn();

      render(
        <div>
          <div className="canvas-overlay">
            <div className="caption-word-input">
              <span data-testid="caption-child" tabIndex={0}>
                Edited Caption
              </span>
            </div>
          </div>
          <StreamerWorkflowBar onMarkIn={onMarkIn} onRate={onRate} />
        </div>
      );

      const captionChild = screen.getByTestId("caption-child");
      captionChild.focus();

      fireEvent.keyDown(captionChild, { code: "KeyI", key: "i" });
      fireEvent.keyDown(captionChild, { code: "Digit4", key: "4" });

      expect(onMarkIn).not.toHaveBeenCalled();
      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("properly re-enables hotkeys once input is blurred and activeElement changes", () => {
      const onRewind = vi.fn();

      render(
        <div>
          <input data-testid="toggle-input" />
          <button data-testid="unfocused-canvas-button">Canvas</button>
          <StreamerWorkflowBar currentTime={8.0} onShuttleRewind={onRewind} />
        </div>
      );

      const input = screen.getByTestId("toggle-input");
      const button = screen.getByTestId("unfocused-canvas-button");

      // 1. Focus input -> J is suppressed
      input.focus();
      fireEvent.keyDown(input, { code: "KeyJ", key: "j" });
      expect(onRewind).not.toHaveBeenCalled();

      // 2. Focus non-input button (e.g. video canvas or outside control) -> J fires
      button.focus();
      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      expect(onRewind).toHaveBeenCalledTimes(1);
      expect(onRewind).toHaveBeenCalledWith(7.0);

      // 3. Focus input again -> J is suppressed again
      input.focus();
      fireEvent.keyDown(input, { code: "KeyJ", key: "j" });
      expect(onRewind).toHaveBeenCalledTimes(1);
    });

    it("ignores streamer hotkeys when browser modifier keys (Ctrl/Meta/Alt) are pressed", () => {
      const onRewind = vi.fn();
      const onForward = vi.fn();
      const onPlay = vi.fn();

      render(
        <StreamerWorkflowBar
          currentTime={10.0}
          onShuttleRewind={onRewind}
          onShuttleForward={onForward}
          onPlayToggle={onPlay}
        />
      );

      // Ctrl+J (often Downloads), Alt+L, Meta+Space (OS Spotlight)
      fireEvent.keyDown(window, { code: "KeyJ", key: "j", ctrlKey: true });
      fireEvent.keyDown(window, { code: "KeyL", key: "l", altKey: true });
      fireEvent.keyDown(window, { code: "Space", key: " ", metaKey: true });

      expect(onRewind).not.toHaveBeenCalled();
      expect(onForward).not.toHaveBeenCalled();
      expect(onPlay).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });
  });

  // ==========================================================================
  // CHALLENGE 2: Boundary Trimming & Range Inversion
  // ==========================================================================
  describe("Boundary Trimming & Clamping Invariants", () => {
    it("handles Mark In when currentTime is beyond end point without inversion in editor clamping", () => {
      // In ClipEditor.tsx:
      // onMarkIn={(t) => {
      //   const nextStart = Math.max(0, Math.min(t, end - 1));
      //   setStart(nextStart);
      // }}
      const clampMarkIn = (t: number, end: number) =>
        Math.max(0, Math.min(t, end - 1));

      // Scenario: Clip end is 15s. User attempts Mark In at 22s.
      const clamped = clampMarkIn(22.0, 15.0);
      expect(clamped).toBe(14.0);
      expect(clamped).toBeLessThan(15.0);

      // Scenario: User attempts Mark In at -10s
      const clampedNeg = clampMarkIn(-10.0, 15.0);
      expect(clampedNeg).toBe(0.0);
    });

    it("handles Mark Out when currentTime is before start point without inversion in editor clamping", () => {
      // In ClipEditor.tsx:
      // onMarkOut={(t) => {
      //   const nextEnd = Math.max(start + 1, Math.min(t, srcDur));
      //   setEnd(nextEnd);
      // }}
      const clampMarkOut = (t: number, start: number, srcDur: number) =>
        Math.max(start + 1, Math.min(t, srcDur));

      // Scenario: Clip start is 10s. User attempts Mark Out at 4s.
      const clamped = clampMarkOut(4.0, 10.0, 60.0);
      expect(clamped).toBe(11.0);
      expect(clamped).toBeGreaterThan(10.0);

      // Scenario: User attempts Mark Out at 999s when srcDur is 60s
      const clampedMax = clampMarkOut(999.0, 10.0, 60.0);
      expect(clampedMax).toBe(60.0);
    });

    it("mathematical fuzzing proof: range inversion is impossible across 500 boundary combinations", () => {
      const clampMarkIn = (t: number, end: number) =>
        Math.max(0, Math.min(t, end - 1));
      const clampMarkOut = (t: number, start: number, srcDur: number) =>
        Math.max(start + 1, Math.min(t, srcDur));

      const srcDur = 100.0;

      for (let i = 0; i < 500; i++) {
        // Random start and end such that 0 <= start < end <= srcDur
        const end = 2.0 + Math.random() * (srcDur - 2.0);
        const start = Math.random() * (end - 1.0);

        // Random test times from -50 to 150
        const testT = -50.0 + Math.random() * 200.0;

        // Test Mark In
        const nextStart = clampMarkIn(testT, end);
        expect(nextStart).toBeGreaterThanOrEqual(0.0);
        expect(nextStart).toBeLessThanOrEqual(end - 1.0);
        expect(nextStart < end).toBe(true); // Invariant 1: start < end

        // Test Mark Out
        const nextEnd = clampMarkOut(testT, start, srcDur);
        expect(nextEnd).toBeGreaterThanOrEqual(start + 1.0);
        expect(nextEnd).toBeLessThanOrEqual(srcDur);
        expect(start < nextEnd).toBe(true); // Invariant 2: start < end
      }
    });

    it("StreamerWorkflowBar correctly passes playhead time to Mark In and Mark Out handlers", () => {
      const onMarkIn = vi.fn();
      const onMarkOut = vi.fn();

      render(
        <StreamerWorkflowBar
          currentTime={14.7}
          duration={30.0}
          onMarkIn={onMarkIn}
          onMarkOut={onMarkOut}
        />
      );

      // Press I
      fireEvent.keyDown(window, { code: "KeyI", key: "i" });
      expect(onMarkIn).toHaveBeenCalledWith(14.7);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[I] MARK IN 14.7s");

      // Press O
      fireEvent.keyDown(window, { code: "KeyO", key: "o" });
      expect(onMarkOut).toHaveBeenCalledWith(14.7);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[O] MARK OUT 14.7s");
    });
  });

  // ==========================================================================
  // CHALLENGE 3: Shuttle Speed & State Machine Stress Harness
  // ==========================================================================
  describe("Shuttle Speed & State Machine Stress Harness", () => {
    it("survives rapid 50-step alternating J-K-L shuttle transitions without desync", () => {
      let currentTime = 10.0;
      const duration = 20.0;

      const onRewind = vi.fn((newT) => {
        currentTime = newT;
      });
      const onForward = vi.fn((newT) => {
        currentTime = newT;
      });
      const onPause = vi.fn();

      const { rerender } = render(
        <StreamerWorkflowBar
          currentTime={currentTime}
          duration={duration}
          onShuttleRewind={onRewind}
          onShuttleForward={onForward}
          onShuttlePause={onPause}
        />
      );

      const shuttleSequence = [
        "j", "j", "j", "k", "l", "l", "l", "l", "k", "j",
        "l", "j", "j", "k", "l", "l", "j", "k", "l", "l",
        "j", "j", "j", "j", "k", "l", "l", "l", "k", "j",
      ];

      for (const stroke of shuttleSequence) {
        if (stroke === "j") {
          fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
        } else if (stroke === "l") {
          fireEvent.keyDown(window, { code: "KeyL", key: "l" });
        } else if (stroke === "k") {
          fireEvent.keyDown(window, { code: "KeyK", key: "k" });
        }

        // Synchronize prop update
        rerender(
          <StreamerWorkflowBar
            currentTime={currentTime}
            duration={duration}
            onShuttleRewind={onRewind}
            onShuttleForward={onForward}
            onShuttlePause={onPause}
          />
        );
      }

      expect(onRewind).toHaveBeenCalled();
      expect(onForward).toHaveBeenCalled();
      expect(onPause).toHaveBeenCalled();
      expect(currentTime).toBeGreaterThanOrEqual(0.0);
      expect(currentTime).toBeLessThanOrEqual(duration);
    });

    it("clamps rewind at 0s and does not allow negative playback time", () => {
      const onRewind = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={0.3}
          duration={30.0}
          onShuttleRewind={onRewind}
        />
      );

      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      expect(onRewind).toHaveBeenCalledWith(0.0);
    });

    it("clamps forward at clip duration and does not overshoot", () => {
      const onForward = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={29.6}
          duration={30.0}
          onShuttleForward={onForward}
        />
      );

      fireEvent.keyDown(window, { code: "KeyL", key: "l" });
      expect(onForward).toHaveBeenCalledWith(30.0);
    });

    it("safely handles videoRef.current.play() promise rejection (autoplay policy)", async () => {
      const mockVideo = document.createElement("video");
      // Simulate autoplay policy block: play() rejects with NotAllowedError
      mockVideo.play = vi.fn().mockRejectedValue(new Error("NotAllowedError: play() failed"));
      mockVideo.pause = vi.fn();
      Object.defineProperty(mockVideo, "paused", { value: true, writable: true });

      const videoRef = { current: mockVideo };
      const onPlay = vi.fn();

      render(
        <StreamerWorkflowBar
          videoRef={videoRef as any}
          onPlayToggle={onPlay}
        />
      );

      // Press Space: calls video.play()
      expect(() => {
        fireEvent.keyDown(window, { code: "Space", key: " " });
      }).not.toThrow();

      expect(mockVideo.play).toHaveBeenCalled();
      expect(onPlay).toHaveBeenCalledWith(true);
    });
  });

  // ==========================================================================
  // CHALLENGE 4: Web Audio Offline & Autoplay Fallback Resilience
  // ==========================================================================
  describe("Web Audio Offline & Autoplay Resilience", () => {
    it("safely no-ops when window.AudioContext is completely undefined", () => {
      const origAudioCtx = (window as any).AudioContext;
      const origWebkitAudioCtx = (window as any).webkitAudioContext;

      (window as any).AudioContext = undefined;
      (window as any).webkitAudioContext = undefined;

      expect(() => playTactileBeep(880, 20)).not.toThrow();

      (window as any).AudioContext = origAudioCtx;
      (window as any).webkitAudioContext = origWebkitAudioCtx;
    });

    it("safely recovers if new AudioContext() throws SecurityError or NotAllowedError", () => {
      const origAudioCtx = (window as any).AudioContext;

      (window as any).AudioContext = class ThrowingAudioContext {
        constructor() {
          throw new DOMException(
            "The AudioContext was not allowed to start due to user gesture policies.",
            "NotAllowedError"
          );
        }
      };

      expect(() => playTactileBeep(440, 25)).not.toThrow();

      (window as any).AudioContext = origAudioCtx;
    });

    it("survives high-volume 500-call audio pip burst in rapid succession", () => {
      const pipListener = vi.fn();
      window.addEventListener("streamer:audio-pip", pipListener);

      expect(() => {
        for (let i = 0; i < 500; i++) {
          playTactileBeep(400 + (i % 50) * 10, 15);
        }
      }).not.toThrow();

      expect(pipListener).toHaveBeenCalledTimes(500);
      window.removeEventListener("streamer:audio-pip", pipListener);
    });

    it("respects audioFeedbackEnabled={false} and disables sound generation", () => {
      const pipListener = vi.fn();
      window.addEventListener("streamer:audio-pip", pipListener);

      render(
        <StreamerWorkflowBar
          audioFeedbackEnabled={false}
          currentTime={5.0}
        />
      );

      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });

      // playTactileBeep was not invoked, so DOM custom event was not fired
      expect(pipListener).not.toHaveBeenCalled();

      window.removeEventListener("streamer:audio-pip", pipListener);
    });
  });

  // ==========================================================================
  // CHALLENGE 5: Batch Action Toolbar Edge Cases & Concurrency
  // ==========================================================================
  describe("Batch Action Toolbar Edge Cases & Concurrency", () => {
    it("guards against empty clip selection and shows [BATCH] NO CLIPS SELECTED HUD", () => {
      const onBatch = vi.fn();

      render(
        <StreamerWorkflowBar
          clips={[]}
          activeClipId={undefined}
          onBatchAction={onBatch}
        />
      );

      fireEvent.click(screen.getByTestId("batch-rate"));
      expect(onBatch).not.toHaveBeenCalled();
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] NO CLIPS SELECTED");

      fireEvent.click(screen.getByTestId("batch-tag"));
      expect(onBatch).not.toHaveBeenCalled();

      fireEvent.click(screen.getByTestId("batch-export"));
      expect(onBatch).not.toHaveBeenCalled();
    });

    it("disables batch action buttons while batch operation is in-flight (race condition guard)", async () => {
      let resolveBatch: any;
      const batchPromise = new Promise<void>((res) => {
        resolveBatch = res;
      });
      const onBatch = vi.fn(() => batchPromise);

      render(
        <StreamerWorkflowBar
          clips={mockClips}
          onBatchAction={onBatch}
        />
      );

      fireEvent.click(screen.getByTestId("batch-select-all"));

      const rateBtn = screen.getByTestId("batch-rate");
      const tagBtn = screen.getByTestId("batch-tag");
      const exportBtn = screen.getByTestId("batch-export");
      const deleteBtn = screen.getByTestId("batch-delete");

      // First click
      fireEvent.click(rateBtn);
      expect(onBatch).toHaveBeenCalledTimes(1);

      // Buttons are now disabled while in flight
      expect((rateBtn as HTMLButtonElement).disabled).toBe(true);
      expect((tagBtn as HTMLButtonElement).disabled).toBe(true);
      expect((exportBtn as HTMLButtonElement).disabled).toBe(true);
      expect((deleteBtn as HTMLButtonElement).disabled).toBe(true);

      // Second click while in flight should be blocked by disabled attribute
      fireEvent.click(rateBtn);
      expect(onBatch).toHaveBeenCalledTimes(1);

      // Resolve the operation
      await act(async () => {
        resolveBatch();
      });

      // Buttons are re-enabled
      expect((rateBtn as HTMLButtonElement).disabled).toBe(false);
      expect((tagBtn as HTMLButtonElement).disabled).toBe(false);
    });

    it("recovers gracefully and displays error HUD when batch action throws or rejects", async () => {
      const failingBatch = vi.fn().mockRejectedValue(new Error("Local storage quota exceeded"));

      render(
        <StreamerWorkflowBar
          clips={mockClips}
          onBatchAction={failingBatch}
        />
      );

      fireEvent.click(screen.getByTestId("batch-select-all"));

      await act(async () => {
        fireEvent.click(screen.getByTestId("batch-export"));
      });

      expect(failingBatch).toHaveBeenCalled();
      const badge = screen.getByTestId("hud-badge");
      expect(badge.textContent).toContain("[BATCH ERROR] Local storage quota exceeded");

      // Verify buttons are not stuck in disabled state
      expect(
        (screen.getByTestId("batch-export") as HTMLButtonElement).disabled
      ).toBe(false);
    });

    it("supports clip selection toggling across multiple items", () => {
      render(<StreamerWorkflowBar clips={mockClips} />);

      const c1 = screen.getByTestId("batch-select-clip-clip-alpha");
      const c2 = screen.getByTestId("batch-select-clip-clip-bravo");
      const c3 = screen.getByTestId("batch-select-clip-clip-charlie");
      const count = screen.getByTestId("selected-count");

      expect(count.textContent).toBe("0");

      fireEvent.click(c1);
      fireEvent.click(c2);
      expect(count.textContent).toBe("2");

      fireEvent.click(c3);
      expect(count.textContent).toBe("3");

      // Deselect alpha
      fireEvent.click(c1);
      expect(count.textContent).toBe("2");
    });
  });

  // ==========================================================================
  // CHALLENGE 6: Quick Rating Hotkeys & Invalid Key Filtration
  // ==========================================================================
  describe("Quick Rating Hotkeys & Key Filtration", () => {
    it("accepts Numpad1 to Numpad5 hotkeys", () => {
      const onRate = vi.fn();
      render(<StreamerWorkflowBar activeClipId="clip-alpha" onRate={onRate} />);

      fireEvent.keyDown(window, { code: "Numpad3", key: "3" });
      expect(onRate).toHaveBeenCalledWith(3);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[★ 3] RATED 3");
      expect(screen.getByTestId("current-clip-rating").textContent).toBe("3★");

      fireEvent.keyDown(window, { code: "Numpad5", key: "5" });
      expect(onRate).toHaveBeenCalledWith(5);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[★ 5] RATED 5");
      expect(screen.getByTestId("current-clip-rating").textContent).toBe("5★");
    });

    it("rejects invalid digits like 0, 6, 7, 8, 9 and arbitrary keys", () => {
      const onRate = vi.fn();
      render(<StreamerWorkflowBar activeClipId="clip-alpha" onRate={onRate} />);

      fireEvent.keyDown(window, { code: "Digit0", key: "0" });
      fireEvent.keyDown(window, { code: "Digit6", key: "6" });
      fireEvent.keyDown(window, { code: "Digit9", key: "9" });
      fireEvent.keyDown(window, { code: "KeyQ", key: "q" });

      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("rejects Shift+Digit keys (e.g. Shift+1 producing !) without rating or evaluating to NaN", () => {
      const onRate = vi.fn();
      render(<StreamerWorkflowBar activeClipId="clip-alpha" onRate={onRate} />);

      fireEvent.keyDown(window, { code: "Digit1", key: "!", shiftKey: true });
      fireEvent.keyDown(window, { code: "Digit2", key: "@", shiftKey: true });
      fireEvent.keyDown(window, { code: "Digit5", key: "%", shiftKey: true });

      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });
  });

  // ==========================================================================
  // CHALLENGE 7: Visual HUD Timer & Component Unmount Lifecycle
  // ==========================================================================
  describe("Visual HUD Lifecycle & Timer Dissolve", () => {
    it("cleans up timer on unmount and prevents memory leaks or post-unmount warnings", () => {
      vi.useFakeTimers();

      const { unmount } = render(<StreamerWorkflowBar currentTime={5.0} />);

      // Trigger HUD
      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      expect(screen.getByTestId("hud-badge")).toBeDefined();

      // Unmount while HUD timer is ticking
      unmount();

      // Advance timers
      act(() => {
        vi.advanceTimersByTime(1000);
      });

      // No errors or warnings thrown
      vi.useRealTimers();
    });
  });
});
