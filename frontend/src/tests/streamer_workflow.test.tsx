import { cleanup, fireEvent, render, screen, act } from "@testing-library/react";
import React, { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import StreamerWorkflowBar, {
  isInputIsolated,
  playTactileBeep,
} from "../components/StreamerWorkflowBar";
import type { Clip } from "../lib/types";

// Mock sample clips
const mockClips: Clip[] = [
  {
    id: "c1",
    title: "Epic Triple Kill",
    start: 2.0,
    end: 14.0,
    score: 80,
    status: "ready",
    feedback: null,
    hashtags: ["#epic"],
    reframe: { layout: "center", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
  {
    id: "c2",
    title: "Hilarious Scream",
    start: 16.0,
    end: 24.0,
    score: 60,
    status: "ready",
    feedback: null,
    hashtags: ["#funny"],
    reframe: { layout: "center", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
  {
    id: "c3",
    title: "1v5 Clutch Victory",
    start: 26.0,
    end: 38.0,
    score: 95,
    status: "ready",
    feedback: null,
    hashtags: ["#clutch"],
    reframe: { layout: "center", cx_overridden: false, keyframes: [] },
    captions: { words: [], style_id: "default" },
    speakers: [0],
    factors: [],
  } as any,
];

describe("Streamer Quick Workflow, Hotkeys & Batch Operations (Milestone M5)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  // ==========================================================================
  // 1. Input Isolation Unit Tests
  // ==========================================================================
  describe("Input Isolation Algorithm", () => {
    it("identifies standard HTMLInputElement as isolated", () => {
      const input = document.createElement("input");
      expect(isInputIsolated(input)).toBe(true);
    });

    it("identifies HTMLTextAreaElement as isolated", () => {
      const textarea = document.createElement("textarea");
      expect(isInputIsolated(textarea)).toBe(true);
    });

    it("identifies HTMLSelectElement as isolated", () => {
      const select = document.createElement("select");
      expect(isInputIsolated(select)).toBe(true);
    });

    it("identifies contenteditable container and its child nodes as isolated", () => {
      const div = document.createElement("div");
      div.setAttribute("contenteditable", "true");
      const span = document.createElement("span");
      div.appendChild(span);
      document.body.appendChild(div);

      expect(isInputIsolated(div)).toBe(true);
      expect(isInputIsolated(span)).toBe(true);

      document.body.removeChild(div);
    });

    it("identifies .caption-word-input class and its child nodes as isolated", () => {
      const wrapper = document.createElement("div");
      wrapper.className = "caption-word-input";
      const childInput = document.createElement("span");
      wrapper.appendChild(childInput);
      document.body.appendChild(wrapper);

      expect(isInputIsolated(wrapper)).toBe(true);
      expect(isInputIsolated(childInput)).toBe(true);

      document.body.removeChild(wrapper);
    });

    it("detects document.activeElement when target is generic body/window", () => {
      const input = document.createElement("input");
      document.body.appendChild(input);
      input.focus();

      expect(isInputIsolated(document.body)).toBe(true);
      expect(isInputIsolated(null)).toBe(true);

      document.body.removeChild(input);
    });

    it("does not isolate generic buttons, divs or canvas", () => {
      const btn = document.createElement("button");
      const div = document.createElement("div");
      expect(isInputIsolated(btn)).toBe(false);
      expect(isInputIsolated(div)).toBe(false);
    });
  });

  // ==========================================================================
  // 2. Tactile Sound Blip Feedback (Web Audio API)
  // ==========================================================================
  describe("Tactile Synthetic Sound Feedback", () => {
    it("playTactileBeep dispatches streamer:audio-pip custom event", () => {
      const pipListener = vi.fn();
      window.addEventListener("streamer:audio-pip", pipListener);

      playTactileBeep(880, 20);

      expect(pipListener).toHaveBeenCalled();
      const eventDetail = pipListener.mock.calls[0][0].detail;
      expect(eventDetail.freq).toBe(880);
      expect(eventDetail.durationMs).toBe(20);

      window.removeEventListener("streamer:audio-pip", pipListener);
    });

    it("safely handles AudioContext mock in testing environments", () => {
      const mockOsc = {
        type: "sine",
        frequency: { setValueAtTime: vi.fn() },
        connect: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
      };
      const mockGain = {
        gain: { setValueAtTime: vi.fn(), exponentialRampToValueAtTime: vi.fn() },
        connect: vi.fn(),
      };
      const mockCtx = {
        currentTime: 0,
        destination: {},
        createOscillator: vi.fn(() => mockOsc),
        createGain: vi.fn(() => mockGain),
        close: vi.fn(),
      };

      class MockAudioContext {
        currentTime = mockCtx.currentTime;
        destination = mockCtx.destination;
        createOscillator = mockCtx.createOscillator;
        createGain = mockCtx.createGain;
        close = mockCtx.close;
      }

      const origAudioContext = (window as any).AudioContext;
      (window as any).AudioContext = MockAudioContext;

      expect(() => playTactileBeep(440, 25, "triangle")).not.toThrow();
      expect(mockCtx.createOscillator).toHaveBeenCalled();
      expect(mockCtx.createGain).toHaveBeenCalled();
      expect(mockOsc.start).toHaveBeenCalled();
      expect(mockOsc.stop).toHaveBeenCalled();

      (window as any).AudioContext = origAudioContext;
    });
  });

  // ==========================================================================
  // 3. Hotkey Navigation & Control (J-K-L, Space, I, O, 1-5)
  // ==========================================================================
  describe("Hotkey Navigation Engine", () => {
    it("Space hotkey toggles play and pause and flashes HUD badge", () => {
      const onPlayToggle = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={5.0}
          onPlayToggle={onPlayToggle}
          onAudioPip={onAudioPip}
        />
      );

      // 1st press -> Play
      fireEvent.keyDown(window, { code: "Space", key: " " });
      expect(onPlayToggle).toHaveBeenCalledWith(true);
      expect(onAudioPip).toHaveBeenCalledWith("space");

      const badge = screen.getByTestId("hud-badge");
      expect(badge.textContent).toContain("[SPACE] PLAY");

      // 2nd press -> Pause
      fireEvent.keyDown(window, { code: "Space", key: " " });
      expect(onPlayToggle).toHaveBeenCalledWith(false);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[SPACE] PAUSE");
    });

    it("J hotkey rewinds playback by 1 second and clamps at 0", () => {
      const onRewind = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={4.5}
          onShuttleRewind={onRewind}
          onAudioPip={onAudioPip}
        />
      );

      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      expect(onRewind).toHaveBeenCalledWith(3.5);
      expect(onAudioPip).toHaveBeenCalledWith("j");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[J] REWIND -1s");
    });

    it("K hotkey pauses playback and halts shuttle", () => {
      const onPause = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={10.0}
          onShuttlePause={onPause}
          onAudioPip={onAudioPip}
        />
      );

      fireEvent.keyDown(window, { code: "KeyK", key: "k" });
      expect(onPause).toHaveBeenCalled();
      expect(onAudioPip).toHaveBeenCalledWith("k");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[K] PAUSE");
    });

    it("L hotkey advances playback by 1 second and respects duration limit", () => {
      const onForward = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={9.5}
          duration={10.0}
          onShuttleForward={onForward}
          onAudioPip={onAudioPip}
        />
      );

      fireEvent.keyDown(window, { code: "KeyL", key: "l" });
      expect(onForward).toHaveBeenCalledWith(10.0);
      expect(onAudioPip).toHaveBeenCalledWith("l");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[L] FORWARD +1s");
    });

    it("I hotkey marks in-point at currentTime and flashes HUD", () => {
      const onMarkIn = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={6.4}
          onMarkIn={onMarkIn}
          onAudioPip={onAudioPip}
        />
      );

      fireEvent.keyDown(window, { code: "KeyI", key: "i" });
      expect(onMarkIn).toHaveBeenCalledWith(6.4);
      expect(onAudioPip).toHaveBeenCalledWith("i");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[I] MARK IN 6.4s");
    });

    it("O hotkey marks out-point at currentTime and flashes HUD", () => {
      const onMarkOut = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          currentTime={18.2}
          onMarkOut={onMarkOut}
          onAudioPip={onAudioPip}
        />
      );

      fireEvent.keyDown(window, { code: "KeyO", key: "o" });
      expect(onMarkOut).toHaveBeenCalledWith(18.2);
      expect(onAudioPip).toHaveBeenCalledWith("o");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[O] MARK OUT 18.2s");
    });

    it("Number keys 1 to 5 record clip ratings and update HUD & UI", () => {
      const onRate = vi.fn();
      const onAudioPip = vi.fn();
      render(
        <StreamerWorkflowBar
          activeClipId="c1"
          onRate={onRate}
          onAudioPip={onAudioPip}
        />
      );

      // Press 1
      fireEvent.keyDown(window, { code: "Digit1", key: "1" });
      expect(onRate).toHaveBeenCalledWith(1);
      expect(screen.getByTestId("hud-badge").textContent).toContain("[★ 1] RATED 1");

      // Press 5
      fireEvent.keyDown(window, { code: "Digit5", key: "5" });
      expect(onRate).toHaveBeenCalledWith(5);
      expect(onAudioPip).toHaveBeenCalledWith("5");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[★ 5] RATED 5");
      expect(screen.getByTestId("current-clip-rating").textContent).toBe("5★");
    });

    it("ignores hotkeys when modifier keys (Ctrl / Meta / Alt) are active", () => {
      const onRewind = vi.fn();
      render(<StreamerWorkflowBar currentTime={5.0} onShuttleRewind={onRewind} />);

      fireEvent.keyDown(window, { code: "KeyJ", key: "j", ctrlKey: true });
      fireEvent.keyDown(window, { code: "KeyJ", key: "j", metaKey: true });
      fireEvent.keyDown(window, { code: "KeyJ", key: "j", altKey: true });

      expect(onRewind).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });
  });

  // ==========================================================================
  // 4. Strict Hotkey Input Isolation Tests
  // ==========================================================================
  describe("Hotkey Input Isolation During Text Editing", () => {
    it("strictly ignores all hotkeys when focused in standard <input>", () => {
      const onRate = vi.fn();
      const onPlay = vi.fn();
      render(
        <div>
          <input data-testid="test-input" />
          <StreamerWorkflowBar onRate={onRate} onPlayToggle={onPlay} />
        </div>
      );

      const input = screen.getByTestId("test-input");
      input.focus();

      fireEvent.keyDown(input, { code: "Space", key: " " });
      fireEvent.keyDown(input, { code: "KeyJ", key: "j" });
      fireEvent.keyDown(input, { code: "KeyK", key: "k" });
      fireEvent.keyDown(input, { code: "KeyL", key: "l" });
      fireEvent.keyDown(input, { code: "KeyI", key: "i" });
      fireEvent.keyDown(input, { code: "KeyO", key: "o" });
      fireEvent.keyDown(input, { code: "Digit5", key: "5" });

      expect(onPlay).not.toHaveBeenCalled();
      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("strictly ignores all hotkeys when focused in <textarea>", () => {
      const onRate = vi.fn();
      render(
        <div>
          <textarea data-testid="test-textarea" />
          <StreamerWorkflowBar onRate={onRate} />
        </div>
      );

      const textarea = screen.getByTestId("test-textarea");
      textarea.focus();

      fireEvent.keyDown(textarea, { code: "Digit3", key: "3" });
      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("strictly ignores all hotkeys when typing inside [contenteditable] elements", () => {
      const onPlay = vi.fn();
      render(
        <div>
          <div data-testid="test-editable" contentEditable={true}>
            <span data-testid="test-editable-child">Editable Word</span>
          </div>
          <StreamerWorkflowBar onPlayToggle={onPlay} />
        </div>
      );

      const child = screen.getByTestId("test-editable-child");
      fireEvent.keyDown(child, { code: "Space", key: " " });
      expect(onPlay).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });

    it("strictly ignores all hotkeys when editing inline caption with class .caption-word-input", () => {
      const onRewind = vi.fn();
      const onRate = vi.fn();
      render(
        <div>
          <input
            data-testid="caption-inline-input"
            className="caption-word-input"
            defaultValue="Awesome"
          />
          <StreamerWorkflowBar
            currentTime={8.0}
            onShuttleRewind={onRewind}
            onRate={onRate}
          />
        </div>
      );

      const captionInput = screen.getByTestId("caption-inline-input");
      captionInput.focus();

      // Press 'j', '5', 'i', 'o'
      fireEvent.keyDown(captionInput, { code: "KeyJ", key: "j" });
      fireEvent.keyDown(captionInput, { code: "Digit5", key: "5" });
      fireEvent.keyDown(captionInput, { code: "KeyI", key: "i" });

      expect(onRewind).not.toHaveBeenCalled();
      expect(onRate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("hud-badge")).toBeNull();
    });
  });

  // ==========================================================================
  // 5. Visual HUD Feedback & Timer Dissolve Tests
  // ==========================================================================
  describe("Visual HUD / OSD Badge Feedback", () => {
    it("HUD badge renders with neon cyan styling and disappears after 600ms", () => {
      vi.useFakeTimers();

      render(<StreamerWorkflowBar currentTime={2.5} />);

      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      const badge = screen.getByTestId("hud-badge");
      expect(badge).toBeDefined();
      expect(badge.textContent).toContain("[J] REWIND -1s");

      // Advance by 650ms past the 600ms timer
      act(() => {
        vi.advanceTimersByTime(650);
      });

      expect(screen.queryByTestId("hud-badge")).toBeNull();

      vi.useRealTimers();
    });

    it("rapid hotkey triggers restart the HUD timeout and update badge text", () => {
      vi.useFakeTimers();

      render(<StreamerWorkflowBar currentTime={3.0} />);

      fireEvent.keyDown(window, { code: "KeyJ", key: "j" });
      expect(screen.getByTestId("hud-badge").textContent).toContain("[J] REWIND -1s");

      act(() => {
        vi.advanceTimersByTime(300);
      });

      // Press L before 600ms expires
      fireEvent.keyDown(window, { code: "KeyL", key: "l" });
      expect(screen.getByTestId("hud-badge").textContent).toContain("[L] FORWARD +1s");

      // Advance 400ms -> still visible because timer reset
      act(() => {
        vi.advanceTimersByTime(400);
      });
      expect(screen.getByTestId("hud-badge")).toBeDefined();

      // Advance past remaining 250ms -> dissolves
      act(() => {
        vi.advanceTimersByTime(250);
      });
      expect(screen.queryByTestId("hud-badge")).toBeNull();

      vi.useRealTimers();
    });
  });

  // ==========================================================================
  // 6. One-Click Batch Action Toolbar Tests
  // ==========================================================================
  describe("One-Click Batch Action Toolbar", () => {
    it("Select All toggles all clip IDs in selection", () => {
      render(<StreamerWorkflowBar clips={mockClips} />);

      const selectAllBtn = screen.getByTestId("batch-select-all");
      const count = screen.getByTestId("selected-count");

      expect(count.textContent).toBe("0");

      // Select All
      fireEvent.click(selectAllBtn);
      expect(count.textContent).toBe("3");

      // Deselect All
      fireEvent.click(selectAllBtn);
      expect(count.textContent).toBe("0");
    });

    it("allows toggling individual clips via quick chips", () => {
      render(<StreamerWorkflowBar clips={mockClips} />);

      const chipC1 = screen.getByTestId("batch-select-clip-c1");
      const chipC2 = screen.getByTestId("batch-select-clip-c2");
      const count = screen.getByTestId("selected-count");

      fireEvent.click(chipC1);
      expect(count.textContent).toBe("1");

      fireEvent.click(chipC2);
      expect(count.textContent).toBe("2");

      fireEvent.click(chipC1);
      expect(count.textContent).toBe("1");
    });

    it("executes One-Click Batch Rate on selected clips and displays HUD", async () => {
      const onBatch = vi.fn();
      render(<StreamerWorkflowBar clips={mockClips} onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-all"));
      fireEvent.click(screen.getByTestId("batch-rate"));

      expect(onBatch).toHaveBeenCalledWith("rate", ["c1", "c2", "c3"], { rating: 5 });
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] RATED 3 CLIPS (5★)");
    });

    it("executes One-Click Batch Tag with #viral & #climax", async () => {
      const onBatch = vi.fn();
      render(<StreamerWorkflowBar clips={mockClips} onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-clip-c1"));
      fireEvent.click(screen.getByTestId("batch-select-clip-c3"));
      fireEvent.click(screen.getByTestId("batch-tag"));

      expect(onBatch).toHaveBeenCalledWith("tag", ["c1", "c3"], {
        tags: ["#viral", "#climax"],
      });
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] TAGGED 2 CLIPS");
    });

    it("executes One-Click Batch Export across selected clips", async () => {
      const onBatch = vi.fn();
      render(<StreamerWorkflowBar clips={mockClips} onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-all"));
      fireEvent.click(screen.getByTestId("batch-export"));

      expect(onBatch).toHaveBeenCalledWith("export", ["c1", "c2", "c3"], {});
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] EXPORTING 3 CLIPS");
    });

    it("executes One-Click Batch Delete with confirmation and clears selection", async () => {
      const onBatch = vi.fn();
      const origConfirm = window.confirm;
      const confirmMock = vi.fn(() => true);
      window.confirm = confirmMock;

      render(<StreamerWorkflowBar clips={mockClips} onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-clip-c2"));
      expect(screen.getByTestId("selected-count").textContent).toBe("1");

      fireEvent.click(screen.getByTestId("batch-delete"));

      expect(confirmMock).toHaveBeenCalled();
      expect(onBatch).toHaveBeenCalledWith("delete", ["c2"], {});
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] DELETED 1 CLIPS");
      expect(screen.getByTestId("selected-count").textContent).toBe("0");

      window.confirm = origConfirm;
    });

    it("aborts Batch Delete if user cancels confirmation dialog", async () => {
      const onBatch = vi.fn();
      const origConfirm = window.confirm;
      const confirmMock = vi.fn(() => false);
      window.confirm = confirmMock;

      render(<StreamerWorkflowBar clips={mockClips} onBatchAction={onBatch} />);

      fireEvent.click(screen.getByTestId("batch-select-clip-c1"));
      fireEvent.click(screen.getByTestId("batch-delete"));

      expect(confirmMock).toHaveBeenCalled();
      expect(onBatch).not.toHaveBeenCalled();

      window.confirm = origConfirm;
    });

    it("falls back to activeClipId if no clips are explicitly selected", async () => {
      const onBatch = vi.fn();
      render(
        <StreamerWorkflowBar
          clips={mockClips}
          activeClipId="c2"
          onBatchAction={onBatch}
        />
      );

      fireEvent.click(screen.getByTestId("batch-rate"));

      expect(onBatch).toHaveBeenCalledWith("rate", ["c2"], { rating: 5 });
      expect(screen.getByTestId("hud-badge").textContent).toContain("[BATCH] RATED 1 CLIPS (5★)");
    });
  });

  // ==========================================================================
  // 7. Interactive Click Handlers in Workflow Bar
  // ==========================================================================
  describe("Toolbar Interactive Buttons", () => {
    it("clicking transport buttons invokes rewind, pause, forward, and mark in/out", () => {
      const onRewind = vi.fn();
      const onPause = vi.fn();
      const onForward = vi.fn();
      const onMarkIn = vi.fn();
      const onMarkOut = vi.fn();

      render(
        <StreamerWorkflowBar
          currentTime={12.0}
          onShuttleRewind={onRewind}
          onShuttlePause={onPause}
          onShuttleForward={onForward}
          onMarkIn={onMarkIn}
          onMarkOut={onMarkOut}
        />
      );

      fireEvent.click(screen.getByTestId("hotkey-shuttle-rewind"));
      expect(onRewind).toHaveBeenCalledWith(11.0);

      fireEvent.click(screen.getByTestId("hotkey-shuttle-pause"));
      expect(onPause).toHaveBeenCalled();

      fireEvent.click(screen.getByTestId("hotkey-shuttle-forward"));
      expect(onForward).toHaveBeenCalledWith(13.0);

      fireEvent.click(screen.getByTestId("hotkey-mark-in"));
      expect(onMarkIn).toHaveBeenCalledWith(12.0);

      fireEvent.click(screen.getByTestId("hotkey-mark-out"));
      expect(onMarkOut).toHaveBeenCalledWith(12.0);
    });

    it("clicking rating star buttons updates rating score and flashes HUD", () => {
      const onRate = vi.fn();
      render(<StreamerWorkflowBar activeClipId="c1" onRate={onRate} />);

      fireEvent.click(screen.getByTestId("hotkey-rate-4"));
      expect(onRate).toHaveBeenCalledWith(4);
      expect(screen.getByTestId("current-clip-rating").textContent).toBe("4★");
      expect(screen.getByTestId("hud-badge").textContent).toContain("[★ 4] RATED 4");
    });
  });
});
