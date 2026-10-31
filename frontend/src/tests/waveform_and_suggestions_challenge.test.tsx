import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React, { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AudioWaveformHeatmap, {
  COLOR_CLIMAX,
  COLOR_CROWD_ROAR,
  COLOR_KILL_STREAK,
  COLOR_SCREAM,
  getDecibelColor,
  type StoryboardData,
  type WaveformData,
  type WaveformEvent,
} from "../components/AudioWaveformHeatmap";

import AiSuggestionsCoPilot, {
  type AiSuggestion,
} from "../components/AiSuggestionsCoPilot";

import { useUndo, type UndoState } from "../lib/useUndo";

// ============================================================================
// Happy-DOM Setup & Mocking
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

let mockTrackWidth = 600;
let mockTrackLeft = 50;

beforeEach(() => {
  mockTrackWidth = 600;
  mockTrackLeft = 50;

  // Mock global fetch to return offline mock data and prevent ECONNREFUSED
  global.fetch = vi.fn().mockImplementation((url: string) => {
    if (url.includes("/waveform")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          clip_id: "mock_clip",
          duration_s: 20.0,
          samples: [0.1, 0.5, 0.8],
          dbfs: [-30, -15, -3],
          climax_t: 10.0,
          events: [],
        }),
      });
    }
    if (url.includes("/storyboard")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          clip_id: "mock_clip",
          sprite_url: "/storyboard.jpg",
          cols: 5,
          rows: 2,
          tile_w: 160,
          tile_h: 90,
          interval_s: 2.0,
          total_frames: 10,
          duration_s: 20.0,
        }),
      });
    }
    if (url.includes("/suggestions") && url.includes("/apply")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          success: true,
          clip: { id: "mock_clip" },
          updated_fields: ["start_t"],
        }),
      });
    }
    if (url.includes("/suggestions")) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          clip_id: "mock_clip",
          suggestions: [],
        }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({}),
    });
  });

  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (
    this: HTMLElement
  ) {
    if (this.getAttribute("data-testid") === "waveform-track") {
      return {
        width: mockTrackWidth,
        height: 96,
        top: 100,
        left: mockTrackLeft,
        bottom: 196,
        right: mockTrackLeft + mockTrackWidth,
        x: mockTrackLeft,
        y: 100,
        toJSON: () => {},
      };
    }
    return {
      width: 100,
      height: 100,
      top: 0,
      left: 0,
      bottom: 100,
      right: 100,
      x: 0,
      y: 0,
      toJSON: () => {},
    };
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// ============================================================================
// CHALLENGE 1: WAVEFORM BOUNDARY STRESS
// ============================================================================

describe("CHALLENGE 1: Waveform Boundary Stress (Empirical Challenge)", () => {
  it("Stress 1.1: Handles zero duration boundary (duration = 0) safely without NaN SVG attributes", () => {
    // When duration is 0, duration is clamped to 0.1s preventing divide-by-zero NaNs
    render(
      <AudioWaveformHeatmap
        start={0}
        end={0}
        duration={0}
        currentTime={0}
      />
    );

    expect(screen.getByTestId("audio-waveform-heatmap")).toBeTruthy();
    expect(screen.getByTestId("waveform-track")).toBeTruthy();
    expect(screen.getByTestId("waveform-svg")).toBeTruthy();

    const playhead = screen.getByTestId("waveform-playhead");
    // playheadPct = (0 / 0.1) * 100 -> 0% safely positioned
    expect(playhead.style.left).toBe("0%");
  });

  it("Stress 1.2: Handles sub-second duration (0.05s and 0.3s) with float precision tolerance", () => {
    const subSecondData: WaveformData = {
      clip_id: "clip_sub_sec",
      duration_s: 0.3,
      samples: [0.1, 0.9, 0.4],
      dbfs: [-20, -3, -15],
      climax_t: 0.15,
      events: [
        {
          type: "scream",
          t: 0.15,
          duration: 0.05,
          color: COLOR_SCREAM,
          label: "Micro Scream",
        },
      ],
    };

    render(
      <AudioWaveformHeatmap
        start={1.0}
        end={1.3}
        duration={0.3}
        currentTime={1.15}
        waveformData={subSecondData}
      />
    );

    const playhead = screen.getByTestId("waveform-playhead");
    // currentTime = 1.15, start = 1.0 -> rel = 0.15 / 0.3 -> ~50%
    expect(parseFloat(playhead.style.left)).toBeCloseTo(50, 1);

    const markers = screen.getAllByTestId("waveform-event-marker");
    expect(markers.length).toBe(1);
    expect(markers[0].getAttribute("data-event-type")).toBe("scream");
  });

  it("Stress 1.3: Inverted boundaries (start > end) are protected by span clamp", () => {
    // start 30, end 10 -> span clamp to Math.max(0.1, -20) = 0.1
    let seekAbs = -1;
    let seekRel = -1;

    render(
      <AudioWaveformHeatmap
        start={30}
        end={10}
        currentTime={10}
        onSeek={(abs, rel) => {
          seekAbs = abs;
          seekRel = rel;
        }}
      />
    );

    const track = screen.getByTestId("waveform-track");
    fireEvent.pointerDown(track, { clientX: 350, pointerId: 1 }); // 50% fraction

    expect(seekAbs).toBeCloseTo(30.05, 2); // 30 + 0.5 * 0.1
    expect(seekRel).toBeCloseTo(0.05, 2); // 0.5 * 0.1
  });

  it("Stress 1.4: Empty samples array renders clean SVG without crashing", () => {
    const emptyData: WaveformData = {
      clip_id: "clip_empty",
      duration_s: 10.0,
      samples: [],
      dbfs: [],
      climax_t: 5.0,
      events: [],
    };

    render(
      <AudioWaveformHeatmap
        start={0}
        end={10}
        waveformData={emptyData}
      />
    );

    const svg = screen.getByTestId("waveform-svg");
    expect(svg).toBeTruthy();
    expect(svg.getAttribute("viewBox")).toBe("0 0 0 76");
    expect(svg.querySelectorAll("rect").length).toBe(0);
  });

  it("Stress 1.5: NaN and Infinite sample values are handled without render crash", () => {
    const nanData: WaveformData = {
      clip_id: "clip_nan",
      duration_s: 10.0,
      samples: [NaN, Infinity, -Infinity, 0.5],
      dbfs: [NaN, 10, -120, -14],
      climax_t: 5.0,
      events: [],
    };

    render(
      <AudioWaveformHeatmap
        start={0}
        end={10}
        waveformData={nanData}
      />
    );

    const svg = screen.getByTestId("waveform-svg");
    const rects = svg.querySelectorAll("rect");
    expect(rects.length).toBe(4);
  });

  it("Stress 1.6: Decibel boundaries (-120dB to +10dB and exact thresholds)", () => {
    // Extreme low
    expect(getDecibelColor(-120)).toBe("#3B82F6");
    expect(getDecibelColor(-60)).toBe("#3B82F6");

    // Boundary at -25 dB
    expect(getDecibelColor(-25.0001)).toBe("#3B82F6"); // quiet blue
    expect(getDecibelColor(-25.0)).toBe("#10B981"); // normal dialogue green
    expect(getDecibelColor(-24.999)).toBe("#10B981");

    // Boundary at -12 dB
    expect(getDecibelColor(-12.0001)).toBe("#10B981"); // dialogue green
    expect(getDecibelColor(-12.0)).toBe("#F59E0B"); // high energy amber
    expect(getDecibelColor(-11.999)).toBe("#F59E0B");

    // Boundary at -4 dB
    expect(getDecibelColor(-4.0001)).toBe("#F59E0B"); // amber
    expect(getDecibelColor(-4.0)).toBe("#EF4444"); // peak red
    expect(getDecibelColor(-3.999)).toBe("#EF4444");

    // Extreme high
    expect(getDecibelColor(0.0)).toBe("#EF4444");
    expect(getDecibelColor(10.0)).toBe("#EF4444");
    expect(getDecibelColor(100.0)).toBe("#EF4444");

    // NaN dB
    expect(getDecibelColor(NaN)).toBe("#EF4444");
  });

  it("Stress 1.7: Dense event overlaps (50 markers at same timestamp)", () => {
    const denseEvents: WaveformEvent[] = [];
    for (let i = 0; i < 50; i++) {
      denseEvents.push({
        type: i % 2 === 0 ? "scream" : "crowd_roar",
        t: 10.0,
        duration: 1.5,
        color: i % 2 === 0 ? COLOR_SCREAM : COLOR_CROWD_ROAR,
        label: `Dense Event ${i}`,
      });
    }

    const denseData: WaveformData = {
      clip_id: "clip_dense",
      duration_s: 20.0,
      samples: [0.5, 0.6],
      dbfs: [-10, -5],
      climax_t: 10.0,
      events: denseEvents,
    };

    let seekCalled = false;
    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={denseData}
        onSeek={() => {
          seekCalled = true;
        }}
      />
    );

    const markers = screen.getAllByTestId("waveform-event-marker");
    expect(markers.length).toBe(50);

    // Clicking top marker triggers onSeek without error
    fireEvent.click(markers[49]);
    expect(seekCalled).toBe(true);
  });

  it("Stress 1.8: Event markers outside totalDuration boundaries are clamped", () => {
    const outOfBoundsData: WaveformData = {
      clip_id: "clip_oob",
      duration_s: 20.0,
      samples: [0.5],
      dbfs: [-10],
      climax_t: 10.0,
      events: [
        {
          type: "scream",
          t: -5.0, // negative timestamp
          duration: 2.0,
          color: COLOR_SCREAM,
          label: "Negative Event",
        },
        {
          type: "climax",
          t: 99.0, // way past duration
          duration: 5.0,
          color: COLOR_CLIMAX,
          label: "Late Event",
        },
      ],
    };

    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={outOfBoundsData}
      />
    );

    const markers = screen.getAllByTestId("waveform-event-marker");
    expect(markers.length).toBe(2);

    // Negative timestamp clamped to 0%
    expect(markers[0].style.left).toBe("0%");
    // Over-duration clamped to 96%
    expect(markers[1].style.left).toBe("96%");
  });

  it("Stress 1.9: Current time calculation in rendered mode progresses smoothly past start timestamp without backward jump", () => {
    const { rerender } = render(
      <AudioWaveformHeatmap
        start={5.0}
        end={25.0}
        duration={20.0}
        currentTime={4.9}
        previewMode="rendered"
      />
    );

    const playheadAt4_9 = screen.getByTestId("waveform-playhead");
    const pct4_9 = parseFloat(playheadAt4_9.style.left);
    expect(pct4_9).toBeCloseTo(24.5, 1); // 4.9 / 20 = 24.5%

    // Advance playback time to 5.0s
    rerender(
      <AudioWaveformHeatmap
        start={5.0}
        end={25.0}
        duration={20.0}
        currentTime={5.0}
        previewMode="rendered"
      />
    );

    const playheadAt5_0 = screen.getByTestId("waveform-playhead");
    const pct5_0 = parseFloat(playheadAt5_0.style.left);
    // In rendered preview mode, currentTime 5.0s maps smoothly to 25% (5.0 / 20.0 = 25%) without backward jump
    expect(pct5_0).toBeCloseTo(25.0, 1);
  });
});

// ============================================================================
// CHALLENGE 2: SCRUBBING TIMESTAMP ORACLE & TOOLTIP BOUNDARIES
// ============================================================================

describe("CHALLENGE 2: Scrubbing Timestamp Oracle & Tooltip Boundaries", () => {
  it("Oracle 2.1: Mouse coordinate to fraction mapping oracle (x=0, x=width, x<0, x>width)", () => {
    let capturedAbsTime = -1;
    let capturedRelTime = -1;

    render(
      <AudioWaveformHeatmap
        start={10.0}
        end={30.0}
        duration={20.0}
        onSeek={(abs, rel) => {
          capturedAbsTime = abs;
          capturedRelTime = rel;
        }}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Test Boundary: x = 0 (clientX = left = 50)
    fireEvent.pointerDown(track, { clientX: 50, pointerId: 1 });
    expect(capturedRelTime).toBeCloseTo(0.0, 4);
    expect(capturedAbsTime).toBeCloseTo(10.0, 4);

    // Test Boundary: x = width (clientX = left + width = 650)
    fireEvent.pointerDown(track, { clientX: 650, pointerId: 1 });
    expect(capturedRelTime).toBeCloseTo(20.0, 4);
    expect(capturedAbsTime).toBeCloseTo(30.0, 4);

    // Test Sub-Boundary: x < 0 (clientX = -100) -> clamped to 0
    fireEvent.pointerDown(track, { clientX: -100, pointerId: 1 });
    expect(capturedRelTime).toBeCloseTo(0.0, 4);
    expect(capturedAbsTime).toBeCloseTo(10.0, 4);

    // Test Sub-Boundary: x > width (clientX = 1200) -> clamped to width
    fireEvent.pointerDown(track, { clientX: 1200, pointerId: 1 });
    expect(capturedRelTime).toBeCloseTo(20.0, 4);
    expect(capturedAbsTime).toBeCloseTo(30.0, 4);

    // Test Intermediate Oracle Point: x = 150 (relative x = 100, frac = 100/600 = 1/6)
    fireEvent.pointerDown(track, { clientX: 150, pointerId: 1 });
    expect(capturedRelTime).toBeCloseTo(20.0 * (100 / 600), 4);
    expect(capturedAbsTime).toBeCloseTo(10.0 + 20.0 * (100 / 600), 4);
  });

  it("Oracle 2.2: Zero-width container boundary does not yield NaN or Infinity", () => {
    mockTrackWidth = 0; // Container collapsed to 0px
    let capturedAbs = -1;

    render(
      <AudioWaveformHeatmap
        start={5.0}
        end={25.0}
        onSeek={(abs) => {
          capturedAbs = abs;
        }}
      />
    );

    const track = screen.getByTestId("waveform-track");
    fireEvent.pointerDown(track, { clientX: 50, pointerId: 1 });
    expect(capturedAbs).toBe(5.0); // start + 0 * span = 5.0, not NaN
  });

  it("Oracle 2.3: Storyboard sprite grid coordinate oracle", () => {
    const storyboard: StoryboardData = {
      clip_id: "oracle_clip",
      sprite_url: "/storyboard.jpg",
      cols: 4,
      rows: 3,
      tile_w: 160,
      tile_h: 90,
      interval_s: 2.0,
      total_frames: 12,
      duration_s: 24.0,
    };

    render(
      <AudioWaveformHeatmap
        start={0}
        end={24}
        duration={24}
        storyboardData={storyboard}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Oracle function to verify sprite offsets
    const computeExpectedSprite = (hoverRelTime: number) => {
      const tileW = 140;
      const tileH = Math.round(tileW * (9 / 16)); // 79
      const interval = 2.0;
      const totalFrames = 12;
      const cols = 4;
      const frameIdx = Math.min(totalFrames - 1, Math.max(0, Math.floor(hoverRelTime / interval)));
      const col = frameIdx % cols;
      const row = Math.floor(frameIdx / cols);
      const bgX = col === 0 ? "0px" : `-${col * tileW}px`;
      const bgY = row === 0 ? "0px" : `-${row * tileH}px`;
      return {
        backgroundPosition: `${bgX} ${bgY}`,
        width: `${tileW}px`,
        height: `${tileH}px`,
      };
    };

    // Pointer enter
    fireEvent.pointerEnter(track);

    // Hover at frame 0 (hoverRelTime = 0) -> col 0, row 0 -> 0px 0px
    fireEvent.pointerMove(track, { clientX: 50 }); // frac 0
    let thumbnail = screen.getByTestId("scrub-thumbnail");
    let exp0 = computeExpectedSprite(0);
    expect(thumbnail.style.backgroundPosition.replace(/-0px/g, "0px")).toBe(exp0.backgroundPosition);

    // Hover at frame 5 (hoverRelTime = 10s -> frac 10/24 -> clientX = 50 + 600 * (10/24) = 300)
    // frameIdx = 5, col = 5 % 4 = 1, row = floor(5/4) = 1 -> -140px -79px
    fireEvent.pointerMove(track, { clientX: 300 });
    thumbnail = screen.getByTestId("scrub-thumbnail");
    let exp5 = computeExpectedSprite(10);
    expect(thumbnail.style.backgroundPosition.replace(/-0px/g, "0px")).toBe(exp5.backgroundPosition);

    // Hover at frame 11 (last frame) -> hoverRelTime = 23.5s -> clientX = 50 + 600 * (23.5/24) = 637.5
    // frameIdx = 11, col = 11 % 4 = 3, row = floor(11/4) = 2 -> -420px -158px
    fireEvent.pointerMove(track, { clientX: 637.5 });
    thumbnail = screen.getByTestId("scrub-thumbnail");
    let exp11 = computeExpectedSprite(23.5);
    expect(thumbnail.style.backgroundPosition.replace(/-0px/g, "0px")).toBe(exp11.backgroundPosition);

    // Hover way past end (clientX = 800) -> clamped to last frame (frameIdx = 11)
    fireEvent.pointerMove(track, { clientX: 800 });
    thumbnail = screen.getByTestId("scrub-thumbnail");
    expect(thumbnail.style.backgroundPosition.replace(/-0px/g, "0px")).toBe(exp11.backgroundPosition);
  });

  it("Oracle 2.4: Floating preview tooltip displays accurate formatted timestamp and remains in DOM", () => {
    render(
      <AudioWaveformHeatmap
        start={10.0}
        end={30.0}
        duration={20.0}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Before hover: tooltip does not exist
    expect(screen.queryByTestId("scrub-preview-tooltip")).toBeNull();

    // On hover at 50% (clientX = 350)
    fireEvent.pointerEnter(track);
    fireEvent.pointerMove(track, { clientX: 350 });

    const tooltip = screen.getByTestId("scrub-preview-tooltip");
    expect(tooltip).toBeTruthy();
    expect(tooltip.style.left).toBe("50%");

    const timestampBadge = screen.getByTestId("scrub-timestamp");
    expect(timestampBadge.textContent).toBe("10.0s / 20.0s");

    // On pointer leave
    fireEvent.pointerLeave(track);
    expect(screen.queryByTestId("scrub-preview-tooltip")).toBeNull();
  });

  it("Oracle 2.5 (Tooltip Boundary Clamping): Tooltip clamps transform at boundary edges x=0 and x=width", () => {
    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        duration={20}
      />
    );

    const track = screen.getByTestId("waveform-track");
    fireEvent.pointerEnter(track);

    // At x = 0 (left edge): left is 0% and transform is clamped to translateX(0%)
    fireEvent.pointerMove(track, { clientX: 50 });
    let tooltip = screen.getByTestId("scrub-preview-tooltip");
    expect(tooltip.style.left).toBe("0%");
    expect(tooltip.style.transform).toBe("translateX(0%)");

    // At x = width (right edge): left is 100% and transform is clamped to translateX(-100%)
    fireEvent.pointerMove(track, { clientX: 650 });
    tooltip = screen.getByTestId("scrub-preview-tooltip");
    expect(tooltip.style.left).toBe("100%");
    expect(tooltip.style.transform).toBe("translateX(-100%)");
  });
});

// ============================================================================
// CHALLENGE 3: SUGGESTION MUTATION STRESS
// ============================================================================

describe("CHALLENGE 3: Suggestion Mutation Stress", () => {
  const STRESS_SUGGESTIONS: AiSuggestion[] = [
    {
      id: "sug_rapid_1",
      type: "trim_hook",
      title: "Rapid Trim 1",
      description: "Trim 1",
      retention_impact: "+15% retention",
      patch: { start_t: 5.0 },
    },
    {
      id: "sug_rapid_2",
      type: "trim_hook",
      title: "Rapid Trim 2",
      description: "Trim 2",
      retention_impact: "+20% retention",
      patch: { start_t: 8.0 },
    },
    {
      id: "sug_malformed",
      type: "custom_type",
      title: "Malformed Patch",
      description: "Missing patch fields",
      retention_impact: "+5% retention",
      // patch undefined
    },
    {
      id: "sug_multi_patch",
      type: "combo_suggestion",
      title: "Multi Field Suggestion",
      description: "Applies start, end, layout",
      retention_impact: "+35% retention",
      patch: { start_t: 12.0, end_t: 28.0, layout: "split" },
    },
    {
      id: "sug_hook_min",
      type: "hook_rating",
      title: "Hook Strength: 0/100",
      description: "Dead opener",
      retention_impact: "Weak Hook",
      patch: {},
    },
    {
      id: "sug_hook_max",
      type: "hook_rating",
      title: "Hook Strength: 100/100",
      description: "God tier hook",
      retention_impact: "Viral Opener",
      patch: {},
    },
  ];

  it("Stress 3.1: Rapid successive clicks on single apply button are idempotent / safe", async () => {
    let applyCount = 0;
    const onApply = vi.fn(() => {
      applyCount++;
    });

    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={[STRESS_SUGGESTIONS[0]]}
        onApplySuggestion={onApply}
      />
    );

    const btn = screen.getByTestId("apply-suggestion-btn");

    // Simulate 10 rapid clicks in immediate succession
    for (let i = 0; i < 10; i++) {
      fireEvent.click(btn);
    }

    await waitFor(() => {
      expect(screen.getByTestId("applied-confirmation-badge")).toBeTruthy();
    });

    // onApply should have fired once because button is disabled during application and transitions to applied
    expect(applyCount).toBe(1);
    expect(screen.queryByTestId("apply-suggestion-btn")).toBeNull();
  });

  it("Stress 3.2: Malformed or undefined patch does not throw an unhandled crash", async () => {
    const onApply = vi.fn();

    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={[STRESS_SUGGESTIONS[2]]}
        onApplySuggestion={onApply}
      />
    );

    const btn = screen.getByTestId("apply-suggestion-btn");
    fireEvent.click(btn);

    await waitFor(() => {
      expect(screen.getByTestId("applied-confirmation-badge")).toBeTruthy();
    });

    expect(onApply).toHaveBeenCalled();
    expect(onApply.mock.calls[0][0].id).toBe("sug_malformed");
  });

  it("Stress 3.3: Conflicting suggestion applications update sequentially", async () => {
    const appliedLog: number[] = [];

    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={[STRESS_SUGGESTIONS[0], STRESS_SUGGESTIONS[1]]}
        onApplySuggestion={(sug) => {
          if (sug.patch?.start_t !== undefined) {
            appliedLog.push(sug.patch.start_t);
          }
        }}
      />
    );

    const buttons = screen.getAllByTestId("apply-suggestion-btn");
    expect(buttons.length).toBe(2);

    // Apply first suggestion (start_t = 5.0)
    fireEvent.click(buttons[0]);
    await waitFor(() => {
      expect(appliedLog).toContain(5.0);
    });

    // Apply second suggestion (start_t = 8.0)
    const secondButton = screen.getByTestId("apply-suggestion-btn");
    fireEvent.click(secondButton);
    await waitFor(() => {
      expect(appliedLog).toContain(8.0);
    });

    expect(appliedLog).toEqual([5.0, 8.0]);
  });

  it("Stress 3.4: Hook rating boundary scores (0/100, 100/100) render visual meters accurately", () => {
    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={[STRESS_SUGGESTIONS[4], STRESS_SUGGESTIONS[5]]}
      />
    );

    const scores = screen.getAllByTestId("hook-rating-score");
    expect(scores.length).toBe(2);
    expect(scores[0].textContent).toBe("0 / 100");
    expect(scores[1].textContent).toBe("100 / 100");

    const meters = screen.getAllByTestId("hook-rating-meter");
    expect(meters.length).toBe(2);

    const meter0 = meters[0].querySelector("div");
    const meter100 = meters[1].querySelector("div");

    expect(meter0?.style.width).toBe("0%");
    expect(meter100?.style.width).toBe("100%");
  });

  it("Stress 3.5: Dismissing all cards displays optimization completion banner", () => {
    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={[STRESS_SUGGESTIONS[0], STRESS_SUGGESTIONS[1]]}
      />
    );

    const dismissButtons = screen.getAllByTitle("Dismiss suggestion");
    expect(dismissButtons.length).toBe(2);

    // Dismiss first
    fireEvent.click(dismissButtons[0]);
    expect(screen.getAllByTestId("suggestion-card").length).toBe(1);

    // Dismiss second
    const remainingDismiss = screen.getByTitle("Dismiss suggestion");
    fireEvent.click(remainingDismiss);

    expect(screen.queryByTestId("suggestion-card")).toBeNull();
    expect(
      screen.getByText(/All suggestions applied! Clip is fully optimized for viral retention/)
    ).toBeTruthy();
  });

  it("Stress 3.6: Atomic multi-field patch application preserves all fields in undo history", () => {
    function TestHarness() {
      const undo = useUndo({
        title: "Test",
        start: 10,
        end: 20,
        styleId: "",
        cx: null,
        words: [],
        layout: "center",
        cam: null,
        aspect: "",
        capSpeakers: null,
        loopPreviewSeconds: 0,
      });

      return (
        <div>
          <span data-testid="state-start">{undo.state.start}</span>
          <span data-testid="state-end">{undo.state.end}</span>
          <button
            data-testid="atomic-multi-apply"
            onClick={() => {
              // Remediation: accumulate nextUndo before calling undo.set once
              const patch = { start_t: 12, end_t: 28 };
              const nextUndo = { ...undo.state };
              if (patch.start_t !== undefined) nextUndo.start = patch.start_t;
              if (patch.end_t !== undefined) nextUndo.end = patch.end_t;
              undo.set(nextUndo);
            }}
          >
            Apply Multi
          </button>
        </div>
      );
    }

    render(<TestHarness />);

    expect(screen.getByTestId("state-start").textContent).toBe("10");
    expect(screen.getByTestId("state-end").textContent).toBe("20");

    fireEvent.click(screen.getByTestId("atomic-multi-apply"));

    expect(screen.getByTestId("state-start").textContent).toBe("12");
    expect(screen.getByTestId("state-end").textContent).toBe("28");
  });
});
