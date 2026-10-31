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
} from "../components/AudioWaveformHeatmap";

import AiSuggestionsCoPilot, {
  type AiSuggestion,
} from "../components/AiSuggestionsCoPilot";

// ============================================================================
// Happy-DOM Setup & Browser API Mocks
// ============================================================================

// Mock HTMLMediaElement
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

// Mock PointerCapture
HTMLElement.prototype.setPointerCapture = vi.fn();
HTMLElement.prototype.releasePointerCapture = vi.fn();

// Mock getBoundingClientRect for waveform track
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (
    this: HTMLElement
  ) {
    if (this.getAttribute("data-testid") === "waveform-track") {
      return {
        width: 600,
        height: 96,
        top: 100,
        left: 50,
        bottom: 196,
        right: 650,
        x: 50,
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
// Test Fixtures
// ============================================================================

const MOCK_WAVEFORM_DATA: WaveformData = {
  clip_id: "clip_test_1",
  duration_s: 20.0,
  samples: [0.12, 0.45, 0.82, 0.96, 0.65, 0.38, 0.22, 0.74, 0.91, 0.53],
  dbfs: [-28.0, -18.5, -8.2, -2.4, -9.0, -22.0, -32.5, -7.1, -1.8, -14.2],
  climax_t: 12.5,
  events: [
    {
      type: "kill_streak",
      t: 6.0,
      duration: 2.5,
      color: COLOR_KILL_STREAK,
      label: "Triple Kill Streak",
    },
    {
      type: "scream",
      t: 12.5,
      duration: 1.2,
      color: COLOR_SCREAM,
      label: "Loud Scream (98 dB)",
    },
    {
      type: "climax",
      t: 12.5,
      duration: 1.5,
      color: COLOR_CLIMAX,
      label: "Peak Climax Moment",
    },
    {
      type: "crowd_roar",
      t: 15.0,
      duration: 3.0,
      color: COLOR_CROWD_ROAR,
      label: "Crowd Cheering",
    },
  ],
};

const MOCK_STORYBOARD_DATA: StoryboardData = {
  clip_id: "clip_test_1",
  sprite_url: "/media/p1/clips/clip_test_1_storyboard.jpg",
  cols: 5,
  rows: 2,
  tile_w: 160,
  tile_h: 90,
  interval_s: 2.0,
  total_frames: 10,
  duration_s: 20.0,
};

const MOCK_SUGGESTIONS: AiSuggestion[] = [
  {
    id: "sug_trim_hook_1",
    type: "trim_hook",
    title: "Cut Dead Air Hook",
    description: "Trim initial 0.8s silence to hook viewers instantly.",
    retention_impact: "+18% retention",
    patch: { start_t: 12.8 },
  },
  {
    id: "sug_hook_rating_1",
    type: "hook_rating",
    title: "Hook Strength: 88/100",
    description: "High vocal energy question opener. Climax reached in 6.2s.",
    retention_impact: "Strong Hook",
    patch: {},
  },
  {
    id: "sug_climax_teaser_1",
    type: "climax_teaser",
    title: "Cold-Open Climax Teaser Loop",
    description: "Peak virality moment detected at 12.5s. Tease climax for 2s before the main hook.",
    retention_impact: "+28% retention",
    patch: { teaser_loop: true },
  },
  {
    id: "sug_split_reframe_1",
    type: "reframe_layout",
    title: "Switch to Split Facecam",
    description: "Detected streamer facecam. Show gameplay and streamer reactions simultaneously.",
    retention_impact: "+24% engagement",
    patch: { layout: "split" },
  },
  {
    id: "sug_tighten_1",
    type: "tighten_dead_air",
    title: "Smart Jump Cuts (Tighten Dead Air)",
    description: "Cut 1.4s internal dead air across 2 jump cuts to sustain momentum.",
    retention_impact: "+12% retention",
    patch: { tighten: true, segments: [[12.8, 16.0], [17.4, 20.0]] },
  },
];

// ============================================================================
// 1. Decibel Band Color & Gradient Tests
// ============================================================================

describe("Decibel Color Band Logic (R2)", () => {
  it("assigns blue for quiet audio below -25 dB", () => {
    expect(getDecibelColor(-35)).toBe("#3B82F6");
    expect(getDecibelColor(-25.1)).toBe("#3B82F6");
  });

  it("assigns green for normal dialogue between -25 dB and -12 dB", () => {
    expect(getDecibelColor(-25.0)).toBe("#10B981");
    expect(getDecibelColor(-18.0)).toBe("#10B981");
    expect(getDecibelColor(-12.1)).toBe("#10B981");
  });

  it("assigns amber for high energy between -12 dB and -4 dB", () => {
    expect(getDecibelColor(-12.0)).toBe("#F59E0B");
    expect(getDecibelColor(-6.0)).toBe("#F59E0B");
    expect(getDecibelColor(-4.1)).toBe("#F59E0B");
  });

  it("assigns red for peaks at or louder than -4 dB", () => {
    expect(getDecibelColor(-4.0)).toBe("#EF4444");
    expect(getDecibelColor(-1.5)).toBe("#EF4444");
    expect(getDecibelColor(0.0)).toBe("#EF4444");
  });
});

// ============================================================================
// 2. AudioWaveformHeatmap Component Rendering & Event Markers
// ============================================================================

describe("AudioWaveformHeatmap Component (R2)", () => {
  it("renders waveform track with decibel gradient bars and timeline axis", () => {
    render(
      <AudioWaveformHeatmap
        start={10.0}
        end={30.0}
        currentTime={15.0}
        waveformData={MOCK_WAVEFORM_DATA}
      />
    );

    expect(screen.getByTestId("audio-waveform-heatmap")).toBeTruthy();
    expect(screen.getByTestId("waveform-track")).toBeTruthy();
    expect(screen.getByTestId("waveform-svg")).toBeTruthy();

    // Check timeline axis and ticks
    expect(screen.getByTestId("timeline-axis")).toBeTruthy();
    const ticks = screen.getAllByTestId("timeline-tick");
    expect(ticks.length).toBeGreaterThan(0);

    // Check playhead position
    const playhead = screen.getByTestId("waveform-playhead");
    expect(playhead).toBeTruthy();
    // currentTime = 15.0, start = 10.0, relative = 5.0, duration = 20.0 -> 25%
    expect(playhead.style.left).toBe("25%");
  });

  it("renders color-coded event markers for screams, crowd roars, kill streaks, and climax", () => {
    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={MOCK_WAVEFORM_DATA}
      />
    );

    const markers = screen.getAllByTestId("waveform-event-marker");
    expect(markers.length).toBe(4);

    // Verify individual marker color coding
    const screamMarker = markers.find((m) => m.getAttribute("data-event-type") === "scream");
    expect(screamMarker).toBeTruthy();
    expect(screamMarker?.getAttribute("data-event-color")).toBe(COLOR_SCREAM);

    const roarMarker = markers.find((m) => m.getAttribute("data-event-type") === "crowd_roar");
    expect(roarMarker).toBeTruthy();
    expect(roarMarker?.getAttribute("data-event-color")).toBe(COLOR_CROWD_ROAR);

    const killMarker = markers.find((m) => m.getAttribute("data-event-type") === "kill_streak");
    expect(killMarker).toBeTruthy();
    expect(killMarker?.getAttribute("data-event-color")).toBe(COLOR_KILL_STREAK);

    const climaxMarker = markers.find((m) => m.getAttribute("data-event-type") === "climax");
    expect(climaxMarker).toBeTruthy();
    expect(climaxMarker?.getAttribute("data-event-color")).toBe(COLOR_CLIMAX);
  });

  it("renders background highlight bands corresponding to climax events", () => {
    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={MOCK_WAVEFORM_DATA}
      />
    );

    const bands = screen.getAllByTestId("waveform-event-band");
    expect(bands.length).toBe(4);
  });
});

// ============================================================================
// 3. Smart Scrubbing & Hover Thumbnail Previews (R2)
// ============================================================================

describe("Smart Scrubbing & Hover Previews (R2)", () => {
  it("displays floating thumbnail preview tooltip on hover with timestamp and sprite tile", () => {
    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={MOCK_WAVEFORM_DATA}
        storyboardData={MOCK_STORYBOARD_DATA}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Initially tooltip is not visible
    expect(screen.queryByTestId("scrub-preview-tooltip")).toBeNull();

    // Hover over track at clientX = 200 (x = 150px into 600px width -> frac = 0.25 -> 5.0s)
    fireEvent.pointerEnter(track);
    fireEvent.pointerMove(track, { clientX: 200, clientY: 120 });

    const tooltip = screen.getByTestId("scrub-preview-tooltip");
    expect(tooltip).toBeTruthy();
    expect(tooltip.style.left).toBe("25%");

    // Verify timestamp formatting
    const timeBadge = screen.getByTestId("scrub-timestamp");
    expect(timeBadge.textContent).toContain("5.0s");

    // Verify thumbnail tile styling
    const thumb = screen.getByTestId("scrub-thumbnail");
    expect(thumb.style.backgroundImage).toContain(MOCK_STORYBOARD_DATA.sprite_url);

    // Mouse leave hides tooltip
    fireEvent.pointerLeave(track);
    expect(screen.queryByTestId("scrub-preview-tooltip")).toBeNull();
  });

  it("click-to-seek invokes onSeek with exact absolute and relative timestamps", () => {
    const handleSeek = vi.fn();

    render(
      <AudioWaveformHeatmap
        start={10.0}
        end={30.0}
        waveformData={MOCK_WAVEFORM_DATA}
        onSeek={handleSeek}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Click at clientX = 350 (left = 50, x = 300 / 600 = 0.5 -> rel = 10.0s, abs = 20.0s)
    fireEvent.pointerDown(track, { clientX: 350, clientY: 130, pointerId: 1 });

    expect(handleSeek).toHaveBeenCalledTimes(1);
    const [absT, relT] = handleSeek.mock.calls[0];
    expect(absT).toBeCloseTo(20.0, 1);
    expect(relT).toBeCloseTo(10.0, 1);
  });

  it("dragging the playhead scrubs smoothly across the waveform", () => {
    const handleSeek = vi.fn();

    render(
      <AudioWaveformHeatmap
        start={0}
        end={20}
        waveformData={MOCK_WAVEFORM_DATA}
        onSeek={handleSeek}
      />
    );

    const track = screen.getByTestId("waveform-track");

    // Start drag at frac = 0.2 (clientX = 170)
    fireEvent.pointerDown(track, { clientX: 170, pointerId: 1 });
    expect(handleSeek).toHaveBeenCalledWith(expect.closeTo(4.0, 1), expect.closeTo(4.0, 1));

    // Drag to frac = 0.6 (clientX = 410)
    fireEvent.pointerMove(track, { clientX: 410, pointerId: 1 });
    expect(handleSeek).toHaveBeenLastCalledWith(expect.closeTo(12.0, 1), expect.closeTo(12.0, 1));

    // Release drag
    fireEvent.pointerUp(track, { clientX: 410, pointerId: 1 });
  });

  it("clicking an event marker seeks directly to the event timestamp", () => {
    const handleSeek = vi.fn();

    render(
      <AudioWaveformHeatmap
        start={10.0}
        end={30.0}
        waveformData={MOCK_WAVEFORM_DATA}
        onSeek={handleSeek}
      />
    );

    const markers = screen.getAllByTestId("waveform-event-marker");
    const screamMarker = markers.find((m) => m.getAttribute("data-event-type") === "scream");
    expect(screamMarker).toBeTruthy();

    // Climax scream is at t = 12.5s relative, so abs = 10.0 + 12.5 = 22.5s
    fireEvent.click(screamMarker!);

    expect(handleSeek).toHaveBeenCalledWith(22.5, 12.5);
  });
});

// ============================================================================
// 4. AiSuggestionsCoPilot Component & Cards (R3)
// ============================================================================

describe("AiSuggestionsCoPilot Component (R3)", () => {
  it("renders contextual AI insight cards with virality, retention estimates, and hook ratings", () => {
    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={MOCK_SUGGESTIONS}
      />
    );

    expect(screen.getByTestId("ai-suggestions-copilot")).toBeTruthy();

    const cards = screen.getAllByTestId("suggestion-card");
    expect(cards.length).toBe(5);

    // Verify silence trimming card
    expect(screen.getByText("Cut Dead Air Hook")).toBeTruthy();
    expect(screen.getByText(/Trim initial 0.8s silence/)).toBeTruthy();
    expect(screen.getByText(/18% retention/)).toBeTruthy();

    // Verify hook rating card & score meter
    expect(screen.getByText("Hook Strength: 88/100")).toBeTruthy();
    expect(screen.getByTestId("hook-rating-score").textContent).toContain("88 / 100");
    expect(screen.getByTestId("hook-rating-meter")).toBeTruthy();

    // Verify peak virality teaser card
    expect(screen.getByText("Cold-Open Climax Teaser Loop")).toBeTruthy();
    expect(screen.getByText(/28% retention/)).toBeTruthy();

    // Verify split reframe card
    expect(screen.getByText("Switch to Split Facecam")).toBeTruthy();
    expect(screen.getByText(/24% engagement/)).toBeTruthy();
  });

  it("clicking Apply Recommendation updates state and displays visual confirmation badge", async () => {
    const handleApply = vi.fn();

    // Mock fetch for suggestion apply endpoint
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        clip: { id: "c1", start: 12.8 },
        updated_fields: ["start_t"],
      }),
    });
    global.fetch = mockFetch;

    render(
      <AiSuggestionsCoPilot
        projectId="proj_1"
        clipId="clip_1"
        suggestions={MOCK_SUGGESTIONS}
        onApplySuggestion={handleApply}
      />
    );

    const applyButtons = screen.getAllByTestId("apply-suggestion-btn");
    expect(applyButtons.length).toBeGreaterThan(0);

    // Click apply on first suggestion (Cut Dead Air Hook)
    fireEvent.click(applyButtons[0]);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        "/api/projects/proj_1/clips/clip_1/suggestions/sug_trim_hook_1/apply",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patch: { start_t: 12.8 } }),
        })
      );
    });

    // Check callback was invoked
    expect(handleApply).toHaveBeenCalledWith(
      MOCK_SUGGESTIONS[0],
      expect.objectContaining({ start: 12.8 })
    );

    // Check visual confirmation badge
    const badge = await screen.findByTestId("applied-confirmation-badge");
    expect(badge).toBeTruthy();
    expect(badge.textContent).toContain("Applied!");
  });

  it("allows dismissing an individual suggestion card", () => {
    render(
      <AiSuggestionsCoPilot
        projectId="p1"
        clipId="c1"
        suggestions={MOCK_SUGGESTIONS}
      />
    );

    expect(screen.getByText("Cut Dead Air Hook")).toBeTruthy();

    const dismissButtons = screen.getAllByText("Dismiss");
    fireEvent.click(dismissButtons[0]);

    expect(screen.queryByText("Cut Dead Air Hook")).toBeNull();
    expect(screen.getAllByTestId("suggestion-card").length).toBe(4);
  });
});

// ============================================================================
// 5. Integrated Studio Workflow (Waveform + AI Co-Pilot State Synchronization)
// ============================================================================

describe("Integrated Studio Workflow & State Synchronization (R2 + R3)", () => {
  function IntegratedEditorHarness() {
    const [start, setStart] = useState(10.0);
    const [end, setEnd] = useState(30.0);
    const [layout, setLayout] = useState("center");
    const [currentTime, setCurrentTime] = useState(10.0);
    const [teaserLoop, setTeaserLoop] = useState(false);

    return (
      <div data-testid="editor-harness">
        <div data-testid="readout-start">{start}</div>
        <div data-testid="readout-end">{end}</div>
        <div data-testid="readout-layout">{layout}</div>
        <div data-testid="readout-time">{currentTime}</div>
        <div data-testid="readout-teaser">{teaserLoop ? "true" : "false"}</div>

        <AudioWaveformHeatmap
          start={start}
          end={end}
          currentTime={currentTime}
          duration={end - start}
          waveformData={MOCK_WAVEFORM_DATA}
          onSeek={(absT) => {
            setCurrentTime(absT);
          }}
        />

        <AiSuggestionsCoPilot
          projectId="proj_harness"
          clipId="clip_harness"
          suggestions={MOCK_SUGGESTIONS}
          onApplySuggestion={(sug) => {
            const patch = sug.patch || {};
            if (patch.start_t !== undefined) setStart(patch.start_t);
            if (patch.end_t !== undefined) setEnd(patch.end_t);
            if (patch.layout) setLayout(patch.layout);
            if (patch.teaser_loop) setTeaserLoop(true);
          }}
        />
      </div>
    );
  }

  it("seeks playback time when scrubbing the waveform", () => {
    render(<IntegratedEditorHarness />);

    const track = screen.getByTestId("waveform-track");

    // Click at 50% across span (start = 10, end = 30 -> 20.0s)
    fireEvent.pointerDown(track, { clientX: 350, pointerId: 1 });

    expect(screen.getByTestId("readout-time").textContent).toBe("20");
  });

  it("atomically updates clip boundary and layout when AI suggestions are applied", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });

    render(<IntegratedEditorHarness />);

    // Verify initial state
    expect(screen.getByTestId("readout-start").textContent).toBe("10");
    expect(screen.getByTestId("readout-layout").textContent).toBe("center");
    expect(screen.getByTestId("readout-teaser").textContent).toBe("false");

    // Apply hook trimming suggestion
    const applyButtons = screen.getAllByTestId("apply-suggestion-btn");
    fireEvent.click(applyButtons[0]);

    await waitFor(() => {
      expect(screen.getByTestId("readout-start").textContent).toBe("12.8");
    });

    // Apply cold-open teaser loop suggestion
    const teaserCard = screen.getByText("Cold-Open Climax Teaser Loop").closest("[data-testid='suggestion-card']");
    const teaserBtn = teaserCard?.querySelector("[data-testid='apply-suggestion-btn']");
    fireEvent.click(teaserBtn!);

    await waitFor(() => {
      expect(screen.getByTestId("readout-teaser").textContent).toBe("true");
    });

    // Apply split reframe suggestion
    const reframeCard = screen.getByText("Switch to Split Facecam").closest("[data-testid='suggestion-card']");
    const reframeBtn = reframeCard?.querySelector("[data-testid='apply-suggestion-btn']");
    fireEvent.click(reframeBtn!);

    await waitFor(() => {
      expect(screen.getByTestId("readout-layout").textContent).toBe("split");
    });
  });
});
