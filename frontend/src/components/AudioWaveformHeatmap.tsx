import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ============================================================================
// Types & Contracts (PROJECT.md & R2 Interface Specification)
// ============================================================================

export interface WaveformEvent {
  type: "scream" | "crowd_roar" | "kill_streak" | "climax" | string;
  t: number; // timestamp in seconds (relative to clip start)
  duration: number; // duration in seconds
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

export interface StoryboardData {
  clip_id: string;
  sprite_url: string;
  cols: number;
  rows: number;
  tile_w: number;
  tile_h: number;
  interval_s: number;
  total_frames: number;
  duration_s: number;
}

export interface AudioWaveformHeatmapProps {
  projectId?: string;
  clipId?: string;
  start: number;
  end: number;
  currentTime?: number; // playback time in seconds
  duration?: number;
  waveformData?: WaveformData;
  storyboardData?: StoryboardData;
  videoSrc?: string;
  previewMode?: "rendered" | "original";
  onSeek?: (absoluteTime: number, relativeTime: number) => void;
  className?: string;
  style?: React.CSSProperties;
  height?: number;
}

// Color constants matching R2 and PROJECT.md
export const COLOR_SCREAM = "#FF4500";
export const COLOR_CROWD_ROAR = "#9D00FF";
export const COLOR_KILL_STREAK = "#00F0FF";
export const COLOR_CLIMAX = "#EC4899";

/**
 * Waveform Decibel Band Color Generator:
 * < -25 dB: Quiet (Blue #3B82F6)
 * -25 to -12 dB: Normal speech / dialogue (Green #10B981)
 * -12 to -4 dB: High energy / action (Amber #F59E0B)
 * >= -4 dB: Peak / Loudest (Red #EF4444)
 */
export function getDecibelColor(db: number): string {
  if (db < -25) return "#3B82F6";
  if (db < -12) return "#10B981";
  if (db < -4) return "#F59E0B";
  return "#EF4444";
}

/**
 * Synthetic fallback waveform generator for offline or initial loading states.
 */
function generateSyntheticWaveform(clipId: string, duration: number, numBins = 140): WaveformData {
  const safeDuration = Math.max(0.1, duration);
  const samples: number[] = [];
  const dbfs: number[] = [];
  const climax_t = Math.round(safeDuration * 0.65 * 10) / 10;

  for (let i = 0; i < numBins; i++) {
    const t = (i / numBins) * safeDuration;
    const dist = Math.abs(t - climax_t);
    const climaxBoost = Math.exp(-0.5 * Math.pow(dist / 1.5, 2)) * 0.45;
    const voiceHarmonics =
      0.35 * Math.sin(2 * Math.PI * 2.8 * (t / safeDuration)) +
      0.2 * Math.sin(2 * Math.PI * 6.5 * (t / safeDuration) + 1.2) +
      0.15 * Math.sin(2 * Math.PI * 14.1 * (t / safeDuration) + 0.5);
    const envelope = 0.5 + 0.3 * Math.sin((t / safeDuration) * Math.PI);
    const amp = Math.max(0.06, Math.min(0.98, Math.abs(voiceHarmonics) * envelope + climaxBoost));
    const db = Math.max(-60, Math.min(-0.5, 20 * Math.log10(amp)));

    samples.push(Math.round(amp * 1000) / 1000);
    dbfs.push(Math.round(db * 10) / 10);
  }

  const events: WaveformEvent[] = [
    {
      type: "kill_streak",
      t: Math.max(0, climax_t - 1.8),
      duration: 1.8,
      color: COLOR_KILL_STREAK,
      label: "Triple Kill Streak",
    },
    {
      type: "scream",
      t: climax_t,
      duration: 1.2,
      color: COLOR_SCREAM,
      label: "Loud Scream (98 dB)",
    },
    {
      type: "climax",
      t: climax_t,
      duration: 1.5,
      color: COLOR_CLIMAX,
      label: "Peak Climax Moment",
    },
    {
      type: "crowd_roar",
      t: Math.min(duration, climax_t + 1.2),
      duration: 2.2,
      color: COLOR_CROWD_ROAR,
      label: "Crowd Cheering",
    },
  ];

  return {
    clip_id: clipId,
    duration_s: duration,
    samples,
    dbfs,
    climax_t,
    events,
  };
}

export default function AudioWaveformHeatmap({
  projectId,
  clipId,
  start,
  end,
  currentTime = 0,
  duration,
  waveformData: propWaveformData,
  storyboardData: propStoryboardData,
  videoSrc,
  previewMode = "original",
  onSeek,
  className = "",
  style = {},
  height = 96,
}: AudioWaveformHeatmapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const span = Math.max(0.1, end - start);
  const totalDuration = Math.max(0.1, duration ?? span);

  // Local state for fetched or synthetic data
  const [data, setData] = useState<WaveformData>(() => {
    return (
      propWaveformData ||
      generateSyntheticWaveform(clipId || "default_clip", totalDuration)
    );
  });
  const [storyboard, setStoryboard] = useState<StoryboardData | null>(propStoryboardData || null);

  // Hover / scrubbing state
  const [isHovering, setIsHovering] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [hoverX, setHoverX] = useState(0);
  const [hoverFrac, setHoverFrac] = useState(0);
  const [activeEvent, setActiveEvent] = useState<WaveformEvent | null>(null);

  // Update data if prop changes
  useEffect(() => {
    if (propWaveformData) {
      setData(propWaveformData);
    }
  }, [propWaveformData]);

  useEffect(() => {
    if (propStoryboardData) {
      setStoryboard(propStoryboardData);
    }
  }, [propStoryboardData]);

  // Fetch from backend endpoints if not supplied via props
  useEffect(() => {
    if (propWaveformData || !projectId || !clipId) return;

    let mounted = true;
    fetch(`/api/projects/${projectId}/clips/${clipId}/waveform`)
      .then((res) => (res.ok ? res.json() : null))
      .then((json: WaveformData | null) => {
        if (mounted && json && json.samples && json.samples.length > 0) {
          setData(json);
        }
      })
      .catch(() => {
        // Fall back gracefully to synthetic waveform
      });

    return () => {
      mounted = false;
    };
  }, [projectId, clipId, propWaveformData]);

  useEffect(() => {
    if (propStoryboardData || !projectId || !clipId) return;

    let mounted = true;
    fetch(`/api/projects/${projectId}/clips/${clipId}/storyboard`)
      .then((res) => (res.ok ? res.json() : null))
      .then((json: StoryboardData | null) => {
        if (mounted && json && json.sprite_url) {
          setStoryboard(json);
        }
      })
      .catch(() => {
        // Fall back
      });

    return () => {
      mounted = false;
    };
  }, [projectId, clipId, propStoryboardData]);

  // Map currentTime to relative [0, totalDuration] respecting previewMode
  const currentRelTime = useMemo(() => {
    if (previewMode === "rendered") {
      return Math.max(0, Math.min(totalDuration, currentTime));
    }
    // original mode: currentTime is in [start, end]
    if (currentTime >= start && currentTime <= end) {
      return currentTime - start;
    }
    return Math.max(0, Math.min(totalDuration, currentTime - start));
  }, [currentTime, start, end, totalDuration, previewMode]);

  const playheadPct = useMemo(() => {
    return Math.max(0, Math.min(100, (currentRelTime / totalDuration) * 100));
  }, [currentRelTime, totalDuration]);

  // Calculate hover timestamp
  const hoverRelTime = hoverFrac * totalDuration;
  const hoverAbsTime = start + hoverFrac * span;

  // Calculate seek from mouse/pointer event
  const getEventFrac = useCallback((e: React.MouseEvent | React.PointerEvent) => {
    if (!containerRef.current) return 0;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    return rect.width > 0 ? x / rect.width : 0;
  }, []);

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const frac = getEventFrac(e);
    setIsDragging(true);
    setHoverFrac(frac);
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setHoverX(e.clientX - rect.left);
      containerRef.current.setPointerCapture(e.pointerId);
    }
    const relT = frac * totalDuration;
    const absT = start + frac * span;
    onSeek?.(absT, relT);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const frac = getEventFrac(e);
    setHoverFrac(frac);
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setHoverX(e.clientX - rect.left);
    }
    if (isDragging) {
      const relT = frac * totalDuration;
      const absT = start + frac * span;
      onSeek?.(absT, relT);
    }
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (isDragging) {
      setIsDragging(false);
      try {
        containerRef.current?.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    }
  };

  // Timeline second tick marks
  const tickInterval = totalDuration > 30 ? 5 : totalDuration > 10 ? 2 : 1;
  const ticks = useMemo(() => {
    const tList: { time: number; label: string; pct: number }[] = [];
    const count = Math.floor(totalDuration / tickInterval);
    for (let i = 0; i <= count; i++) {
      const t = i * tickInterval;
      tList.push({
        time: t,
        label: `${t.toFixed(0)}s`,
        pct: (t / totalDuration) * 100,
      });
    }
    return tList;
  }, [totalDuration, tickInterval]);

  // Storyboard tile coordinates for hover preview
  const storyboardSpriteStyle = useMemo(() => {
    if (!storyboard || !storyboard.sprite_url) return null;
    const cols = storyboard.cols || 5;
    const rows = storyboard.rows || 2;
    const totalFrames = storyboard.total_frames || cols * rows;
    const interval = storyboard.interval_s || totalDuration / totalFrames;
    const frameIdx = Math.min(totalFrames - 1, Math.max(0, Math.floor(hoverRelTime / Math.max(0.01, interval))));
    const col = frameIdx % cols;
    const row = Math.floor(frameIdx / cols);

    const tileW = 140;
    const tileH = Math.round(tileW * (9 / 16));

    return {
      backgroundImage: `url(${storyboard.sprite_url})`,
      backgroundPosition: `-${col * tileW}px -${row * tileH}px`,
      backgroundSize: `${cols * tileW}px ${rows * tileH}px`,
      width: tileW,
      height: tileH,
    };
  }, [storyboard, hoverRelTime, totalDuration]);

  // Render SVG waveform bars
  const samples = data.samples || [];
  const dbfs = data.dbfs || [];
  const barCount = samples.length;

  return (
    <div
      className={`audio-waveform-heatmap-wrap ${className}`}
      style={{
        position: "relative",
        userSelect: "none",
        width: "100%",
        marginBottom: 14,
        ...style,
      }}
      data-testid="audio-waveform-heatmap"
    >
      {/* Floating Scrubbing Thumbnail & Timestamp Tooltip */}
      {(isHovering || isDragging) && (
        <div
          data-testid="scrub-preview-tooltip"
          style={{
            position: "absolute",
            left: `${hoverFrac * 100}%`,
            bottom: "100%",
            transform:
              hoverFrac < 0.12
                ? "translateX(0%)"
                : hoverFrac > 0.88
                ? "translateX(-100%)"
                : "translateX(-50%)",
            marginBottom: 8,
            zIndex: 30,
            pointerEvents: "none",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          {/* Storyboard Frame Preview */}
          <div
            data-testid="scrub-thumbnail"
            style={{
              width: 140,
              height: 79,
              backgroundColor: "#000",
              borderRadius: 6,
              border: "2px solid var(--accent, #8b5cf6)",
              boxShadow: "0 6px 16px rgba(0,0,0,0.6)",
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 4,
              ...(storyboardSpriteStyle || {}),
            }}
          >
            {!storyboardSpriteStyle && (
              <span style={{ fontSize: 11, color: "var(--muted, #94a3b8)" }}>
                ▶ Frame {hoverRelTime.toFixed(1)}s
              </span>
            )}
          </div>
          {/* Time Badge */}
          <div
            data-testid="scrub-timestamp"
            style={{
              backgroundColor: "rgba(17, 22, 36, 0.95)",
              color: "#fff",
              padding: "2px 8px",
              borderRadius: 4,
              fontSize: 11,
              fontFamily: "monospace",
              border: "1px solid var(--border-2, #2b3752)",
              whiteSpace: "nowrap",
            }}
          >
            {hoverRelTime.toFixed(1)}s / {totalDuration.toFixed(1)}s
          </div>
        </div>
      )}

      {/* Main Interactive Waveform Track */}
      <div
        ref={containerRef}
        data-testid="waveform-track"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerEnter={() => setIsHovering(true)}
        onPointerLeave={(e) => {
          if (!isDragging) {
            setIsHovering(false);
          }
        }}
        style={{
          position: "relative",
          width: "100%",
          height: height,
          backgroundColor: "var(--panel, #111624)",
          borderRadius: 8,
          border: "1px solid var(--border, #1d2538)",
          overflow: "hidden",
          cursor: "crosshair",
        }}
      >
        {/* Color-Coded Event Highlights Background Bands */}
        {data.events.map((ev, i) => {
          const leftPct = Math.max(0, Math.min(100, (ev.t / totalDuration) * 100));
          const widthPct = Math.max(1.5, Math.min(100 - leftPct, (ev.duration / totalDuration) * 100));
          return (
            <div
              key={`band-${i}`}
              data-testid="waveform-event-band"
              data-event-type={ev.type}
              style={{
                position: "absolute",
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                top: 0,
                bottom: 0,
                backgroundColor: ev.color,
                opacity: 0.16,
                pointerEvents: "none",
              }}
            />
          );
        })}

        {/* SVG Waveform Bars with Decibel Gradients */}
        <svg
          data-testid="waveform-svg"
          viewBox={`0 0 ${barCount * 4} ${height - 20}`}
          preserveAspectRatio="none"
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: "100%",
            height: height - 20,
            display: "block",
          }}
        >
          {samples.map((amp, idx) => {
            const db = dbfs[idx] ?? -30;
            const safeAmp = Number.isFinite(amp) ? Math.max(0, Math.min(1, amp)) : 0.05;
            const barH = Math.max(3, safeAmp * (height - 24));
            const y = (height - 20 - barH) / 2;
            const x = idx * 4;
            const color = getDecibelColor(db);

            return (
              <rect
                key={idx}
                x={x}
                y={y}
                width={2.8}
                height={barH}
                rx={1.4}
                fill={color}
                opacity={0.85}
              />
            );
          })}
        </svg>

        {/* Event Marker Overlay Badges along top of waveform */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 4,
            right: 0,
            height: 20,
            pointerEvents: "none",
          }}
        >
          {data.events.map((ev, i) => {
            const leftPct = Math.max(0, Math.min(96, (ev.t / totalDuration) * 100));
            let badgeIcon = "⚡";
            if (ev.type === "scream") badgeIcon = "😱";
            else if (ev.type === "crowd_roar") badgeIcon = "📢";
            else if (ev.type === "climax") badgeIcon = "👑";

            return (
              <div
                key={`marker-${i}`}
                data-testid="waveform-event-marker"
                data-event-type={ev.type}
                data-event-color={ev.color}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek?.(start + ev.t, ev.t);
                }}
                onMouseEnter={() => setActiveEvent(ev)}
                onMouseLeave={() => setActiveEvent(null)}
                style={{
                  position: "absolute",
                  left: `${leftPct}%`,
                  transform: "translateX(-50%)",
                  backgroundColor: ev.color,
                  color: "#fff",
                  fontSize: 10,
                  fontWeight: 600,
                  padding: "1px 6px",
                  borderRadius: 10,
                  boxShadow: `0 0 10px ${ev.color}88`,
                  display: "flex",
                  alignItems: "center",
                  gap: 3,
                  cursor: "pointer",
                  pointerEvents: "auto",
                  whiteSpace: "nowrap",
                  zIndex: 10,
                }}
                title={`${ev.label} (${ev.t.toFixed(1)}s)`}
              >
                <span>{badgeIcon}</span>
                <span style={{ fontSize: 9 }}>{ev.type.replace("_", " ")}</span>
              </div>
            );
          })}
        </div>

        {/* Active Playhead Cursor Line */}
        <div
          data-testid="waveform-playhead"
          style={{
            position: "absolute",
            left: `${playheadPct}%`,
            top: 0,
            bottom: 0,
            width: 2,
            backgroundColor: "#fff",
            boxShadow: "0 0 8px #fff, 0 0 14px var(--accent, #8b5cf6)",
            pointerEvents: "none",
            zIndex: 20,
            transform: "translateX(-50%)",
          }}
        >
          {/* Playhead Top Knob */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: "50%",
              transform: "translate(-50%, -30%)",
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: "#fff",
              boxShadow: "0 0 6px #fff",
            }}
          />
        </div>

        {/* Hover Cursor Line */}
        {isHovering && !isDragging && (
          <div
            data-testid="waveform-hover-line"
            style={{
              position: "absolute",
              left: `${hoverFrac * 100}%`,
              top: 0,
              bottom: 0,
              width: 1,
              backgroundColor: "rgba(255,255,255,0.4)",
              pointerEvents: "none",
              zIndex: 15,
              transform: "translateX(-50%)",
            }}
          />
        )}

        {/* Timeline Axis with Second Tick Markers along bottom */}
        <div
          data-testid="timeline-axis"
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: 18,
            backgroundColor: "rgba(14, 18, 29, 0.75)",
            borderTop: "1px solid var(--border, #1d2538)",
            display: "flex",
            alignItems: "center",
            pointerEvents: "none",
          }}
        >
          {ticks.map((tick, i) => (
            <div
              key={i}
              data-testid="timeline-tick"
              style={{
                position: "absolute",
                left: `${tick.pct}%`,
                transform: "translateX(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <div
                style={{
                  width: 1,
                  height: 4,
                  backgroundColor: "var(--muted-2, #64748b)",
                }}
              />
              <span
                style={{
                  fontSize: 9,
                  fontFamily: "monospace",
                  color: "var(--muted, #94a3b8)",
                }}
              >
                {tick.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Heatmap Legend Bar below track */}
      <div
        className="row"
        style={{
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 6,
          fontSize: 11,
          color: "var(--muted, #94a3b8)",
        }}
      >
        <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
          <span className="row" style={{ gap: 4, alignItems: "center" }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: COLOR_SCREAM,
              }}
            />
            Scream
          </span>
          <span className="row" style={{ gap: 4, alignItems: "center" }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: COLOR_CROWD_ROAR,
              }}
            />
            Crowd Roar
          </span>
          <span className="row" style={{ gap: 4, alignItems: "center" }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: COLOR_KILL_STREAK,
              }}
            />
            Kill Streak
          </span>
          <span className="row" style={{ gap: 4, alignItems: "center" }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: COLOR_CLIMAX,
              }}
            />
            Climax
          </span>
        </div>
        <div style={{ fontFamily: "monospace" }}>
          {currentRelTime.toFixed(1)}s / {totalDuration.toFixed(1)}s
        </div>
      </div>
    </div>
  );
}
