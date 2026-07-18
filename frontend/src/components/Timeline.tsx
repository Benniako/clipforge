import { useCallback, useEffect, useRef, useState } from "react";
import type { TimelineData } from "../lib/types";
import { api } from "../lib/api";

interface TimelineProps {
  projectId: string;
  currentTime: number;
  duration: number;
  onSeek?: (t: number) => void;
}

const MIN_PX_PER_SECOND = 20;
const MAX_PX_PER_SECOND = 400;
const DEFAULT_PX_PER_SECOND = 60;
const WAVEFORM_HEIGHT = 48;
const WORD_ROW_HEIGHT = 28;
const MARKER_AREA_HEIGHT = 16;
const TRACK_HEIGHT = WAVEFORM_HEIGHT + WORD_ROW_HEIGHT + MARKER_AREA_HEIGHT;

function syntheticWaveformFromEmotion(emotion: number[], count: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / count;
    const idx = Math.min(Math.floor(t * emotion.length), emotion.length - 1);
    const e = emotion[idx] ?? 0.5;
    const v =
      0.5 * Math.sin(2 * Math.PI * 3.2 * t) +
      0.3 * Math.sin(2 * Math.PI * 7.1 * t + 0.8) +
      0.2 * Math.sin(2 * Math.PI * 13.7 * t + 2.1);
    const peak = Math.max(0.05, Math.abs(v) * (0.4 + e * 0.6));
    out.push(Math.min(peak, 1.0));
  }
  return out;
}

const CLIP_COLORS = ["#6c63ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff922b"];

export default function Timeline({ projectId, currentTime, duration, onSeek }: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<TimelineData | null>(null);
  const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SECOND);
  const [scrollLeft, setScrollLeft] = useState(0);
  const dragRef = useRef<{ startX: number; startScroll: number } | null>(null);

  useEffect(() => {
    if (!projectId) return;
    let live = true;
    api
      .timeline(projectId)
      .then((d) => {
        if (live) setData(d);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [projectId]);

  const totalWidth = duration * pxPerSec;
  const waveCount = Math.max(60, Math.floor(totalWidth / 6));
  const wavePeaks = data?.emotion_curve
    ? syntheticWaveformFromEmotion(data.emotion_curve, waveCount)
    : syntheticWaveformFromEmotion([0.5], waveCount);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const mouseFrac = (e.clientX - rect.left + scrollLeft) / totalWidth;
        const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
        setPxPerSec((prev) => {
          const next = Math.max(MIN_PX_PER_SECOND, Math.min(MAX_PX_PER_SECOND, prev * factor));
          return next;
        });
      } else if (containerRef.current) {
        containerRef.current.scrollLeft += e.deltaY;
      }
    },
    [scrollLeft, totalWidth],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragRef.current = { startX: e.clientX, startScroll: containerRef.current?.scrollLeft ?? 0 };
      const onMove = (ev: MouseEvent) => {
        if (!dragRef.current) return;
        const dx = ev.clientX - dragRef.current.startX;
        if (containerRef.current) {
          containerRef.current.scrollLeft = dragRef.current.startScroll - dx;
        }
      };
      const onUp = () => {
        dragRef.current = null;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [],
  );

  const handleClickTrack = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!onSeek || dragRef.current) return;
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left + (containerRef.current?.scrollLeft ?? 0);
      const time = Math.max(0, Math.min(x / pxPerSec, duration));
      onSeek(time);
    },
    [onSeek, pxPerSec, duration],
  );

  const playheadX = currentTime * pxPerSec;

  return (
    <div
      ref={containerRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      style={{
        width: "100%",
        overflowX: "auto",
        overflowY: "hidden",
        borderRadius: 8,
        background: "var(--bg-secondary, #1a1a2e)",
        position: "relative",
        userSelect: "none",
        cursor: "grab",
      }}
    >
      <div style={{ width: totalWidth, minHeight: TRACK_HEIGHT + 16, position: "relative" }}>
        {/* Waveform + word track */}
        <svg
          width={totalWidth}
          height={TRACK_HEIGHT}
          onClick={handleClickTrack}
          style={{ display: "block", cursor: onSeek ? "pointer" : "default" }}
        >
          {/* Speech interval markers (green bars at bottom) */}
          {data?.speech_intervals.map(([s, e], i) => (
            <rect
              key={`speech-${i}`}
              x={s * pxPerSec}
              y={WAVEFORM_HEIGHT + WORD_ROW_HEIGHT}
              width={(e - s) * pxPerSec}
              height={MARKER_AREA_HEIGHT}
              fill="rgba(107,203,119,0.35)"
              rx={2}
            />
          ))}

          {/* Clip regions */}
          {data?.clips.map((clip, i) => {
            const x = clip.start * pxPerSec;
            const w = (clip.end - clip.start) * pxPerSec;
            const color = CLIP_COLORS[i % CLIP_COLORS.length];
            return (
              <g key={clip.id}>
                <rect
                  x={x}
                  y={0}
                  width={w}
                  height={WAVEFORM_HEIGHT}
                  fill={color}
                  opacity={0.12}
                />
                <rect
                  x={x}
                  y={0}
                  width={w}
                  height={WAVEFORM_HEIGHT}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                  opacity={0.5}
                />
              </g>
            );
          })}

          {/* Waveform bars */}
          {wavePeaks.map((peak, i) => {
            const barH = Math.max(2, peak * WAVEFORM_HEIGHT * 0.85);
            const x = (i / waveCount) * totalWidth;
            const barW = Math.max(2, totalWidth / waveCount - 1);
            return (
              <rect
                key={i}
                x={x}
                y={(WAVEFORM_HEIGHT - barH) / 2}
                width={barW}
                height={barH}
                rx={1}
                fill="var(--accent, #6c63ff)"
                opacity={0.5 + 0.5 * peak}
              />
            );
          })}

          {/* Word text row */}
          {data?.words.map((w, i) => {
            const x = w.t * pxPerSec;
            const wordWidth = Math.max((w.d * pxPerSec) - 2, 12);
            return (
              <text
                key={i}
                x={x + 2}
                y={WAVEFORM_HEIGHT + 18}
                fontSize={12}
                fill="var(--text, #ccc)"
                style={{ pointerEvents: "none" }}
              >
                {w.text.length > Math.floor(wordWidth / 7) ? w.text.slice(0, Math.floor(wordWidth / 7)) + "…" : w.text}
              </text>
            );
          })}

          {/* Scene cut markers (red vertical lines) */}
          {data?.scene_cuts.map((t, i) => (
            <line
              key={`cut-${i}`}
              x1={t * pxPerSec}
              y1={0}
              x2={t * pxPerSec}
              y2={TRACK_HEIGHT}
              stroke="var(--bad, #ff4444)"
              strokeWidth={1.5}
              opacity={0.7}
            />
          ))}

          {/* Playhead */}
          <line
            x1={playheadX}
            y1={0}
            x2={playheadX}
            y2={TRACK_HEIGHT}
            stroke="var(--text, #fff)"
            strokeWidth={2}
            opacity={0.9}
          />
        </svg>

        {/* Clip score badges (HTML overlay for text rendering) */}
        {data?.clips.map((clip, i) => {
          const x = clip.start * pxPerSec;
          const w = (clip.end - clip.start) * pxPerSec;
          if (w < 30) return null;
          const color = CLIP_COLORS[i % CLIP_COLORS.length];
          return (
            <div
              key={clip.id}
              style={{
                position: "absolute",
                left: x + 4,
                top: 4,
                fontSize: 10,
                fontWeight: 600,
                color: "#fff",
                background: color,
                borderRadius: 4,
                padding: "1px 5px",
                pointerEvents: "none",
                opacity: 0.85,
              }}
            >
              {clip.score}
            </div>
          );
        })}
      </div>
    </div>
  );
}
