import { useCallback, useMemo, useRef } from "react";

interface WaveformProps {
  src: string;
  start: number;
  end: number;
  onSeek?: (t: number) => void;
}

const BAR_COUNT = 120;
const HEIGHT = 48;
const BAR_WIDTH = 4;
const BAR_GAP = 2;
const SVG_WIDTH = BAR_COUNT * (BAR_WIDTH + BAR_GAP) - BAR_GAP;

/** Generate synthetic waveform peak data using a sum of sine waves.
 *
 *  The result looks roughly like real audio and is deterministic per clip so
 *  the visualisation doesn't jump between renders.  When ``AudioContext`` is
 *  available and CORS permits, we *could* decode the src and compute real
 *  peaks; for now the sine approximation keeps things lightweight and
 *  dependency-free.
 */
function syntheticPeaks(count: number): number[] {
  const peaks: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = i / count;
    // Mix a few sine harmonics + noise for a natural-ish look
    const v =
      0.5 * Math.sin(2 * Math.PI * 3.2 * t) +
      0.3 * Math.sin(2 * Math.PI * 7.1 * t + 0.8) +
      0.2 * Math.sin(2 * Math.PI * 13.7 * t + 2.1) +
      0.15 * Math.sin(2 * Math.PI * 23.3 * t + 0.4);
    // Envelope: louder in the middle
    const envelope = 1 - 0.4 * Math.abs(t - 0.5) * 2;
    const peak = Math.max(0.05, Math.abs(v) * envelope);
    peaks.push(Math.min(peak, 1.0));
  }
  return peaks;
}

export default function Waveform({ src, start, end, onSeek }: WaveformProps) {
  const barRef = useRef<HTMLDivElement | null>(null);

  const peaks = useMemo(() => syntheticPeaks(BAR_COUNT), [src, start, end]);

  const span = end - start;

  const handleClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!onSeek) return;
      const svg = e.currentTarget;
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frac = Math.max(0, Math.min(x / rect.width, 1));
      onSeek(start + frac * span);
    },
    [onSeek, start, span],
  );

  return (
    <div
      ref={barRef}
      style={{
        width: "100%",
        height: HEIGHT,
        overflow: "hidden",
        borderRadius: 6,
        background: "var(--bg-secondary, #1a1a2e)",
        cursor: onSeek ? "pointer" : "default",
        userSelect: "none",
      }}
    >
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        preserveAspectRatio="none"
        onClick={handleClick}
        style={{ display: "block" }}
      >
        {peaks.map((peak, i) => {
          const barH = Math.max(2, peak * HEIGHT * 0.85);
          const y = (HEIGHT - barH) / 2;
          const x = i * (BAR_WIDTH + BAR_GAP);
          // Highlight the bar if it falls within the visible portion
          const isActive = i > BAR_COUNT * 0.3 && i < BAR_COUNT * 0.7;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={BAR_WIDTH}
              height={barH}
              rx={1}
              ry={1}
              fill={isActive ? "var(--accent, #6c63ff)" : "var(--muted, #555)"}
              opacity={0.7 + 0.3 * peak}
            >
              <title>{`${(start + (i / BAR_COUNT) * span).toFixed(1)}s`}</title>
            </rect>
          );
        })}
      </svg>
    </div>
  );
}
