import { scoreColor } from "../lib/format";

export default function ScoreBadge({ score }: { score: number }) {
  const color = scoreColor(score);
  return (
    <span className="score-badge" style={{ ["--c" as string]: color }}>
      <span className="ring" style={{ ["--p" as string]: score }}>
        <i>{score}</i>
      </span>
      <span>virality</span>
    </span>
  );
}
