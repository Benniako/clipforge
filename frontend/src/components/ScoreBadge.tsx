import { scoreColor } from "../lib/format";
import { useT } from "../lib/i18n";

export default function ScoreBadge({ score }: { score: number }) {
  const { t } = useT();
  const color = scoreColor(score);
  const icon = score >= 85 ? "🔥" : score >= 70 ? "⚡" : "⭐";
  return (
    <span className="score-badge" style={{ ["--c" as string]: color }}>
      <span className="ring" style={{ ["--p" as string]: score }}>
        <i>{score}</i>
      </span>
      <span className="score-label">
        <span className="score-icon">{icon}</span> {t("sb.virality")}
      </span>
    </span>
  );
}
