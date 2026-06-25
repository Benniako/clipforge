import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { Clip } from "../lib/types";
import { fmtDuration } from "../lib/format";
import { useT } from "../lib/i18n";
import ScoreBadge from "./ScoreBadge";

interface Props {
  clip: Clip;
  projectId: string;
  rank?: number;
  selected?: boolean;
  onToggleSelect?: (id: string) => void;
}

// A small "Top pick" ribbon for the strongest few clips — the at-a-glance cue
// every social clipper uses so you know what to post first.
function RankRibbon({ rank }: { rank?: number }) {
  const { t } = useT();
  if (!rank || rank > 3) return null;
  const label = rank === 1 ? t("cc.topPick") : t("cc.rank", { rank });
  return (
    <span className={"rank-ribbon r" + rank} title={t("cc.rankTitle")}>
      {rank === 1 ? "★ " : ""}
      {label}
    </span>
  );
}

function StatusChip({ status }: { status: Clip["status"] }) {
  const { t } = useT();
  if (status === "ready") return null;
  const map: Record<string, [string, string]> = {
    rendering: [t("cc.statusRendering"), "var(--warn)"],
    pending: [t("cc.statusPending"), "var(--muted)"],
    failed: [t("cc.statusFailed"), "var(--bad)"],
  };
  const [label, color] = map[status] ?? [status, "var(--muted)"];
  return (
    <span className="status-chip" style={{ color }}>
      {label}
    </span>
  );
}

export default function ClipCard({ clip, projectId, rank, selected, onToggleSelect }: Props) {
  const { t } = useT();
  const nav = useNavigate();
  const open = () => nav(`/p/${projectId}/clip/${clip.id}`);
  const ready = clip.status === "ready" && clip.export_url;
  const [fb, setFb] = useState<"up" | "down" | null>(clip.feedback);

  const rate = async (e: React.MouseEvent, r: "up" | "down") => {
    e.stopPropagation();
    const next = fb === r ? "none" : r;
    setFb(next === "none" ? null : (next as "up" | "down"));
    await api.rateClip(projectId, clip.id, next).catch(() => {});
  };

  return (
    <div className="clip-card" style={selected ? { outline: "2px solid var(--accent)" } : undefined}>
      <div
        className="thumb"
        style={clip.thumb_url ? { backgroundImage: `url(${clip.thumb_url})` } : undefined}
        onClick={open}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } }}
      >
        <StatusChip status={clip.status} />
        <RankRibbon rank={rank} />
        {ready && <span className="thumb-scrim" />}
        {onToggleSelect && ready && (
          <span
            onClick={(e) => {
              e.stopPropagation();
              onToggleSelect(clip.id);
            }}
            title={t("cc.selectTitle")}
            style={{
              position: "absolute", top: 8, right: 8, width: 26, height: 26,
              borderRadius: 6, display: "grid", placeItems: "center", fontSize: 15,
              background: selected ? "var(--accent)" : "rgba(0,0,0,0.55)",
              border: "1px solid rgba(255,255,255,0.3)", color: "#fff", cursor: "pointer",
            }}
          >
            {selected ? "✓" : ""}
          </span>
        )}
        {ready && (
          <span className="play">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white">
              <path d="M8 5v14l11-7z" />
            </svg>
          </span>
        )}
        <span className="dur">
          {fmtDuration(clip.tightened_duration ?? clip.end - clip.start)}
          {clip.tightened_duration != null && " " + t("cc.tightened")}
        </span>
      </div>
      <div className="clip-body">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <ScoreBadge score={clip.score} />
        </div>
        <div className="clip-title" onClick={open} role="button" tabIndex={0}
             onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } }}>
          {clip.title || t("cc.untitled")}
        </div>
        <div className="factors">
          {clip.factors.slice(0, 2).map((f, i) => (
            <span className="factor" key={i} title={f.detail}>
              {f.label}
            </span>
          ))}
        </div>
        <div className="card-actions">
          <button className="btn sm ghost" onClick={open}>
            {t("cc.edit")}
          </button>
          {ready && (
            <a
              className="btn sm"
              href={api.downloadClipUrl(projectId, clip.id)}
              download
            >
              {t("cc.download")}
            </a>
          )}
          <div className="spacer" style={{ flex: 1 }} />
          <button
            className="btn sm ghost"
            title={t("cc.likeTitle")}
            onClick={(e) => rate(e, "up")}
            style={{ padding: "7px 9px", color: fb === "up" ? "var(--good)" : undefined }}
          >
            {t("cc.like")}
          </button>
          <button
            className="btn sm ghost"
            title={t("cc.dislikeTitle")}
            onClick={(e) => rate(e, "down")}
            style={{ padding: "7px 9px", color: fb === "down" ? "var(--bad)" : undefined }}
          >
            {t("cc.dislike")}
          </button>
        </div>
      </div>
    </div>
  );
}


