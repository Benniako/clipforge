import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import type { Clip } from "../lib/types";
import { fmtClock, fmtDuration } from "../lib/format";
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
  const [copied, setCopied] = useState(false);

  const rate = async (e: React.MouseEvent, r: "up" | "down") => {
    e.stopPropagation();
    const next = fb === r ? "none" : r;
    setFb(next === "none" ? null : (next as "up" | "down"));
    await api.rateClip(projectId, clip.id, next).catch(() => {});
  };

  const copyCaption = (e: React.MouseEvent) => {
    e.stopPropagation();
    const tags = clip.hashtags && clip.hashtags.length
      ? clip.hashtags.join(" ")
      : "#gaming #fyp #viral #streamer";
    const text = `${clip.title || "Stream Highlight"}\n\n${tags}`;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={"clip-card" + (selected ? " selected" : "")}>
      <div
        className="thumb"
        style={clip.thumb_url ? { backgroundImage: `url(${clip.thumb_url})` } : undefined}
        onClick={open}
        role="button"
        tabIndex={0}
        aria-label={clip.title || t("cc.untitled")}
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
            role="checkbox"
            aria-checked={!!selected}
            title={t("cc.selectTitle")}
            className={"thumb-select" + (selected ? " on" : "")}
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
        <div className="clip-top">
          <ScoreBadge score={clip.score} />
          <div className="rate-pair">
            <button
              className={"rate-btn up" + (fb === "up" ? " on" : "")}
              title={t("cc.likeTitle")}
              aria-pressed={fb === "up"}
              onClick={(e) => rate(e, "up")}
            >
              {t("cc.like")}
            </button>
            <button
              className={"rate-btn down" + (fb === "down" ? " on" : "")}
              title={t("cc.dislikeTitle")}
              aria-pressed={fb === "down"}
              onClick={(e) => rate(e, "down")}
            >
              {t("cc.dislike")}
            </button>
          </div>
        </div>
        <div className="clip-title" onClick={open} role="button" tabIndex={0}
             onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } }}>
          {clip.title || t("cc.untitled")}
        </div>
        <div className="clip-span muted tiny">
          {t("cc.inSource", { start: fmtClock(clip.start), end: fmtClock(clip.end) })}
        </div>
        {clip.factors.length > 0 && (
          <div className="factors inline">
            {clip.factors.slice(0, 3).map((f, i) => (
              <span className="factor" key={i} title={f.detail}>
                {f.label}
                <span className="w">+{Math.round(f.weight)}</span>
              </span>
            ))}
          </div>
        )}
        {/* Quick trim: drag the two ends to nudge clip start/end before re-render. */}
        {ready && (
          <details className="clip-trim">
            <summary>{t("cc.trim")}</summary>
            <div className="clip-trim-body">
              <label>
                <span className="muted tiny">{t("cc.trimStart")} {fmtClock(clip.start)}</span>
                <input
                  type="range"
                  min={Math.max(0, clip.start - 5)}
                  max={Math.max(clip.start + 1, clip.end - 3)}
                  step={0.1}
                  value={clip.start}
                  onChange={(e) => { void api.trimClip(projectId, clip.id, { start: Number(e.target.value) }); }}
                  title={t("cc.trimStart")}
                />
              </label>
              <label>
                <span className="muted tiny">{t("cc.trimEnd")} {fmtClock(clip.end)}</span>
                <input
                  type="range"
                  min={Math.min(clip.start + 3, clip.end + 5)}
                  max={clip.end + 5}
                  step={0.1}
                  value={clip.end}
                  onChange={(e) => { void api.trimClip(projectId, clip.id, { end: Number(e.target.value) }); }}
                  title={t("cc.trimEnd")}
                />
              </label>
            </div>
          </details>
        )}
        <div className="card-actions">
          <button className="btn sm ghost" onClick={open}>
            {t("cc.edit")}
          </button>
          {ready && (
            <a
              className="btn sm primary"
              href={api.downloadClipUrl(projectId, clip.id)}
              download
            >
              {t("cc.download")}
            </a>
          )}
          {ready && (
            <a
              className="btn sm ghost"
              href={api.downloadSrtUrl(projectId, clip.id)}
              download
              title={t("cc.srtTitle")}
            >
              .srt
            </a>
          )}
          <div className="spacer" />
          {ready && (
            <button
              className={"copy-tiktok-btn" + (copied ? " copied" : "")}
              onClick={copyCaption}
              title={t("cc.copyCaptionTitle")}
            >
              {copied ? t("cc.copied") : t("cc.copyCaption")}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
