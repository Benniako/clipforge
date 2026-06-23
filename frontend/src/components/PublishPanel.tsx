import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { PublishContent } from "../lib/types";

const PLATFORMS = ["generic", "tiktok", "reels", "shorts"];

/**
 * Publish-ready content panel: AI-generated titles, description, and hashtags
 * for a rendered clip, tailored per platform. Lives inside the ClipEditor as
 * a collapsible section.
 */
export default function PublishPanel({
  projectId,
  clipId,
}: {
  projectId: string;
  clipId: string;
}) {
  const [data, setData] = useState<PublishContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [platform, setPlatform] = useState("generic");
  const [copied, setCopied] = useState<string | null>(null);

  const load = (plat: string) => {
    setLoading(true);
    setErr(null);
    api
      .publishContent(projectId, clipId, plat)
      .then(setData)
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load(platform);
  }, [platform]); // eslint-disable-line react-hooks/exhaustive-deps

  const copy = (text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label);
      setTimeout(() => setCopied(null), 1800);
    });
  };

  const tags = data?.hashtags?.join(" ") ?? "";
  const fullDesc = [data?.description ?? "", "", tags].filter(Boolean).join("\n");

  if (loading && !data) {
    return (
      <div className="panel section">
        <h4>Generate publish content…</h4>
        <p className="muted tiny">
          <span className="spinner" /> Asking the local LLM…
        </p>
      </div>
    );
  }

  return (
    <div className="panel section">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <h4 style={{ margin: 0 }}>Publish</h4>
        <select
          className="input"
          style={{ width: "auto", fontSize: 12 }}
          value={platform}
          onChange={(e) => setPlatform(e.target.value)}
        >
          {PLATFORMS.map((p) => (
            <option key={p} value={p}>
              {p === "generic" ? "All platforms" : p.charAt(0).toUpperCase() + p.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {err && (
        <p className="tiny bad" style={{ marginBottom: 8 }}>
          {err}
        </p>
      )}
      {loading && <p className="tiny muted">Updating for {platform}…</p>}

      {data && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Titles */}
          <div>
            <label className="tiny muted" style={{ fontWeight: 700 }}>Titles</label>
            <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 4 }}>
              {data.titles.map((t, i) => (
                <div key={i} className="row" style={{ gap: 4 }}>
                  <span className="tiny muted" style={{ width: 18 }}>#{i + 1}</span>
                  <span style={{ flex: 1, fontSize: 14 }}>{t}</span>
                  <button
                    className="btn ghost sm"
                    onClick={() => copy(t, `title-${i}`)}
                  >
                    {copied === `title-${i}` ? "✓" : "📋"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Description */}
          <div>
            <label className="tiny muted" style={{ fontWeight: 700 }}>Description</label>
            <div className="row" style={{ gap: 4, marginTop: 4, alignItems: "flex-start" }}>
              <pre
                style={{
                  flex: 1, fontSize: 12, lineHeight: 1.4, whiteSpace: "pre-wrap",
                  background: "var(--bg)", padding: "6px 8px", borderRadius: 6,
                  margin: 0, maxHeight: 120, overflowY: "auto",
                }}
              >
                {fullDesc || "(no description)"}
              </pre>
              <button
                className="btn ghost sm"
                onClick={() => copy(fullDesc, "desc")}
                style={{ whiteSpace: "nowrap" }}
              >
                {copied === "desc" ? "✓" : "📋"}
              </button>
            </div>
          </div>

          {/* Hashtags */}
          <div>
            <label className="tiny muted" style={{ fontWeight: 700 }}>Hashtags</label>
            <div className="row" style={{ gap: 4, marginTop: 4 }}>
              <span style={{ flex: 1, fontSize: 13, color: "var(--accent)" }}>
                {tags || "(no tags)"}
              </span>
              <button
                className="btn ghost sm"
                onClick={() => copy(tags, "tags")}
              >
                {copied === "tags" ? "✓" : "📋"}
              </button>
            </div>
          </div>

          {/* Refresh */}
          <button className="btn ghost sm" onClick={() => load(platform)}>
            {loading ? "…" : "Re-generate"}
          </button>
        </div>
      )}
    </div>
  );
}
