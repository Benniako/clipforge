import { useState } from "react";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";

export default function CueManager({
  game,
  cues,
  onChange,
}: {
  game: string;
  cues: CuesStatus | null;
  onChange: (c: CuesStatus) => void;
}) {
  const pack = cues?.[game];
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  if (!pack) return null;

  const add = async (event: string, file?: File) => {
    const url = urls[event]?.trim();
    if (!file && !url) return;
    setBusy(event);
    setErr(null);
    try {
      onChange(await api.addCue(game, event, file ? { file } : { url }));
      setUrls((u) => ({ ...u, [event]: "" }));
    } catch (e: any) {
      setErr(`Could not add the "${event}" cue: ${e?.message ?? "unknown error"}`);
    } finally {
      setBusy(null);
    }
  };

  const remove = async (event: string) => {
    setBusy(event);
    setErr(null);
    try {
      onChange(await api.removeCue(game, event));
    } catch (e: any) {
      setErr(`Could not remove the "${event}" cue: ${e?.message ?? "unknown error"}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel section" style={{ marginTop: 16 }}>
      <h3>
        Pinpoint cues — {pack.label}{" "}
        <span className="muted tiny">({pack.configured}/{pack.total} configured)</span>
      </h3>
      <p className="muted tiny" style={{ marginBottom: 12 }}>
        Optional: paste a sound URL (e.g. from MyInstants) or upload a file for each event to
        detect it exactly. Without cues, ClipForge still finds the loud moments automatically.
      </p>
      {err && (
        <p className="tiny" style={{ color: "var(--bad)", marginBottom: 10 }}>
          ⚠ {err}
        </p>
      )}
      <div className="caption-list">
        {pack.events.map((e) => (
          <div key={e.name} className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ width: 130, fontWeight: 600, color: e.configured ? "var(--good)" : undefined }}>
              {e.configured ? "✓" : "○"} {e.name}
            </span>
            <input
              className="input"
              style={{ flex: 1, minWidth: 170 }}
              placeholder={`${e.hint} — paste URL`}
              value={urls[e.name] || ""}
              onChange={(ev) => setUrls((u) => ({ ...u, [e.name]: ev.target.value }))}
            />
            <a
              className="btn sm ghost"
              href={`https://www.myinstants.com/en/search/?name=${encodeURIComponent(e.hint)}`}
              target="_blank"
              rel="noreferrer"
              title={`Search MyInstants for "${e.hint}" — right-click the sound's Download link, copy the address, paste it here`}
            >
              🔍 Find
            </a>
            <button className="btn sm" disabled={busy === e.name} onClick={() => add(e.name)}>
              {busy === e.name ? "…" : "Add"}
            </button>
            <label className="btn sm ghost" style={{ cursor: "pointer" }} title="Upload a sound file">
              File
              <input
                type="file"
                accept="audio/*"
                hidden
                onChange={(ev) => {
                  const f = ev.target.files?.[0];
                  // reset so picking the same file again re-fires the event
                  ev.target.value = "";
                  if (f) add(e.name, f);
                }}
              />
            </label>
            {e.configured && (
              <button className="btn sm danger" disabled={busy === e.name} onClick={() => remove(e.name)}>
                ✕
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
