import { useState } from "react";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";

const EVENT_LABELS: Record<string, string> = {
  kill: "Kill",
  double_kill: "Doppel-Kill",
  triple_kill: "Triple-Kill",
  quad_kill: "Vierfach-Kill",
  ace: "Ace",
  clutch: "Clutch",
  spike_plant: "Spike platziert",
  spike_defuse: "Spike entschärft",
  headshot: "Headshot",
  bomb_plant: "Bombe gelegt",
  bomb_defuse: "Bombe entschärft",
  goal: "Tor",
  whistle: "Pfiff",
  crowd_roar: "Jubel",
  demolition: "Demolition",
  save: "Parade",
  stinger: "Stinger",
  scream: "Schrei",
  jumpscare: "Jumpscare",
  airhorn: "Airhorn",
  hype: "Hype",
  laugh: "Lacher",
  applause: "Applaus",
  bruh: "Bruh",
  wow: "Wow",
};

export default function CueManager({
  game,
  cues,
  onChange,
  enabled,
  onToggle,
}: {
  game: string;
  cues: CuesStatus | null;
  onChange: (c: CuesStatus) => void;
  enabled?: boolean;
  onToggle?: (enabled: boolean) => void;
}) {
  const pack = cues?.[game];
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [busyAll, setBusyAll] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!pack) return null;

  const filled = pack.events.filter((e) => (urls[e.name] ?? "").trim());

  const add = async (event: string, file?: File) => {
    const url = urls[event]?.trim();
    if (!file && !url) return;
    setBusy(event);
    setErr(null);
    try {
      onChange(await api.addCue(game, event, file ? { file } : { url }));
      setUrls((u) => ({ ...u, [event]: "" }));
    } catch (e: any) {
      setErr(`Cue "${event}" konnte nicht hinzugefügt werden: ${e?.message ?? "unbekannter Fehler"}`);
    } finally {
      setBusy(null);
    }
  };

  const saveAll = async () => {
    setBusyAll(true);
    setErr(null);
    const errors: string[] = [];
    for (const ev of pack.events) {
      const url = (urls[ev.name] ?? "").trim();
      if (!url) continue;
      setBusy(ev.name);
      try {
        onChange(await api.addCue(game, ev.name, { url }));
        setUrls((u) => ({ ...u, [ev.name]: "" }));
      } catch (e: any) {
        errors.push(`${ev.name}: ${e?.message ?? "fehlgeschlagen"}`);
      }
    }
    setBusy(null);
    setBusyAll(false);
    if (errors.length) setErr(`Einige Cues sind fehlgeschlagen: ${errors.join(" | ")}`);
  };

  const remove = async (event: string) => {
    setBusy(event);
    setErr(null);
    try {
      onChange(await api.removeCue(game, event));
    } catch (e: any) {
      setErr(`Cue "${event}" konnte nicht entfernt werden: ${e?.message ?? "unbekannter Fehler"}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel section" style={{ marginTop: 16 }}>
      <div className="cue-manager-head">
        <div>
          <h3>
            Eigene Sound-Cues - {pack.label}{" "}
            <span className="muted tiny">({pack.configured}/{pack.total} konfiguriert)</span>
          </h3>
          {onToggle && (
            <p className="muted tiny" style={{ margin: "6px 0 0" }}>
              Das steuert nur, ob diese gespeicherten Sounds für die Clip-Erkennung genutzt werden.
            </p>
          )}
        </div>
        {onToggle && (
          <button
            className={"toggle cue-use-toggle" + (enabled ? " on" : "")}
            onClick={() => onToggle(!enabled)}
            title="Eigene Referenzsounds für die nächste Erkennung ein- oder ausschalten"
          >
            <span>In Clips nutzen</span>
            <i>{enabled ? "An" : "Aus"}</i>
          </button>
        )}
      </div>
      <p className="muted tiny" style={{ marginBottom: 12 }}>
        Optional: Füge eine Sound-URL ein oder lade einen sauberen Referenzsound hoch.
        ClipForge nutzt diese Sounds als zusätzliches Signal, nicht als garantiertes Highlight.
        <b> Eingegebene URLs werden erst gespeichert, wenn du Hinzufügen oder Alle speichern klickst.</b>
      </p>
      {onToggle && !enabled && (
        <p className="tiny cue-off-note">
          Eigene Sound-Erkennung ist für neue Clips aus. Du kannst hier trotzdem Sounds verwalten.
        </p>
      )}
      {err && (
        <p className="tiny" style={{ color: "var(--bad)", marginBottom: 10 }}>
          {err}
        </p>
      )}
      <div className="caption-list">
        {pack.events.map((e) => (
          <div key={e.name} className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ width: 130, fontWeight: 600, color: e.configured ? "var(--good)" : undefined }}>
                {e.configured ? "aktiv" : "neu"} {EVENT_LABELS[e.name] ?? e.name}
              </span>
              <input
                className="input"
                style={{ flex: 1, minWidth: 170 }}
                placeholder={`${EVENT_LABELS[e.name] ?? e.name} - URL einfügen`}
                value={urls[e.name] || ""}
                onChange={(ev) => setUrls((u) => ({ ...u, [e.name]: ev.target.value }))}
              />
              <a
                className="btn sm ghost"
                href={`https://www.myinstants.com/de/search/?name=${encodeURIComponent(e.hint)}`}
                target="_blank"
                rel="noreferrer"
                title={`MyInstants nach ${EVENT_LABELS[e.name] ?? e.name} durchsuchen`}
              >
                Finden
              </a>
            <button className="btn sm" disabled={busy === e.name} onClick={() => add(e.name)}>
              {busy === e.name ? "..." : "Hinzufügen"}
            </button>
            <label className="btn sm ghost" style={{ cursor: "pointer" }} title="Sounddatei hochladen">
              Datei
              <input
                type="file"
                accept="audio/*"
                hidden
                onChange={(ev) => {
                  const f = ev.target.files?.[0];
                  ev.target.value = "";
                  if (f) add(e.name, f);
                }}
              />
            </label>
            {e.configured && (
              <button className="btn sm danger" disabled={busy === e.name} onClick={() => remove(e.name)}>
                Entfernen
              </button>
            )}
          </div>
        ))}
      </div>
      {filled.length > 0 && (
        <div className="row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
          <button
            className="btn primary sm"
            disabled={busyAll}
            onClick={saveAll}
            title="Alle oben eingefügten URLs herunterladen und installieren"
          >
            {busyAll ? (
              <>
                <span className="spinner" /> Speichert...
              </>
            ) : (
              <>Alle speichern ({filled.length})</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
