import { useState } from "react";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import { useT } from "../lib/i18n";

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

const EVENT_LABELS_EN: Record<string, string> = {
  kill: "Kill",
  double_kill: "Double kill",
  triple_kill: "Triple kill",
  quad_kill: "Quad kill",
  ace: "Ace",
  clutch: "Clutch",
  spike_plant: "Spike planted",
  spike_defuse: "Spike defused",
  headshot: "Headshot",
  bomb_plant: "Bomb planted",
  bomb_defuse: "Bomb defused",
  goal: "Goal",
  whistle: "Whistle",
  crowd_roar: "Crowd roar",
  demolition: "Demolition",
  save: "Save",
  stinger: "Stinger",
  scream: "Scream",
  jumpscare: "Jumpscare",
  airhorn: "Airhorn",
  hype: "Hype",
  laugh: "Laugh",
  applause: "Applause",
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
  const { t, lang } = useT();
  const labels = lang === "en" ? EVENT_LABELS_EN : EVENT_LABELS;
  const eventLabel = (name: string) => labels[name] ?? name;
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
      setErr(t("cm.addFailed", { event, error: e?.message ?? t("cm.unknownError") }));
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
        errors.push(`${ev.name}: ${e?.message ?? t("cm.failed")}`);
      }
    }
    setBusy(null);
    setBusyAll(false);
    if (errors.length) setErr(t("cm.someFailed", { errors: errors.join(" | ") }));
  };

  const remove = async (event: string) => {
    setBusy(event);
    setErr(null);
    try {
      onChange(await api.removeCue(game, event));
    } catch (e: any) {
      setErr(t("cm.removeFailed", { event, error: e?.message ?? t("cm.unknownError") }));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="panel section" style={{ marginTop: 16 }}>
      <div className="cue-manager-head">
        <div>
          <h3>
            {t("cm.heading", { label: pack.label })}{" "}
            <span className="muted tiny">
              {t("cm.configured", { done: pack.configured, total: pack.total })}
            </span>
          </h3>
          {onToggle && (
            <p className="muted tiny" style={{ margin: "6px 0 0" }}>
              {t("cm.toggleNote")}
            </p>
          )}
        </div>
        {onToggle && (
          <button
            className={"toggle cue-use-toggle" + (enabled ? " on" : "")}
            onClick={() => onToggle(!enabled)}
            title={t("cm.useToggleTitle")}
          >
            <span>{t("cm.useInClips")}</span>
            <i>{enabled ? t("cm.on") : t("cm.off")}</i>
          </button>
        )}
      </div>
      <p className="muted tiny" style={{ marginBottom: 12 }}>
        {t("cm.intro")}
        <b>{t("cm.introSave")}</b>
      </p>
      {onToggle && !enabled && (
        <p className="tiny cue-off-note">{t("cm.offNote")}</p>
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
                {e.configured ? t("cm.active") : t("cm.new")} {eventLabel(e.name)}
              </span>
              <input
                className="input"
                style={{ flex: 1, minWidth: 170 }}
                placeholder={t("cm.urlPlaceholder", { label: eventLabel(e.name) })}
                value={urls[e.name] || ""}
                onChange={(ev) => setUrls((u) => ({ ...u, [e.name]: ev.target.value }))}
              />
              <a
                className="btn sm ghost"
                href={`https://www.myinstants.com/${lang}/search/?name=${encodeURIComponent(e.hint)}`}
                target="_blank"
                rel="noreferrer"
                title={t("cm.findTitle", { label: eventLabel(e.name) })}
              >
                {t("cm.find")}
              </a>
            <button className="btn sm" disabled={busy === e.name} onClick={() => add(e.name)}>
              {busy === e.name ? "..." : t("cm.add")}
            </button>
            <label className="btn sm ghost" style={{ cursor: "pointer" }} title={t("cm.fileTitle")}>
              {t("cm.file")}
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
                {t("cm.remove")}
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
            title={t("cm.saveAllTitle")}
          >
            {busyAll ? (
              <>
                <span className="spinner" /> {t("cm.saving")}
              </>
            ) : (
              <>{t("cm.saveAll", { count: filled.length })}</>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
