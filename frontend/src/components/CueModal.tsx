import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import { useT, LanguageToggle } from "../lib/i18n";
import CueLab from "./CueLab";
import CueManager from "./CueManager";

export default function CueModal({ onClose }: { onClose: () => void }) {
  const { t } = useT();
  const [cues, setCues] = useState<CuesStatus | null>(null);
  const [game, setGame] = useState<string>("");

  const refreshCues = () => {
    api
      .cues()
      .then((c) => {
        setCues(c);
        setGame((g) =>
          g && g !== "common" ? g : Object.keys(c).find((id) => id !== "common") || "");
      })
      .catch(() => {});
  };

  useEffect(() => {
    refreshCues();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal cue-modal" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
          <h3>{t("cues.title")}</h3>
          <div className="row" style={{ gap: 8 }}>
            <LanguageToggle />
            <button className="btn ghost sm" onClick={onClose}>
              {t("cues.close")}
            </button>
          </div>
        </div>
        <p className="muted tiny" style={{ marginTop: 0 }}>
          {t("cues.intro")}
        </p>
        {cues ? (
          <>
            <div className="seg" style={{ marginBottom: 4, flexWrap: "wrap" }}>
              {Object.entries(cues).filter(([id]) => id !== "common").map(([id, pack]) => (
                <button
                  key={id}
                  className={game === id ? "on" : ""}
                  onClick={() => setGame(id)}
                  title={`${pack.configured}/${pack.total} Audio-Cues konfiguriert`}
                >
                  {pack.label}
                  {pack.configured > 0 ? ` (${pack.configured})` : ""}
                </button>
              ))}
            </div>
            {game && (
              <>
                <CueLab
                  game={game}
                  visual={cues[game]?.visual}
                  onVisualChange={(visual) =>
                    setCues((prev) => {
                      if (!prev) return prev;
                      const next = { ...prev };
                      const pack = next[game] ?? {
                        label: game,
                        configured: 0,
                        total: 0,
                        events: [],
                      };
                      next[game] = { ...pack, visual: visual[game] ?? {} };
                      return next;
                    })
                  }
                  onAudioChange={refreshCues}
                />
                <CueManager game={game} cues={cues} onChange={setCues} />
              </>
            )}
            <p className="muted tiny" style={{ marginBottom: 0 }}>
              Gespeicherte Cues bleiben auf diesem Rechner und werden in künftigen Läufen
              verwendet. Bei bestehenden Projekten die Erkennung neu starten, damit neue Cues
              berücksichtigt werden.
            </p>
          </>
        ) : (
          <div className="empty">
            <span className="spinner" />
          </div>
        )}
      </div>
    </div>
  );
}
