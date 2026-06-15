import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CuesStatus } from "../lib/api";
import CueManager from "./CueManager";

/**
 * Modal for managing reference game-sound cues across ALL games — reachable
 * from the nav and from gameplay projects, so cue setup isn't gated behind
 * picking a game profile on the upload screen.
 */
export default function CueModal({ onClose }: { onClose: () => void }) {
  const [cues, setCues] = useState<CuesStatus | null>(null);
  const [game, setGame] = useState<string>("");

  useEffect(() => {
    api
      .cues()
      .then((c) => {
        setCues(c);
        setGame((g) => g || Object.keys(c)[0] || "");
      })
      .catch(() => {});
  }, []);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
          <h3>🎯 Game cues — pinpoint key moments</h3>
          <button className="btn ghost sm" onClick={onClose}>
            ✕ Close
          </button>
        </div>
        <p className="muted tiny" style={{ marginTop: 0 }}>
          Add a short reference sound per event (kill ding, goal horn, jump-scare sting…) and
          ClipForge finds every occurrence in your footage by audio matching — exact moments
          instead of loudness guesses. Paste a URL (e.g. MyInstants) or upload a file.
        </p>
        {cues ? (
          <>
            <div className="seg" style={{ marginBottom: 4, flexWrap: "wrap" }}>
              {Object.entries(cues).map(([id, pack]) => (
                <button
                  key={id}
                  className={game === id ? "on" : ""}
                  onClick={() => setGame(id)}
                  title={`${pack.configured}/${pack.total} cues configured`}
                >
                  {pack.label}
                  {pack.configured > 0 ? ` (${pack.configured})` : ""}
                </button>
              ))}
            </div>
            {game && <CueManager game={game} cues={cues} onChange={setCues} />}
            <p className="muted tiny" style={{ marginBottom: 0 }}>
              Added cues are saved on this machine and baked into <b>every</b> future run (on an
              existing project, hit ↻ Re-run to re-detect with them). ClipForge can't ship the
              game sounds themselves — they're copyrighted — so grab each one via 🔍 Find, an
              SFX pack, or the game files (see docs/GAME_CUES.md).
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
