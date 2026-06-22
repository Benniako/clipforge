import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { CapabilityDetail } from "../lib/types";
import { useT } from "../lib/i18n";

const CATEGORY_LABELS: Record<string, string> = {};
const CATEGORY_ORDER = ["core", "transcription", "vision", "ocr", "audio", "gpu", "scenework"];

/**
 * Full-screen diagnostics modal: a grouped inventory of every optional
 * dependency ClipForge detected (or didn't), with an impact line per item so
 * the user knows exactly what's available and what each missing piece would
 * unlock. Opened from the nav's compact capability strip.
 */
export default function DiagnosticsPanel({ onClose }: { onClose: () => void }) {
  const { t } = useT();
  const [detail, setDetail] = useState<CapabilityDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    setErr(null);
    api
      .capabilities()
      .then((r) => setDetail(r.detail))
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const cats = (detail?.categories ?? []).slice().sort(
    (a, b) => CATEGORY_ORDER.indexOf(a.name) - CATEGORY_ORDER.indexOf(b.name),
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal diagnostics-panel"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 760 }}
      >
        <div className="diag-head row" style={{ justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <h3 style={{ margin: 0 }}>{t("diag.title")}</h3>
            <p className="muted tiny" style={{ margin: "4px 0 0" }}>{t("diag.subtitle")}</p>
          </div>
          <div className="row">
            <button className="btn ghost sm" onClick={load} disabled={loading}>
              {loading ? "…" : t("diag.refresh")}
            </button>
            <button className="btn ghost sm" onClick={onClose}>
              {t("diag.close")}
            </button>
          </div>
        </div>

        {err && (
          <p className="tiny bad">
            {err} — <a onClick={load}>{t("diag.refresh")}</a>
          </p>
        )}
        {loading && !detail && (
          <div className="empty">
            <span className="spinner" />
          </div>
        )}
        {cats.map((cat) => {
          const avail = cat.items.filter((i) => i.available).length;
          const labelKey = `diag.cat.${cat.name}`;
          const translated = t(labelKey);
          const label = translated === labelKey ? cat.name : translated;
          return (
            <div key={cat.name} className="diag-category">
              <div className="diag-cat-head row" style={{ justifyContent: "space-between" }}>
                <span className="diag-cat-name">{label}</span>
                <span className={"pill " + (avail === cat.items.length ? "ok" : avail === 0 ? "bad" : "")}>
                  {avail}/{cat.items.length}
                </span>
              </div>
              <div className="diag-items">
                {cat.items.map((it) => (
                  <div
                    key={it.key}
                    className={"diag-item " + (it.available ? "on" : "off")}
                    title={it.impact}
                  >
                    <span className={"cap-dot" + (it.available ? "" : " off")} />
                    <div className="diag-item-body">
                      <span className="diag-item-label">
                        {it.label}
                        <span className={"diag-state " + (it.available ? "ok" : "muted")}>
                          {it.available ? t("diag.available") : t("diag.missing")}
                        </span>
                      </span>
                      <span className="diag-impact muted tiny">{it.impact}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
