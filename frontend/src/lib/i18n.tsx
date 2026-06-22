// Minimal i18n layer. Strings live in flat de/en dictionaries keyed by a dotted
// name; t(key) resolves against the active language, falling back to English and
// then the key itself so a missing translation is visible but never crashes.
// Language is persisted in localStorage; default is German to preserve the
// app's original behaviour.
import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Lang = "de" | "en";

const STRINGS: Record<Lang, Record<string, string>> = {
  de: {
    "lang.name": "Deutsch",
    "lang.toggle": "EN",
    "notice.info": "Info",
    "notice.warn": "Hinweis",
    "notice.error": "Fehler",
    "nav.cues": "Spiel-Cues",
    "nav.cuesTitle":
      "Referenzsounds oder OCR-Begriffe für Spielereignisse hinzufügen und testen",
    "nav.newProject": "+ Neues Projekt",
    "gc.section": "Erkennungs-Tuning",
    "gc.sectionHint": "(optional, nur Gameplay)",
    "gc.detectionMode": "Erkennungsmodus",
    "gc.modeAuto": "Automatisch (Zero-Shot)",
    "gc.modeHybrid": "Hybrid (meine Cues + strenge Auto-Hits)",
    "gc.modeManual": "Nur meine Cues (kein Zero-Shot)",
    "gc.audioCues": "Audio-Cues",
    "gc.audioCuesHint": "Was nach Action klingt, z. B. „ace celebration, crowd hype“",
    "gc.visualCues": "Bildschirm-Text-Cues",
    "gc.visualCuesHint": "Banner-Text für OCR, z. B. „VICTORY, ELIMINATED“",
    "gc.vlmCues": "KI-Bild-Hinweise",
    "gc.vlmCuesHint": "Worauf die KI-Bildanalyse achten soll, z. B. „kill feed, victory screen“",
    "gc.placeholder": "Mit Komma oder Zeile trennen",
    "swipe.grid": "Grid",
    "swipe.premiereEdl": "Premiere EDL",
    "swipe.toGrid": "Zur Grid-Ansicht",
    "swipe.emptyTitle": "Noch keine gerenderten Clips",
    "swipe.untitled": "Unbenannter Clip",
    "swipe.back": "Zurück",
    "swipe.bad": "Schlecht",
    "swipe.good": "Gut",
    "swipe.next": "Weiter",
    "swipe.edit": "Bearbeiten",
    "swipe.download": "Download",
    "cues.title": "Spiel-Cues",
    "cues.close": "Schließen",
    "cues.intro":
      "Teste hier Sounds und Texterkennung im Bild und speichere nützliche Cues für spätere Renderings. Visuelle Cues sind OCR-Begriffe. Audio-Cues sind saubere Referenzsounds.",
    "cm.heading": "Eigene Sound-Cues - {label}",
    "cm.configured": "({done}/{total} konfiguriert)",
    "cm.toggleNote":
      "Das steuert nur, ob diese gespeicherten Sounds für die Clip-Erkennung genutzt werden.",
    "cm.useToggleTitle":
      "Eigene Referenzsounds für die nächste Erkennung ein- oder ausschalten",
    "cm.useInClips": "In Clips nutzen",
    "cm.on": "An",
    "cm.off": "Aus",
    "cm.intro":
      "Optional: Füge eine Sound-URL ein oder lade einen sauberen Referenzsound hoch. ClipForge nutzt diese Sounds als zusätzliches Signal, nicht als garantiertes Highlight.",
    "cm.introSave":
      " Eingegebene URLs werden erst gespeichert, wenn du Hinzufügen oder Alle speichern klickst.",
    "cm.offNote":
      "Eigene Sound-Erkennung ist für neue Clips aus. Du kannst hier trotzdem Sounds verwalten.",
    "cm.active": "aktiv",
    "cm.new": "neu",
    "cm.urlPlaceholder": "{label} - URL einfügen",
    "cm.findTitle": "MyInstants nach {label} durchsuchen",
    "cm.find": "Finden",
    "cm.add": "Hinzufügen",
    "cm.fileTitle": "Sounddatei hochladen",
    "cm.file": "Datei",
    "cm.remove": "Entfernen",
    "cm.saveAllTitle": "Alle oben eingefügten URLs herunterladen und installieren",
    "cm.saving": "Speichert...",
    "cm.saveAll": "Alle speichern ({count})",
    "cm.addFailed":
      "Cue \"{event}\" konnte nicht hinzugefügt werden: {error}",
    "cm.removeFailed":
      "Cue \"{event}\" konnte nicht entfernt werden: {error}",
    "cm.someFailed": "Einige Cues sind fehlgeschlagen: {errors}",
    "cm.unknownError": "unbekannter Fehler",
    "cm.failed": "fehlgeschlagen",
  },
  en: {
    "lang.name": "English",
    "lang.toggle": "DE",
    "notice.info": "Info",
    "notice.warn": "Note",
    "notice.error": "Error",
    "nav.cues": "Game cues",
    "nav.cuesTitle": "Add and test reference sounds or OCR terms for game events",
    "nav.newProject": "+ New project",
    "gc.section": "Detection tuning",
    "gc.sectionHint": "(optional, gameplay only)",
    "gc.detectionMode": "Detection mode",
    "gc.modeAuto": "Automatic (zero-shot)",
    "gc.modeHybrid": "Hybrid (my cues + strict auto-hits)",
    "gc.modeManual": "My cues only (no zero-shot)",
    "gc.audioCues": "Audio cues",
    "gc.audioCuesHint": "What action sounds like, e.g. \"ace celebration, crowd hype\"",
    "gc.visualCues": "On-screen text cues",
    "gc.visualCuesHint": "Banner text for OCR, e.g. \"VICTORY, ELIMINATED\"",
    "gc.vlmCues": "AI vision hints",
    "gc.vlmCuesHint": "What the AI vision read should watch for, e.g. \"kill feed, victory screen\"",
    "gc.placeholder": "Separate with commas or new lines",
    "swipe.grid": "Grid",
    "swipe.premiereEdl": "Premiere EDL",
    "swipe.toGrid": "Back to grid",
    "swipe.emptyTitle": "No rendered clips yet",
    "swipe.untitled": "Untitled clip",
    "swipe.back": "Back",
    "swipe.bad": "Bad",
    "swipe.good": "Good",
    "swipe.next": "Next",
    "swipe.edit": "Edit",
    "swipe.download": "Download",
    "cues.title": "Game cues",
    "cues.close": "Close",
    "cues.intro":
      "Test sounds and on-screen text recognition here, and save useful cues for later renders. Visual cues are OCR terms. Audio cues are clean reference sounds.",
    "cm.heading": "Custom sound cues - {label}",
    "cm.configured": "({done}/{total} configured)",
    "cm.toggleNote":
      "This only controls whether these saved sounds are used for clip detection.",
    "cm.useToggleTitle":
      "Turn your own reference sounds on or off for the next detection run",
    "cm.useInClips": "Use in clips",
    "cm.on": "On",
    "cm.off": "Off",
    "cm.intro":
      "Optional: paste a sound URL or upload a clean reference sound. ClipForge uses these sounds as an extra signal, not as a guaranteed highlight.",
    "cm.introSave":
      " URLs you enter are only saved once you click Add or Save all.",
    "cm.offNote":
      "Custom sound detection is off for new clips. You can still manage sounds here.",
    "cm.active": "active",
    "cm.new": "new",
    "cm.urlPlaceholder": "{label} - paste URL",
    "cm.findTitle": "Search MyInstants for {label}",
    "cm.find": "Find",
    "cm.add": "Add",
    "cm.fileTitle": "Upload a sound file",
    "cm.file": "File",
    "cm.remove": "Remove",
    "cm.saveAllTitle": "Download and install every URL pasted above",
    "cm.saving": "Saving...",
    "cm.saveAll": "Save all ({count})",
    "cm.addFailed": "Couldn't add cue \"{event}\": {error}",
    "cm.removeFailed": "Couldn't remove cue \"{event}\": {error}",
    "cm.someFailed": "Some cues failed: {errors}",
    "cm.unknownError": "unknown error",
    "cm.failed": "failed",
  },
};

const STORAGE_KEY = "clipforge.lang";

function initialLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "de" || v === "en") return v;
  } catch {
    /* ignore */
  }
  return "de";
}

interface I18n {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18n | null>(null);

function format(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (m, k) =>
    k in vars ? String(vars[k]) : m,
  );
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      /* ignore */
    }
  }, []);
  const t = useCallback(
    (key: string, vars?: Record<string, string | number>) =>
      format(STRINGS[lang][key] ?? STRINGS.en[key] ?? key, vars),
    [lang],
  );
  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useT(): I18n {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    // Outside a provider (e.g. an isolated test) — fall back to German strings.
    return {
      lang: "de",
      setLang: () => {},
      t: (key: string, vars?: Record<string, string | number>) =>
        format(STRINGS.de[key] ?? STRINGS.en[key] ?? key, vars),
    };
  }
  return ctx;
}

export function LanguageToggle({ className }: { className?: string }) {
  const { lang, setLang, t } = useT();
  return (
    <button
      className={className ?? "btn ghost sm"}
      title={t("lang.name")}
      onClick={() => setLang(lang === "de" ? "en" : "de")}
    >
      {t("lang.toggle")}
    </button>
  );
}
