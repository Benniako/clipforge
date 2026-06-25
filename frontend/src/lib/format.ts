export function fmtDuration(s: number): string {
  if (!s || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// Compact elapsed/remaining clock: "H:MM:SS" past an hour, else "M:SS".
export function fmtHMS(s: number): string {
  if (!s || s < 0) s = 0;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function fmtClock(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const cs = Math.floor((s % 1) * 10);
  return `${m}:${sec.toString().padStart(2, "0")}.${cs}`;
}

// Green (great) -> yellow (ok) -> red (weak).
export function scoreColor(score: number): string {
  if (score >= 75) return "var(--good)";
  if (score >= 55) return "#b6e36b";
  if (score >= 40) return "var(--warn)";
  return "var(--bad)";
}

export function timeAgo(
  epochSeconds: number,
  t?: (key: string, vars?: Record<string, string | number>) => string,
): string {
  const diff = Date.now() / 1000 - epochSeconds;
  if (t) {
    if (diff < 60) return t("timeAgo.justNow");
    if (diff < 3600) return t("timeAgo.minutes", { n: Math.floor(diff / 60) });
    if (diff < 86400) return t("timeAgo.hours", { n: Math.floor(diff / 3600) });
    return t("timeAgo.days", { n: Math.floor(diff / 86400) });
  }
  // Legacy fallback (German) when no t function is provided.
  if (diff < 60) return "gerade eben";
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`;
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`;
  return `vor ${Math.floor(diff / 86400)} Tg`;
}
}

