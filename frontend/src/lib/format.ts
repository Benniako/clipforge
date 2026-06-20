export function fmtDuration(s: number): string {
  if (!s || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function fmtClock(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const cs = Math.floor((s % 1) * 10);
  return `${m}:${sec.toString().padStart(2, "0")}.${cs}`;
}

// Green (great) â†’ yellow (ok) â†’ red (weak).
export function scoreColor(score: number): string {
  if (score >= 75) return "var(--good)";
  if (score >= 55) return "#b6e36b";
  if (score >= 40) return "var(--warn)";
  return "var(--bad)";
}

export function timeAgo(epochSeconds: number): string {
  const diff = Date.now() / 1000 - epochSeconds;
  if (diff < 60) return "gerade eben";
  if (diff < 3600) return `vor ${Math.floor(diff / 60)} Min`;
  if (diff < 86400) return `vor ${Math.floor(diff / 3600)} Std`;
  return `vor ${Math.floor(diff / 86400)} Tg`;
}

