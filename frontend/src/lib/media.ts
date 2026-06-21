export function mediaUrl(path?: string | null): string | undefined {
  if (!path) return undefined;
  const parts = path.split(/[\\/]+/).filter(Boolean).map(encodeURIComponent);
  return parts.length ? `/media/${parts.join("/")}` : undefined;
}

export function mediaTimeUrl(path: string | null | undefined, start: number, end: number): string | undefined {
  const base = mediaUrl(path);
  return base ? `${base}#t=${start.toFixed(3)},${end.toFixed(3)}` : undefined;
}
