// Ordering for the swipe-review deck. Kept pure (no React) so the
// sort-stability contract can be unit-tested: equal scores must keep a
// deterministic order across project refreshes, otherwise a refresh after
// rating could reorder the deck under the fixed index and jump cards.
export interface ReviewClip {
  id: string;
  score: number;
  export_url?: string | null;
}

export function orderReviewClips<T extends ReviewClip>(clips: T[]): T[] {
  return clips
    .filter((c) => c.export_url)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
}
