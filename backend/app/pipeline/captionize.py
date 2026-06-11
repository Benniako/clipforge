"""Build a clip-relative CaptionSet from the source transcript.

Captions for a clip are just the transcript words that fall inside the clip's
span, retimed so t=0 is the clip's start. We include any word that overlaps the
span (not only fully-contained ones) so the first/last word isn't dropped, and
clamp timings into the clip's duration.
"""
from __future__ import annotations

import re

from ..models import CaptionSet, CaptionWord, Transcript

# Trim leading punctuation that Whisper attaches to tokens (", me" -> "me") and
# drop tokens that are punctuation only — both read as glitches on screen.
_LEAD_PUNCT = re.compile(r"^[\s,;:.\-–—]+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


def _clean(text: str) -> str:
    return _LEAD_PUNCT.sub("", text.strip()).strip()


# ---- silence tightening (jump cuts) ---------------------------------------- #
MAX_GAP = 0.7      # a pause longer than this gets cut out
EDGE_PAD = 0.12    # keep a little air around speech so cuts don't clip words
MIN_SAVING = 0.5   # only tighten when it actually removes this many seconds


def compute_tight_segments(transcript: Transcript, start: float,
                           end: float) -> list[tuple[float, float]]:
    """Speech segments (absolute source times) inside [start, end] with dead air
    removed. Returns [(start, end)] — a single full-span segment means
    "nothing worth cutting"."""
    words = [w for w in transcript.words if w.end > start and w.t < end]
    if not words:
        return [(start, end)]
    segs: list[list[float]] = []
    for w in words:
        ws, we = max(w.t, start), min(w.end, end)
        if segs and ws - segs[-1][1] <= MAX_GAP:
            segs[-1][1] = max(segs[-1][1], we)
        else:
            segs.append([ws, we])
    padded = [(max(start, a - EDGE_PAD), min(end, b + EDGE_PAD)) for a, b in segs]
    merged: list[tuple[float, float]] = []
    for a, b in padded:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    saved = (end - start) - sum(b - a for a, b in merged)
    if len(merged) < 2 or saved < MIN_SAVING:
        return [(start, end)]
    return merged


def map_to_tight(t_abs: float, segments: list[tuple[float, float]]) -> float:
    """Map an absolute source time onto the tightened (concatenated) timeline."""
    cum = 0.0
    for a, b in segments:
        if t_abs < a:
            return cum
        if t_abs <= b:
            return cum + (t_abs - a)
        cum += b - a
    return cum


def build_tight_caption_set(transcript: Transcript,
                            segments: list[tuple[float, float]],
                            style_id: str) -> CaptionSet:
    """Caption set retimed onto the tightened timeline."""
    words: list[CaptionWord] = []
    for a, b in segments:
        base = map_to_tight(a, segments)
        for w in transcript.words:
            if w.end <= a or w.t >= b:
                continue
            text = _clean(w.text)
            if not text or not _ALNUM.search(text):
                continue
            rel = base + max(w.t - a, 0.0)
            d = max(min(w.end, b) - max(w.t, a), 0.04)
            words.append(CaptionWord(t=round(rel, 3), d=round(d, 3), text=text))
    return CaptionSet(words=words, style_id=style_id)


def build_caption_set(transcript: Transcript, start: float, end: float,
                      style_id: str) -> CaptionSet:
    dur = max(end - start, 0.01)
    words: list[CaptionWord] = []
    for w in transcript.words:
        if w.end <= start or w.t >= end:
            continue
        text = _clean(w.text)
        if not text or not _ALNUM.search(text):  # skip pure-punctuation tokens
            continue
        rel = max(w.t - start, 0.0)
        wend = min(w.end - start, dur)
        d = max(wend - rel, 0.04)
        words.append(CaptionWord(t=round(rel, 3), d=round(d, 3), text=text))
    return CaptionSet(words=words, style_id=style_id)
