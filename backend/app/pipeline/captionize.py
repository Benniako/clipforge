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


def _speaker_ok(w, speakers: set[int] | None) -> bool:
    """Whether word ``w`` belongs to a speaker the user kept in captions."""
    return speakers is None or (w.speaker or 0) in speakers


def build_tight_caption_set(transcript: Transcript,
                            segments: list[tuple[float, float]],
                            style_id: str,
                            speakers: set[int] | None = None) -> CaptionSet:
    """Caption set retimed onto the tightened timeline. ``speakers`` (a set of
    diarized speaker ids) keeps only those speakers' words; None keeps all."""
    words: list[CaptionWord] = []
    emitted: set[int] = set()  # word ids already placed (a word straddling two
    #                            segment edges must not be rendered twice)
    for a, b in segments:
        base = map_to_tight(a, segments)
        for i, w in enumerate(transcript.words):
            if w.end <= a or w.t >= b or i in emitted:
                continue
            if not _speaker_ok(w, speakers):
                continue
            text = _clean(w.text)
            if not text or not _ALNUM.search(text):
                continue
            emitted.add(i)
            rel = base + max(w.t - a, 0.0)
            d = max(min(w.end, b) - max(w.t, a), 0.04)
            words.append(CaptionWord(t=round(rel, 3), d=round(d, 3), text=text,
                                     speaker=w.speaker))
    return CaptionSet(words=words, style_id=style_id,
                      lang=(transcript.language or "en")[:2])


def build_caption_set(transcript: Transcript, start: float, end: float,
                      style_id: str,
                      exclude: list[tuple[float, float]] | None = None,
                      speakers: set[int] | None = None) -> CaptionSet:
    """``exclude`` lists absolute source-time spans whose words are dropped —
    used to keep in-game announcer lines (matched audio cues) out of the
    captions: ASR transcribes "Double Kill!" too, but the streamer didn't say it.
    ``speakers`` (diarized speaker ids) keeps only those speakers' words; None
    keeps all — the per-speaker caption toggle in the editor.
    """
    dur = max(end - start, 0.01)
    words: list[CaptionWord] = []
    for w in transcript.words:
        if w.end <= start or w.t >= end:
            continue
        if exclude and any(w.t < b and w.end > a for a, b in exclude):
            continue
        if not _speaker_ok(w, speakers):
            continue
        text = _clean(w.text)
        if not text or not _ALNUM.search(text):  # skip pure-punctuation tokens
            continue
        rel = max(w.t - start, 0.0)
        wend = min(w.end - start, dur)
        d = max(wend - rel, 0.04)
        words.append(CaptionWord(t=round(rel, 3), d=round(d, 3), text=text,
                                 speaker=w.speaker))
    return CaptionSet(words=words, style_id=style_id,
                      lang=(transcript.language or "en")[:2])


def speakers_in(transcript: Transcript, start: float, end: float) -> list[int]:
    """Sorted distinct speaker ids that actually speak inside [start, end].

    Drives the editor's per-speaker caption toggles — we only offer to mute a
    speaker who is present in the clip."""
    seen = {(w.speaker or 0) for w in transcript.words
            if w.end > start and w.t < end and _ALNUM.search(w.text or "")}
    return sorted(seen)


# --------------------------------------------------------------------------- #
# In-game voice suppression (gameplay captions)
# --------------------------------------------------------------------------- #
# Stock announcer/agent lines per game profile — ASR picks these up from the
# game audio, but they aren't the streamer talking and read as noise burned
# into captions. Phrases are matched case-insensitively as whole-word n-grams.
_GAME_NOISE: dict[str, frozenset[str]] = {
    "valorant": frozenset({
        "double kill", "triple kill", "quadra kill", "ace", "team ace",
        "clutch", "last player standing", "spike planted", "spike defused",
        "flawless", "match point", "lost lead", "taken lead",
    }),
    "cs2": frozenset({
        "double kill", "triple kill", "headshot", "the bomb has been planted",
        "the bomb has been defused", "counter terrorists win", "terrorists win",
        "bomb has been planted", "bomb has been defused",
    }),
    "rocketleague": frozenset({"what a save", "nice shot", "great pass", "calculated"}),
    # EA FC commentary is continuous prose — a phrase list can't separate it
    # from the streamer; rely on the cue-window exclusion there.
    "eafc": frozenset(),
    "horror": frozenset(),
}
_NOISE_ALIAS = {"auto": "generic", "cs": "cs2", "fifa": "eafc"}
_TOKEN_RE = re.compile(r"[^a-zà-ÿ0-9]")


def game_noise(profile: str | None) -> frozenset[str]:
    name = (profile or "").lower().replace(" ", "")
    return _GAME_NOISE.get(_NOISE_ALIAS.get(name, name), frozenset())


def remove_phrases(words: list[CaptionWord],
                   phrases: frozenset[str]) -> list[CaptionWord]:
    """Drop caption words that form any of the given phrases (n-gram match)."""
    if not words or not phrases:
        return words
    toks = [_TOKEN_RE.sub("", w.text.lower()) for w in words]
    drop = [False] * len(words)
    for phrase in phrases:
        ptoks = phrase.split()
        n = len(ptoks)
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == ptoks:
                for j in range(i, i + n):
                    drop[j] = True
    return [w for w, d in zip(words, drop) if not d]
