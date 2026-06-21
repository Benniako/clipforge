"""Moment detection — find self-contained segments worth clipping.

Strategy (transcript-driven, fully local):

1.  Segment the transcript into sentences using punctuation and speech pauses,
    so candidate boundaries land on natural speech edges (PRD: never cut a word
    or a punchline).
2.  Grow candidate clips from each sentence, accreting following sentences while
    the total stays inside the user's target length range. This yields several
    candidate lengths per starting point.
3.  Rank candidates by a provisional salience (a blend of the explainable
    signals) and apply non-maximum suppression so the returned set doesn't
    splice the same moment ten times.
4.  Synthesise a title/hook from the candidate's own words.

This is a heuristic stand-in for a tuned model; it lives behind a stable
function so a learned detector can replace it without touching the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import ImportSettings, Word
from . import signals


@dataclass
class Sentence:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].t

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Candidate:
    start: float
    end: float
    words: list[Word]
    salience: float
    title: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _segment_sentences(words: list[Word], *, pause: float = 0.65) -> list[Sentence]:
    """Split words into sentences on terminal punctuation or a long pause."""
    sentences: list[Sentence] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        ends_punct = w.text.strip().endswith((".", "!", "?"))
        gap_next = (words[i + 1].t - w.end) if i + 1 < len(words) else 0.0
        if ends_punct or gap_next >= pause:
            sentences.append(Sentence(cur))
            cur = []
    if cur:
        sentences.append(Sentence(cur))
    return sentences


def _salience(words: list[Word], duration: float, st: ImportSettings,
              lex: signals.Lexicon, weights: dict[str, float] | None = None) -> float:
    vals = {
        "instant_hook": signals.instant_hook(words, lex)[0],
        "swipe": signals.swipe_resistance(words, duration, lex)[0],
        "hook": signals.hook_strength(words, lex)[0],
        "emotion": signals.emotional_payoff(words, lex)[0],
        "clarity": signals.standalone_clarity(words, lex)[0],
        "quote": signals.quotability(words, lex)[0],
        "length": signals.length_fit(duration, st.min_len, st.max_len)[0],
        "pace": signals.pace_energy(words, duration)[0],
        "list": signals.list_payoff(words, lex)[0],
    }
    if weights:  # personalised ranking — same weights as scoring
        return sum(v * weights.get(k, 0.0) for k, v in vals.items())
    return (0.20 * vals["instant_hook"] + 0.14 * vals["swipe"]
            + 0.18 * vals["hook"] + 0.14 * vals["emotion"]
            + 0.16 * vals["clarity"] + 0.10 * vals["quote"]
            + 0.04 * vals["length"] + 0.04 * vals["pace"])


def _make_title(words: list[Word]) -> str:
    text = " ".join(w.text for w in words).strip()
    # Prefer the first sentence-ish chunk, capped for a hook line.
    import re

    first = re.split(r"(?<=[.!?])\s", text)[0]
    title = first if 12 <= len(first) <= 70 else text
    title = re.sub(r"\s+", " ", title).strip(" ,.-")
    words_out = title.split()
    if len(words_out) > 11:
        title = " ".join(words_out[:11]) + "…"
    return (title[:1].upper() + title[1:]) if title else "Untitled clip"


def _overlap(a: Candidate, b: Candidate) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    inter = max(0.0, hi - lo)
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def detect_moments(words: list[Word], settings: ImportSettings,
                   *, lang: str = "en", weights: dict[str, float] | None = None,
                   max_overlap: float = 0.35) -> list[Candidate]:
    if not words:
        return []
    lex = signals.get_lexicon(lang)
    sentences = _segment_sentences(words)
    cands: list[Candidate] = []

    for i in range(len(sentences)):
        acc: list[Word] = []
        for j in range(i, len(sentences)):
            acc = acc + sentences[j].words
            dur = acc[-1].end - acc[0].t
            if dur < settings.min_len:
                continue
            if dur > settings.max_len:
                break
            sal = _salience(acc, dur, settings, lex, weights)
            cands.append(Candidate(
                start=acc[0].t, end=acc[-1].end, words=list(acc),
                salience=sal, title=_make_title(acc),
            ))

    # If sentences are so long nothing fits the window, fall back to fixed
    # windows snapped to word edges so we still return something usable.
    if not cands:
        cands = _window_fallback(words, settings, lex, weights)

    cands.sort(key=lambda c: c.salience, reverse=True)

    # Greedy non-maximum suppression on temporal overlap.
    kept: list[Candidate] = []
    for c in cands:
        if all(_overlap(c, k) <= max_overlap for k in kept):
            kept.append(c)
        if len(kept) >= settings.target_clips:
            break
    kept.sort(key=lambda c: c.start)
    return kept


def _window_fallback(words: list[Word], st: ImportSettings,
                     lex: signals.Lexicon, weights: dict[str, float] | None = None) -> list[Candidate]:
    target = (st.min_len + st.max_len) / 2.0
    out: list[Candidate] = []
    i = 0
    n = len(words)
    while i < n:
        start_t = words[i].t
        j = i
        while j < n and words[j].end - start_t < target:
            j += 1
        span = words[i:j] or words[i:i + 1]
        dur = span[-1].end - span[0].t
        out.append(Candidate(
            start=span[0].t, end=span[-1].end, words=list(span),
            salience=_salience(span, dur, st, lex, weights), title=_make_title(span),
        ))
        # j == i when a single word already spans the target (bad ASR timestamp);
        # always advance or the loop never terminates.
        i = max(j, i + 1)
    return out
