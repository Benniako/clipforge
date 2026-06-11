"""Transcript signal extraction shared by detection and scoring.

These are deliberately transparent, lexicon-driven features — the PRD's hardest
requirement for the score is that it be *explainable*, not a black box. Each
feature returns a value in [0, 1] and a short human-readable reason, so the same
computation drives both candidate ranking and the user-facing score.

The lexicons are language-aware: pass the transcript's detected language and the
matching keyword sets are used, so moment detection and scoring work on German
source videos as well as English. Unknown languages fall back to English.
The reason strings stay in English (the UI is English).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import Word


@dataclass(frozen=True)
class Lexicon:
    hook: frozenset[str]
    emotion: frozenset[str]
    payoff: frozenset[str]
    dangling: frozenset[str]       # weak sentence openers
    second_person: frozenset[str]  # "you/your" equivalents
    quote_extra: frozenset[str]    # extra punchy/quotable words
    enumeration: frozenset[str]    # list/steps markers

    @property
    def quote(self) -> frozenset[str]:
        return self.hook | self.emotion | self.second_person | self.quote_extra


_EN = Lexicon(
    hook=frozenset({
        "how", "why", "what", "secret", "nobody", "everyone", "never", "always",
        "stop", "mistake", "truth", "reason", "actually", "surprising", "imagine",
        "warning", "honestly", "crazy", "wild", "ever", "biggest", "worst", "best",
    }),
    emotion=frozenset({
        "love", "hate", "fear", "amazing", "incredible", "terrible", "shocking",
        "insane", "beautiful", "painful", "hilarious", "scary", "exciting",
        "heartbreaking", "powerful", "frustrating", "grateful", "angry", "happy",
        "sad", "proud", "afraid", "excited", "wow", "unbelievable",
    }),
    payoff=frozenset({
        "because", "so", "therefore", "result", "realized", "learned", "lesson",
        "point", "means", "answer", "finally", "turns", "discovered", "secret",
        "key", "bottom", "ultimately", "conclusion",
    }),
    dangling=frozenset({
        "and", "but", "so", "or", "because", "which", "that", "it", "they", "he",
        "she", "this", "those", "these", "then", "also", "however",
    }),
    second_person=frozenset({"you", "your", "yourself"}),
    quote_extra=frozenset({
        "everything", "anything", "nothing", "life", "world", "matter", "moment",
        "change", "now",
    }),
    enumeration=frozenset({
        "first", "second", "third", "three", "two", "steps", "ways", "reasons", "tips",
    }),
)

_DE = Lexicon(
    hook=frozenset({
        "wie", "warum", "wieso", "weshalb", "was", "geheimnis", "niemand", "jeder",
        "nie", "niemals", "immer", "stopp", "fehler", "wahrheit", "grund",
        "eigentlich", "überraschend", "achtung", "ehrlich", "verrückt", "wahnsinn",
        "je", "größte", "schlimmste", "beste", "krass", "unglaublich", "nimm",
    }),
    emotion=frozenset({
        "liebe", "hasse", "angst", "erstaunlich", "unglaublich", "schrecklich",
        "schockierend", "wahnsinnig", "schön", "schmerzhaft", "lustig",
        "beängstigend", "aufregend", "herzzerreißend", "kraftvoll", "frustrierend",
        "dankbar", "wütend", "glücklich", "stolz", "begeistert", "wow", "krass",
    }),
    payoff=frozenset({
        "weil", "also", "deshalb", "deswegen", "ergebnis", "erkannt", "gelernt",
        "lektion", "punkt", "bedeutet", "antwort", "endlich", "entdeckt",
        "schlüssel", "letztendlich", "fazit", "darum",
    }),
    dangling=frozenset({
        "und", "aber", "oder", "weil", "der", "die", "das", "dass", "es", "sie",
        "er", "dies", "diese", "dann", "auch", "jedoch", "denn", "dieser",
    }),
    second_person=frozenset({"du", "dich", "dir", "dein", "deine", "deinen", "ihr", "euch", "euer"}),
    quote_extra=frozenset({
        "alles", "nichts", "etwas", "leben", "welt", "moment", "ändern", "jetzt", "heute",
    }),
    enumeration=frozenset({
        "erstens", "zweitens", "drittens", "drei", "zwei", "schritte", "wege",
        "gründe", "tipps", "punkte",
    }),
)

_LEXICONS: dict[str, Lexicon] = {"en": _EN, "de": _DE}


def get_lexicon(lang: str | None) -> Lexicon:
    return _LEXICONS.get((lang or "en").lower()[:2], _EN)


_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)


def _tokens(words: list[Word]) -> list[str]:
    out: list[str] = []
    for w in words:
        out.extend(m.lower() for m in _WORD_RE.findall(w.text))
    return out


def _has_number(words: list[Word]) -> bool:
    return any(re.search(r"\d", w.text) for w in words)


def _ends_complete(words: list[Word]) -> bool:
    return bool(words) and words[-1].text.strip().endswith((".", "!", "?", '"'))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Features. Each returns (value 0..1, reason). `lex` defaults to English.
# --------------------------------------------------------------------------- #
def hook_strength(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Curiosity in the first ~3 seconds — does it stop the scroll?"""
    if not words:
        return 0.0, ""
    start = words[0].t
    head = [w for w in words if w.t - start < 3.0] or words[:8]
    toks = _tokens(head)
    if not toks:
        return 0.0, ""
    hooks = sum(t in lex.hook for t in toks)
    second_person = sum(t in lex.second_person for t in toks)
    question = any(w.text.strip().endswith("?") for w in head)
    number = _has_number(head)
    raw = 0.34 * min(hooks, 2) / 2 + 0.22 * min(second_person, 2) / 2
    raw += 0.28 * question + 0.16 * number
    reason = "Opens with a question" if question else (
        "Curiosity hook up front" if hooks else (
            "Speaks directly to the viewer" if second_person else "Opens on a concrete detail"))
    return _clamp(raw), reason


def emotional_payoff(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    toks = _tokens(words)
    if not toks:
        return 0.0, ""
    hits = sum(t in lex.emotion for t in toks)
    density = hits / max(len(toks), 1)
    return _clamp(density * 9.0), "Clear emotional charge"


def standalone_clarity(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """Does it stand on its own — clean open, complete close?"""
    if not words:
        return 0.0, ""
    first = _WORD_RE.findall(words[0].text.lower())
    dangling = bool(first) and first[0] in lex.dangling
    complete = _ends_complete(words)
    raw = (0.0 if dangling else 0.55) + (0.45 if complete else 0.0)
    reason = "Clean standalone story" if not dangling and complete else (
        "Resolves cleanly" if complete else "Self-contained")
    return _clamp(raw), reason


def pace_energy(words: list[Word], duration: float) -> tuple[float, str]:
    if duration <= 0:
        return 0.0, ""
    wps = len(words) / duration
    if wps < 1.4 or wps > 4.5:
        raw = 0.2
    else:
        raw = 1.0 - abs(wps - 2.8) / 2.0
    return _clamp(raw), "Energetic, well-paced delivery"


def quotability(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    """A short, punchy line lands well as a standalone quote."""
    if not words:
        return 0.0, ""
    toks = _tokens(words)
    strong = sum(t in lex.quote for t in toks)
    short_punchy = len(words) <= 28
    raw = _clamp(0.5 * min(strong, 4) / 4 + (0.5 if short_punchy else 0.2))
    return raw, "Quotable, punchy line"


def length_fit(duration: float, min_len: float, max_len: float) -> tuple[float, str]:
    ideal = (min_len + max_len) / 2.0
    span = max((max_len - min_len) / 2.0, 1.0)
    raw = _clamp(1.0 - abs(duration - ideal) / (span * 1.6))
    return raw, "Ideal length for the format"


def list_payoff(words: list[Word], lex: Lexicon = _EN) -> tuple[float, str]:
    has_num = _has_number(words)
    toks = _tokens(words)
    enumeration = any(t in lex.enumeration for t in toks)
    raw = _clamp(0.6 * has_num + 0.6 * enumeration)
    return raw, "Concrete, list-style payoff"
