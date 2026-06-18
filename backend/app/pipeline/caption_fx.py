"""Caption production-value pass — keyword emphasis + tasteful auto-emoji.

The single biggest thing that makes social captions read as *professionally
edited* (Submagic / Hormozi / MrBeast style) is that the **meaningful** word in a
line is emphasised — coloured and enlarged for the whole line, not just while it's
spoken — and the occasional power word gets an emoji. This module marks those,
purely and language-aware, reusing the same signal lexicons the scorer uses so
"emphasis" means the same thing as "what makes this quotable".

Pure and deterministic (no I/O), so it's fully unit-tested. Applied at
caption-build time; rendering (captions.py) just honours the marks.
"""
from __future__ import annotations

import re

from ..models import CaptionWord
from ..providers import signals

_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d")

# Power word -> emoji. Kept small and unambiguous; matched on the normalized
# token. Language-shared where the concept is the same; de adds its own keys.
_EMOJI: dict[str, str] = {
    # money / growth
    "money": "💰", "rich": "💰", "cash": "💰", "profit": "📈", "millionaire": "💰",
    "geld": "💰", "reich": "💰",
    # fire / hype
    "fire": "🔥", "insane": "🔥", "crazy": "🔥", "wild": "🔥", "best": "🔥",
    "amazing": "🔥", "incredible": "🤯", "unbelievable": "🤯", "shocking": "😱",
    "wahnsinn": "🔥", "krass": "🔥", "unglaublich": "🤯",
    # mind / ideas
    "secret": "🤫", "truth": "💡", "idea": "💡", "mistake": "❌", "stop": "✋",
    "geheimnis": "🤫", "wahrheit": "💡", "fehler": "❌", "stopp": "✋",
    # emotion
    "love": "❤️", "hate": "😤", "fear": "😨", "scary": "😱", "happy": "😄",
    "liebe": "❤️", "angst": "😨",
    # win / time
    "win": "🏆", "won": "🏆", "victory": "🏆", "finally": "🙌", "now": "⏰",
    "sieg": "🏆", "endlich": "🙌", "jetzt": "⏰",
}


def _norm(text: str) -> str:
    m = _WORD_RE.search(text.lower())
    return m.group(0) if m else ""


def _emphasis_set(lex: signals.Lexicon) -> frozenset[str]:
    """Words worth emphasising: the quotable set + payoff + enumeration markers."""
    return lex.quote | lex.payoff | lex.enumeration


def annotate(words: list[CaptionWord], *, lang: str = "en",
             emphasis: bool = True, emoji: bool = False,
             max_words_per_line: int = 3,
             max_emphasis_per_line: int = 2,
             max_emoji_per_line: int = 1) -> list[CaptionWord]:
    """Return a copy of ``words`` with ``emphasis``/``emoji`` marks set.

    Emphasis is capped per on-screen line so a whole line never lights up (which
    would defeat the point); numbers always count as emphasis (they're the most
    scannable beat). Emoji are rarer still. Pure — mutates nothing in place.
    """
    if not words or (not emphasis and not emoji):
        return words
    lex = signals.get_lexicon(lang)
    emph = _emphasis_set(lex)
    out: list[CaptionWord] = []
    line_emph = 0
    line_emoji = 0
    pos_in_line = 0
    for w in words:
        if pos_in_line >= max_words_per_line:
            pos_in_line = line_emph = line_emoji = 0  # new on-screen line
        tok = _norm(w.text)
        is_num = bool(_NUM_RE.search(w.text))
        mark_e = False
        emo: str | None = None
        if emphasis and (is_num or tok in emph) and line_emph < max_emphasis_per_line:
            mark_e = True
            line_emph += 1
        if emoji and tok in _EMOJI and line_emoji < max_emoji_per_line:
            emo = _EMOJI[tok]
            line_emoji += 1
        out.append(w.model_copy(update={"emphasis": mark_e, "emoji": emo}))
        pos_in_line += 1
    return out
