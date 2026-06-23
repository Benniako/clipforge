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

import json
import logging
import re
from pathlib import Path

from ..models import CaptionWord
from ..providers import signals

log = logging.getLogger("clipforge.caption_fx")

_WORD_RE = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d")

# Built-in fallback: used when the editable emoji_map.json is missing or
# unparseable so captions always render. Users add their own pairs by editing
# emoji_map.json next to this file (no restart needed — reloaded per render).
_EMOJI_FALLBACK: dict[str, str] = {
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

_EMOJI_PATH = Path(__file__).parent / "emoji_map.json"
_EMOJI_CACHE: tuple[float, dict[str, str]] | None = None  # (mtime, map)


def _load_emoji_map() -> dict[str, str]:
    """Load the editable emoji map, falling back to the built-in dict.

    Cached with mtime invalidation so a hot edit to emoji_map.json takes effect
    on the next render without a process restart. Missing/corrupt file degrades
    silently to the fallback — captions must never fail to build.
    """
    global _EMOJI_CACHE
    try:
        mtime = _EMOJI_PATH.stat().st_mtime
    except OSError:
        return _EMOJI_FALLBACK  # file absent — built-ins are still good captions
    if _EMOJI_CACHE and _EMOJI_CACHE[0] == mtime:
        return _EMOJI_CACHE[1]
    try:
        with _EMOJI_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
        # Drop the _comment key + any non-string entries; keep the rest verbatim.
        merged = {k: v for k, v in raw.items()
                  if not k.startswith("_") and isinstance(k, str) and isinstance(v, str)}
        _EMOJI_CACHE = (mtime, merged)
        return merged
    except (json.JSONDecodeError, OSError) as e:
        log.warning("emoji_map.json unreadable (%s); using built-in emoji set", e)
        return _EMOJI_FALLBACK


def _norm(text: str) -> str:
    m = _WORD_RE.search(text.lower())
    return m.group(0) if m else ""


def _emphasis_set(lex: signals.Lexicon) -> frozenset[str]:
    """Words worth emphasising: the quotable set + payoff + enumeration markers."""
    return lex.quote | lex.payoff | lex.enumeration


# When False, the per-line cap is computed only against `max_words_per_line`. The
# real on-screen grouping in captions.py also breaks on speech-pause gaps wider
# than LINE_GAP, so we accept a ``line_break`` predicate from the caller (defaults
# to "after every N words") to keep the two views consistent.
def _default_line_break(prev: CaptionWord | None, w: CaptionWord, *, line_gap: float
                        ) -> bool:
    return prev is not None and (w.t - (prev.t + prev.d)) > line_gap


def annotate(words: list[CaptionWord], *, lang: str = "en",
             emphasis: bool = True, emoji: bool = False,
             max_words_per_line: int = 3,
             max_emphasis_per_line: int = 2,
             max_emoji_per_line: int = 1,
             line_gap: float = 0.9) -> list[CaptionWord]:
    """Return a copy of ``words`` with ``emphasis``/``emoji`` marks set.

    Emphasis is capped per on-screen line so a whole line never lights up (which
    would defeat the point); numbers always count as emphasis (they're the most
    scannable beat). Emoji are rarer still. Pure — mutates nothing in place.

    A new line starts after ``max_words_per_line`` words OR after a speech pause
    longer than ``line_gap`` — the same rule the renderer uses for line breaks
    in ``captions._group_lines``, so a single-word line after a pause still
    obeys its own emphasis budget instead of inheriting a stale one.
    """
    if not words or (not emphasis and not emoji):
        return words
    lex = signals.get_lexicon(lang)
    emph = _emphasis_set(lex)
    emoji_map = _load_emoji_map() if emoji else {}
    out: list[CaptionWord] = []
    line_emph = 0
    line_emoji = 0
    pos_in_line = 0
    prev: CaptionWord | None = None
    for w in words:
        if pos_in_line >= max_words_per_line or _default_line_break(
                prev, w, line_gap=line_gap):
            pos_in_line = line_emph = line_emoji = 0  # new on-screen line
        tok = _norm(w.text)
        is_num = bool(_NUM_RE.search(w.text))
        mark_e = False
        emo: str | None = None
        if emphasis and (is_num or tok in emph) and line_emph < max_emphasis_per_line:
            mark_e = True
            line_emph += 1
        if emoji and tok in emoji_map and line_emoji < max_emoji_per_line:
            emo = emoji_map[tok]
            line_emoji += 1
        out.append(w.model_copy(update={"emphasis": mark_e, "emoji": emo}))
        pos_in_line += 1
        prev = w
    return out
