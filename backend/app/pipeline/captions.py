"""Burned-in caption generation as ASS (Advanced SubStation) for libass.

We render social-native captions: large, high-contrast, positioned in the safe
zone, with the spoken word highlighted (and slightly enlarged) one at a time —
the TikTok/Reels look. Word timing comes straight from the transcript, so the
highlight stays locked to the audio.

We emit one Dialogue event per word, tiled so each word's highlight persists
until the next word begins (no flicker). Words are grouped into short on-screen
lines for phone readability.
"""
from __future__ import annotations

from pathlib import Path

from ..models import CaptionSet, StyleTemplate


def _ass_color(rrggbb: str) -> str:
    """RRGGBB -> ASS &H00BBGGRR& (opaque)."""
    rr, gg, bb = rrggbb[0:2], rrggbb[2:4], rrggbb[4:6]
    return f"&H00{bb}{gg}{rr}".upper() + "&"


def _esc(text: str) -> str:
    return text.replace("\\", "⧵").replace("{", "(").replace("}", ")").replace("\n", " ")


def _ts(t: float) -> str:
    # Integer centisecond math — float formatting can yield an invalid ":60.00"
    # for values like 59.999.
    cs = max(int(round(t * 100)), 0)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{c:02d}"


# A word normally stays on screen until the next word starts (no flicker). But
# across a real pause — silence, or a span where another (toggled-off) speaker
# was talking — holding the word that long leaves a caption frozen on a silent
# shot. So once the gap past a word exceeds SILENCE_GAP, the caption clears
# LINGER_PAD after the word instead of lingering. Must be < LINE_GAP, or lines
# always break before the clamp can fire and captions over-hold into silence.
SILENCE_GAP = 0.35
LINGER_PAD = 0.15
# Start a fresh caption line after a pause this long, even mid-count — keeps a
# line from spanning silence so captions begin/end with the speech.
LINE_GAP = 0.9


def _group_lines(words, n: int, max_gap: float = LINE_GAP):
    """Group words into on-screen lines: a new line every ``n`` words OR after a
    speech pause longer than ``max_gap`` (whichever comes first)."""
    lines: list = []
    cur: list = []
    for w in words:
        if cur and (len(cur) >= n or (w.t - (cur[-1].t + cur[-1].d)) > max_gap):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _srt_ts(t: float) -> str:
    ms_total = max(int(round(t * 1000)), 0)
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(captions: CaptionSet) -> str:
    """Plain .srt sidecar (clip-relative times) for editing in an NLE."""
    lines = _group_lines(captions.words, captions.max_words_per_line)
    out: list[str] = []
    idx = 1
    for line in lines:
        if not line:
            continue
        start, end = line[0].t, line[-1].t + line[-1].d
        text = " ".join(w.text for w in line).strip()
        if not text:
            continue
        out.append(f"{idx}\n{_srt_ts(start)} --> {_srt_ts(end)}\n{text}\n")
        idx += 1
    return "\n".join(out) + "\n"


def build_ass(captions: CaptionSet, style: StyleTemplate,
              out_w: int, out_h: int, *, ai_boost=None) -> str:
    """Return a complete ASS document for one clip.

    ``ai_boost`` is an optional ``AiBoostSettings`` instance. When provided, its
    ``speakerColors`` flag gates the per-speaker colour pass so the project-level
    AI Boost toggle works. Default (None) preserves the old behaviour — speaker
    colours are always on when multiple speakers are present.
    """
    primary = _ass_color(style.primary)
    highlight = _ass_color(style.highlight)
    outline = _ass_color(style.outline)
    margin_v = int((1.0 - style.y_frac) * out_h)

    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {out_w}
PlayResY: {out_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{style.font_size},{primary.rstrip("&")},{primary.rstrip("&")},{outline.rstrip("&")},&H64000000,1,0,0,0,100,100,0,0,1,{style.outline_w},2,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""

    # Production-value pass: mark power words for keyword emphasis + auto-emoji
    # (Submagic/Hormozi look), language-aware, capped per line so it stays
    # tasteful. Honoured below in _dialogue; no-op when the style opts out.
    from .caption_fx import annotate
    words = annotate(captions.words, lang=captions.lang,
                     emphasis=style.emphasis, emoji=style.emoji,
                     max_words_per_line=captions.max_words_per_line)
    lines = _group_lines(words, captions.max_words_per_line)

    # Speaker-aware caption colours: when more than one speaker is present,
    # give each their own primary colour so a podcast reads as a conversation
    # (host vs guest) rather than a monochrome wall. The style's primary is
    # still used as speaker 0's colour so a single-speaker clip is unchanged.
    # Gated by ai_boost.speakerColors (default on) so the per-project toggle
    # in the AI Boost panel can disable it.
    speakers = sorted({getattr(w, "speaker", 0) or 0 for w in words})
    speaker_colors: dict[int, str] = {}
    if len(speakers) > 1 and (ai_boost is None or ai_boost.speakerColors):
        palette = ["F4F4F8", "FFD166", "06D6A0", "EF476F", "8338EC", "3A86FF"]
        for i, sp in enumerate(speakers):
            speaker_colors[sp] = _ass_color(palette[i % len(palette)])
    else:
        speaker_colors[speakers[0] if speakers else 0] = primary

    events: list[str] = []
    for line in lines:
        if not line:
            continue
        line_end = line[-1].t + line[-1].d
        # The active word's speaker picks this line's primary colour.
        line_primary = speaker_colors.get(
            getattr(line[0], "speaker", 0) or 0, primary)
        for idx, w in enumerate(line):
            start = w.t
            # Hold until the next word in the line starts; last word holds to its end.
            end = line[idx + 1].t if idx + 1 < len(line) else max(w.t + w.d, line_end)
            # Don't let a word freeze on screen through a silence/other-speaker
            # gap — clear it shortly after it's spoken instead.
            if end - (w.t + w.d) > SILENCE_GAP:
                end = w.t + w.d + LINGER_PAD
            # Unconditional floor: a zero/negative-duration Dialogue (words that
            # share a timestamp, a near-zero w.d) would be dropped by libass and
            # the highlight flickers off. Always keep a visible minimum span.
            end = max(end, start + 0.08)
            events.append(_dialogue(line, idx, start, end, line_primary, highlight,
                                    style.uppercase))

    return head + "\n".join(events) + "\n"


def _dialogue(line, active_idx, start, end, primary, highlight, upper) -> str:
    parts: list[str] = []
    for i, w in enumerate(line):
        token = _esc(w.text)
        if upper:
            token = token.upper()
        emoji = getattr(w, "emoji", None)
        if emoji:
            token = f"{token} {emoji}"
        if i == active_idx:
            # active word: highlight colour + a slight pop for the animated feel
            parts.append(f"{{\\c{highlight}\\fscx112\\fscy112}}{token}{{\\c{primary}\\fscx100\\fscy100}}")
        elif getattr(w, "emphasis", False):
            # power word: stays highlighted + slightly larger for the whole line
            # (keyword emphasis) even when it isn't the currently-spoken word.
            parts.append(f"{{\\c{highlight}\\fscx106\\fscy106}}{token}{{\\c{primary}\\fscx100\\fscy100}}")
        else:
            parts.append(token)
    text = " ".join(parts)
    # Fields: Layer,Start,End,Style,MarginL,MarginR,MarginV,Effect,Text — exactly
    # 8 commas before Text, or Text inherits a stray leading comma.
    return f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,0,0,0,,{text}"


def write_ass(captions: CaptionSet, style: StyleTemplate,
              out_w: int, out_h: int, dst: str | Path, *,
              ai_boost=None) -> Path:
    dst = Path(dst)
    dst.write_text(build_ass(captions, style, out_w, out_h, ai_boost=ai_boost),
                   encoding="utf-8")
    return dst
