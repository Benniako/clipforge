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
# was talking — holding the last word that long leaves a caption frozen on a
# silent shot. So once the gap past a word exceeds SILENCE_GAP, the caption
# clears LINGER_PAD after the word instead of lingering.
SILENCE_GAP = 1.0
LINGER_PAD = 0.4
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
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
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
              out_w: int, out_h: int) -> str:
    """Return a complete ASS document for one clip."""
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

    words = captions.words
    lines = _group_lines(words, captions.max_words_per_line)
    events: list[str] = []

    for line in lines:
        if not line:
            continue
        line_end = line[-1].t + line[-1].d
        for idx, w in enumerate(line):
            start = w.t
            # Hold until the next word in the line starts; last word holds to its end.
            end = line[idx + 1].t if idx + 1 < len(line) else max(w.t + w.d, line_end)
            # Don't let a word freeze on screen through a silence/other-speaker
            # gap — clear it shortly after it's spoken instead.
            if end - (w.t + w.d) > SILENCE_GAP:
                end = w.t + w.d + LINGER_PAD
            if end <= start:
                end = start + 0.08
            events.append(_dialogue(line, idx, start, end, primary, highlight,
                                    style.uppercase))

    return head + "\n".join(events) + "\n"


def _dialogue(line, active_idx, start, end, primary, highlight, upper) -> str:
    parts: list[str] = []
    for i, w in enumerate(line):
        token = _esc(w.text)
        if upper:
            token = token.upper()
        if i == active_idx:
            # active word: highlight colour + a slight pop for the animated feel
            parts.append(f"{{\\c{highlight}\\fscx112\\fscy112}}{token}{{\\c{primary}\\fscx100\\fscy100}}")
        else:
            parts.append(token)
    text = " ".join(parts)
    # Fields: Layer,Start,End,Style,MarginL,MarginR,MarginV,Effect,Text — exactly
    # 8 commas before Text, or Text inherits a stray leading comma.
    return f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,0,0,0,,{text}"


def write_ass(captions: CaptionSet, style: StyleTemplate,
              out_w: int, out_h: int, dst: str | Path) -> Path:
    dst = Path(dst)
    dst.write_text(build_ass(captions, style, out_w, out_h), encoding="utf-8")
    return dst
