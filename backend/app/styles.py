"""Built-in caption style templates (PRD §3.2 — a small, social-native set).

Each template is a reusable look the user can pick per clip or set as a default.
Colours are stored RRGGBB and converted to ASS's &HBBGGRR& at render time.
"""
from __future__ import annotations

from .models import StyleTemplate

_TEMPLATES: dict[str, StyleTemplate] = {
    t.id: t
    for t in [
        StyleTemplate(
            id="bold-pop", name="Bold Pop",
            font="DejaVu Sans", font_size=96,
            primary="FFFFFF", highlight="F5C518", outline="000000",
            outline_w=7, y_frac=0.76, uppercase=True, emphasis=True,
        ),
        StyleTemplate(
            id="clean-minimal", name="Clean Minimal",
            font="DejaVu Sans", font_size=82,
            primary="FFFFFF", highlight="3DDC97", outline="101010",
            outline_w=4, y_frac=0.80, uppercase=False, emphasis=False,
        ),
        StyleTemplate(
            id="hype-yellow", name="Hype",
            font="FreeSans", font_size=104,
            primary="FFE45C", highlight="FF4D4D", outline="1A1A1A",
            outline_w=8, y_frac=0.72, uppercase=True, emphasis=True, emoji=True,
        ),
        StyleTemplate(
            id="news-lower", name="Lower Third",
            font="DejaVu Sans", font_size=70,
            primary="FFFFFF", highlight="59A5FF", outline="0A1A2F",
            outline_w=5, y_frac=0.86, uppercase=False, emphasis=False,
        ),
        # The MrBeast/Hormozi big-yellow look: black-outlined, all-caps, emphasis
        # on every power word, with the occasional emoji punch.
        StyleTemplate(
            id="creator-bold", name="Creator Bold",
            font="FreeSans", font_size=108,
            primary="FFE45C", highlight="FFFFFF", outline="000000",
            outline_w=9, y_frac=0.74, uppercase=True, emphasis=True, emoji=True,
        ),
        # High-energy gameplay caption: punchy green keyword + emoji.
        StyleTemplate(
            id="gamer-green", name="Gamer",
            font="DejaVu Sans", font_size=98,
            primary="FFFFFF", highlight="6CFF4D", outline="06140A",
            outline_w=7, y_frac=0.74, uppercase=True, emphasis=True, emoji=True,
        ),
        # Calm podcast look: soft pink keyword emphasis, no emoji, mixed case.
        StyleTemplate(
            id="podcast-soft", name="Podcast",
            font="DejaVu Sans", font_size=80,
            primary="F4F4F8", highlight="FF8FB1", outline="14101A",
            outline_w=4, y_frac=0.82, uppercase=False, emphasis=True,
        ),
        # The MrBeast signature: huge white text, thick black outline, every
        # power word punched in yellow. Maximum contrast, maximum readability —
        # the look that survives being thumb-stopped at 10% screen size.
        StyleTemplate(
            id="beast-outline", name="MrBeast Outline",
            font="FreeSans", font_size=112,
            primary="FFFFFF", highlight="FFE000", outline="000000",
            outline_w=10, y_frac=0.74, uppercase=True, emphasis=True, emoji=True,
        ),
        # TikTok bubble: rounded, all-white, soft drop shadow (no hard outline).
        # The native-TikTok look — reads clean over any background because the
        # shadow does the separation work instead of a stroke.
        StyleTemplate(
            id="tiktok-bubble", name="TikTok Bubble",
            font="DejaVu Sans", font_size=88,
            primary="FFFFFF", highlight="25F4EE", outline="000000",
            outline_w=0, y_frac=0.80, uppercase=False, emphasis=True, emoji=True,
        ),
        # Hormozi: the dense all-caps yellow wall. Very high information density,
        # aggressive emphasis, minimal whitespace. Signature Alex Hormozi hook look.
        StyleTemplate(
            id="hormozi-yellow", name="Hormozi",
            font="FreeSans", font_size=100,
            primary="FFE45C", highlight="FF3B3B", outline="0D0D0D",
            outline_w=8, y_frac=0.72, uppercase=True, emphasis=True, emoji=True,
        ),
        # Subtle/news: restrained, mixed-case, thin outline. For interview B-roll
        # and serious content where hype captions would undermine credibility.
        StyleTemplate(
            id="subtle-news", name="Subtle",
            font="DejaVu Sans", font_size=72,
            primary="FFFFFF", highlight="FFD166", outline="000000",
            outline_w=3, y_frac=0.84, uppercase=False, emphasis=False, emoji=False,
        ),
    ]
}

DEFAULT_STYLE_ID = "bold-pop"


def get_style(style_id: str | None) -> StyleTemplate:
    return _TEMPLATES.get(style_id or DEFAULT_STYLE_ID, _TEMPLATES[DEFAULT_STYLE_ID])


def all_styles() -> list[StyleTemplate]:
    return list(_TEMPLATES.values())
