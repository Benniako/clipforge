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
            outline_w=7, y_frac=0.76, uppercase=True,
        ),
        StyleTemplate(
            id="clean-minimal", name="Clean Minimal",
            font="DejaVu Sans", font_size=82,
            primary="FFFFFF", highlight="3DDC97", outline="101010",
            outline_w=4, y_frac=0.80, uppercase=False,
        ),
        StyleTemplate(
            id="hype-yellow", name="Hype",
            font="FreeSans", font_size=104,
            primary="FFE45C", highlight="FF4D4D", outline="1A1A1A",
            outline_w=8, y_frac=0.72, uppercase=True,
        ),
        StyleTemplate(
            id="news-lower", name="Lower Third",
            font="DejaVu Sans", font_size=70,
            primary="FFFFFF", highlight="59A5FF", outline="0A1A2F",
            outline_w=5, y_frac=0.86, uppercase=False,
        ),
    ]
}

DEFAULT_STYLE_ID = "bold-pop"


def get_style(style_id: str | None) -> StyleTemplate:
    return _TEMPLATES.get(style_id or DEFAULT_STYLE_ID, _TEMPLATES[DEFAULT_STYLE_ID])


def all_styles() -> list[StyleTemplate]:
    return list(_TEMPLATES.values())
