"""User-saved visual/OCR cue phrases.

Audio cues are saved as reference sounds. Visual cues are saved as phrases that
OCR should treat as named events for a game profile.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import get_settings


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", (name or "").lower().replace(" ", "_"))[:40] or "cue"


def _dir() -> Path:
    d = get_settings().data_dir / "visual_cues"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(game: str) -> Path:
    return _dir() / f"{_safe(game)}.json"


def _read(game: str) -> dict[str, list[str]]:
    p = _path(game)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for label, phrases in raw.items():
            if isinstance(phrases, list):
                clean = [str(x).strip() for x in phrases if str(x).strip()]
                if clean:
                    out[_safe(str(label))] = clean
    return out


def _write(game: str, data: dict[str, list[str]]) -> None:
    p = _path(game)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_visual_cue(game: str, label: str, phrase: str) -> dict[str, list[str]]:
    label = _safe(label)
    phrase = " ".join((phrase or "").split())
    if not phrase:
        raise ValueError("visual cue text is empty")
    data = _read(game)
    phrases = data.setdefault(label, [])
    if phrase.lower() not in {p.lower() for p in phrases}:
        phrases.append(phrase)
    _write(game, data)
    return data


def remove_visual_cue(game: str, label: str, phrase: str | None = None) -> dict[str, list[str]]:
    data = _read(game)
    label = _safe(label)
    if label not in data:
        return data
    if phrase is None:
        data.pop(label, None)
    else:
        target = phrase.strip().lower()
        data[label] = [p for p in data[label] if p.lower() != target]
        if not data[label]:
            data.pop(label, None)
    _write(game, data)
    return data


def list_visual_cues() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for p in _dir().glob("*.json"):
        out[p.stem] = _read(p.stem)
    return out


def lexicon_extra(profile: str | None) -> dict[str, tuple[str, ...]]:
    """Return common + profile visual cues in OCR lexicon shape."""
    names = ["common"]
    name = _safe(profile or "generic")
    if name not in names:
        names.append(name)
    merged: dict[str, list[str]] = {}
    for game in names:
        for label, phrases in _read(game).items():
            merged.setdefault(label, [])
            for phrase in phrases:
                if phrase.lower() not in {p.lower() for p in merged[label]}:
                    merged[label].append(phrase)
    return {k: tuple(v) for k, v in merged.items()}
