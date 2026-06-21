"""User-saved visual/OCR cue phrases and calibrated screen regions.

Audio cues are saved as reference sounds. Visual cues are saved as phrases that
OCR should treat as named events, optional regions where the OCR should look,
and false-positive phrases that should be ignored in future scans.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import get_settings


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9äöüß_-]", "", (name or "").lower().replace(" ", "_"))[:40] or "cue"


def _dir() -> Path:
    d = get_settings().data_dir / "visual_cues"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(game: str) -> Path:
    return _dir() / f"{_safe(game)}.json"


def _empty() -> dict[str, Any]:
    return {"phrases": {}, "regions": {}, "false": {}}


def _fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    folded = folded.encode("ascii", "ignore").decode("ascii").lower()
    folded = re.sub(r"[^a-z0-9 ]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _clean_phrase_list(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        phrase = " ".join(str(item or "").split())
        key = phrase.lower()
        if phrase and key not in seen:
            out.append(phrase)
            seen.add(key)
    return out


def _clean_phrase_map(raw: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return out
    for label, phrases in raw.items():
        clean = _clean_phrase_list(phrases)
        if clean:
            out[_safe(str(label))] = clean
    return out


def _clean_region(raw: Any) -> dict[str, float | str] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x", 0.0))
        y = float(raw.get("y", 0.0))
        w = float(raw.get("w", 1.0))
        h = float(raw.get("h", 1.0))
    except (TypeError, ValueError):
        return None
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    w = max(0.01, min(1.0 - x, w))
    h = max(0.01, min(1.0 - y, h))
    return {
        "name": _safe(str(raw.get("name") or raw.get("label") or "region")),
        "x": round(x, 4),
        "y": round(y, 4),
        "w": round(w, 4),
        "h": round(h, 4),
    }


def _clean_region_map(raw: Any) -> dict[str, list[dict[str, float | str]]]:
    out: dict[str, list[dict[str, float | str]]] = {}
    if not isinstance(raw, dict):
        return out
    for label, regions in raw.items():
        if not isinstance(regions, list):
            continue
        clean = [r for r in (_clean_region(x) for x in regions) if r is not None]
        if clean:
            out[_safe(str(label))] = clean
    return out


def _read_doc(game: str) -> dict[str, Any]:
    p = _path(game)
    if not p.exists():
        return _empty()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _empty()
    if not isinstance(raw, dict):
        return _empty()

    # Back-compat: older files were simply {"label": ["phrase", ...]}.
    if not any(k in raw for k in ("phrases", "regions", "false")):
        return {"phrases": _clean_phrase_map(raw), "regions": {}, "false": {}}

    return {
        "phrases": _clean_phrase_map(raw.get("phrases")),
        "regions": _clean_region_map(raw.get("regions")),
        "false": _clean_phrase_map(raw.get("false")),
    }


def _write_doc(game: str, doc: dict[str, Any]) -> None:
    p = _path(game)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "phrases": _clean_phrase_map(doc.get("phrases")),
        "regions": _clean_region_map(doc.get("regions")),
        "false": _clean_phrase_map(doc.get("false")),
    }
    p.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def _read(game: str) -> dict[str, list[str]]:
    return _read_doc(game)["phrases"]


def _write(game: str, data: dict[str, list[str]]) -> None:
    doc = _read_doc(game)
    doc["phrases"] = data
    _write_doc(game, doc)


def _append_unique(items: list[str], phrase: str) -> None:
    phrase = " ".join((phrase or "").split())
    if phrase and phrase.lower() not in {p.lower() for p in items}:
        items.append(phrase)


def add_visual_cue(game: str, label: str, phrase: str) -> dict[str, list[str]]:
    label = _safe(label)
    phrase = " ".join((phrase or "").split())
    if not phrase:
        raise ValueError("visual cue text is empty")
    doc = _read_doc(game)
    phrases = doc["phrases"].setdefault(label, [])
    _append_unique(phrases, phrase)
    _write_doc(game, doc)
    return doc["phrases"]


def add_visual_region(game: str, label: str, box: dict[str, float],
                      name: str | None = None) -> dict[str, Any]:
    label = _safe(label)
    region = _clean_region({**box, "name": name or label})
    if region is None:
        raise ValueError("visual cue region is invalid")
    doc = _read_doc(game)
    regions = doc["regions"].setdefault(label, [])
    key = (region["x"], region["y"], region["w"], region["h"])
    if not any((r.get("x"), r.get("y"), r.get("w"), r.get("h")) == key for r in regions):
        regions.append(region)
    _write_doc(game, doc)
    return doc


def add_false_visual_cue(game: str, label: str, phrase: str) -> dict[str, Any]:
    label = _safe(label)
    phrase = " ".join((phrase or "").split())
    if not phrase:
        raise ValueError("false-positive text is empty")
    doc = _read_doc(game)
    phrases = doc["false"].setdefault(label, [])
    _append_unique(phrases, phrase)
    _write_doc(game, doc)
    return doc


def remove_visual_cue(game: str, label: str, phrase: str | None = None) -> dict[str, list[str]]:
    doc = _read_doc(game)
    label = _safe(label)
    data = doc["phrases"]
    if label not in data:
        return data
    if phrase is None:
        data.pop(label, None)
    else:
        target = phrase.strip().lower()
        data[label] = [p for p in data[label] if p.lower() != target]
        if not data[label]:
            data.pop(label, None)
    _write_doc(game, doc)
    return data


def list_visual_cues() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for p in _dir().glob("*.json"):
        out[p.stem] = _read(p.stem)
    return out


def list_visual_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in _dir().glob("*.json"):
        out[p.stem] = _read_doc(p.stem)
    return out


def _profile_names(profile: str | None) -> list[str]:
    names = ["common"]
    name = _safe(profile or "generic")
    if name not in names:
        names.append(name)
    return names


def lexicon_extra(profile: str | None) -> dict[str, tuple[str, ...]]:
    """Return common + profile visual cues in OCR lexicon shape."""
    merged: dict[str, list[str]] = {}
    for game in _profile_names(profile):
        for label, phrases in _read_doc(game)["phrases"].items():
            merged.setdefault(label, [])
            for phrase in phrases:
                _append_unique(merged[label], phrase)
    return {k: tuple(v) for k, v in merged.items()}


def regions_extra(profile: str | None) -> dict[str, list[dict[str, float | str]]]:
    """Return common + profile calibrated OCR regions."""
    merged: dict[str, list[dict[str, float | str]]] = {}
    for game in _profile_names(profile):
        for label, regions in _read_doc(game)["regions"].items():
            merged.setdefault(label, [])
            for region in regions:
                if region not in merged[label]:
                    merged[label].append(region)
    return merged


def is_false_positive(profile: str | None, label: str, text: str) -> bool:
    """True when user feedback says this OCR text should not count."""
    label = _safe(label)
    norm_text = _fold(text)
    if not norm_text:
        return False
    for game in _profile_names(profile):
        false_map = _read_doc(game)["false"]
        labels = [label, "all", "*"] + [lab for lab in false_map if lab not in {label, "all", "*"}]
        for lab in labels:
            for phrase in false_map.get(lab, []):
                norm_phrase = _fold(phrase)
                if norm_phrase and norm_phrase in norm_text:
                    return True
    return False
