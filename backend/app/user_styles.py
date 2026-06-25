"""User-defined caption style templates (brand templates).

Extends the built-in styles (``app/styles.py``) with custom templates the user
creates. Stored as a JSON file alongside the project database so they persist
across restarts.

API:
    GET    /api/styles          — returns built-in + user templates
    POST   /api/styles          — create a new custom template
    PUT    /api/styles/<id>     — update an existing custom template
    DELETE /api/styles/<id>     — delete a custom template
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import get_settings
from .models import StyleTemplate
from .styles import _TEMPLATES as _BUILTIN

_lock = threading.Lock()


def _user_styles_path() -> Path:
    return get_settings().data_dir / "user-styles.json"


def _load_user_styles() -> dict[str, StyleTemplate]:
    p = _user_styles_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {sid: StyleTemplate.model_validate(item) for sid, item in raw.items()}
    except Exception:
        return {}


def _save_user_styles(styles: dict[str, StyleTemplate]) -> None:
    p = _user_styles_path()
    raw = {sid: s.model_dump() for sid, s in styles.items()}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def all_styles() -> list[StyleTemplate]:
    """Return built-in + user-defined styles (user styles override built-in ids)."""
    with _lock:
        user = _load_user_styles()
    merged = dict(_BUILTIN)
    merged.update(user)
    return list(merged.values())


def get_style(style_id: str | None) -> StyleTemplate:
    """Get a style by id, with user styles taking priority over built-in."""
    if not style_id:
        from .styles import DEFAULT_STYLE_ID
        style_id = DEFAULT_STYLE_ID
    with _lock:
        user = _load_user_styles()
    if style_id in user:
        return user[style_id]
    from .styles import get_style as get_builtin
    return get_builtin(style_id)


def create_style(style: StyleTemplate) -> StyleTemplate:
    """Save a new custom style. Overwrites if the id already exists."""
    with _lock:
        user = _load_user_styles()
        user[style.id] = style
        _save_user_styles(user)
    return style


def update_style(style_id: str, updates: dict) -> StyleTemplate | None:
    """Update fields on an existing user style. Returns None if not found."""
    with _lock:
        user = _load_user_styles()
        if style_id not in user:
            return None
        existing = user[style_id]
        updated = existing.model_copy(update=updates)
        user[style_id] = updated
        _save_user_styles(user)
    return updated


def delete_style(style_id: str) -> bool:
    """Delete a user-defined style. Returns True if it existed."""
    with _lock:
        user = _load_user_styles()
        if style_id not in user:
            return False
        del user[style_id]
        _save_user_styles(user)
    return True
