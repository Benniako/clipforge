"""Cue packs — the canonical "what sounds to add" per game.

ClipForge matches reference game sounds (see app/providers/detect_cues.py) to
pinpoint events. We can't ship the copyrighted audio, but we ship the *structure*:
which events each supported game has, a search hint for finding the sound, and a
status check for which cues you've added. Drop files (any audio format) named
``<event>.<ext>`` into ``<data>/game_cues/<game>/``.
"""
from __future__ import annotations

import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .config import get_settings
from .media import ffmpeg
from .providers.detect_cues import CUE_EXTS

PACKS: dict[str, dict] = {
    "valorant": {"label": "Valorant", "events": [
        ("kill", "Kill banner ‘ding’", "valorant kill sound"),
        ("ace", "Ace announcer line", "valorant ace sound"),
        ("spike_plant", "Spike planted", "valorant spike planted sound"),
        ("spike_defuse", "Spike defused", "valorant defuse sound"),
    ]},
    "cs2": {"label": "CS2", "events": [
        ("kill", "Kill confirm", "cs2 kill sound"),
        ("headshot", "Headshot ‘ding’", "cs2 headshot sound"),
        ("bomb_plant", "Bomb planted", "cs2 bomb planted sound"),
        ("bomb_defuse", "Bomb defused", "cs2 defuse sound"),
    ]},
    "eafc": {"label": "EA FC / FIFA", "events": [
        ("goal", "Goal net + commentary", "ea fc goal sound"),
        ("whistle", "Referee whistle", "fc 26 referee whistle"),
        ("crowd_roar", "Crowd roar", "football crowd roar"),
    ]},
    "rocketleague": {"label": "Rocket League", "events": [
        ("goal", "Goal explosion", "rocket league goal explosion sound"),
        ("demolition", "Demolition", "rocket league demolition sound"),
        ("save", "Save", "rocket league save sound"),
    ]},
    "horror": {"label": "Horror", "events": [
        ("stinger", "Musical sting", "horror stinger sound"),
        ("scream", "Scream", "scream sound effect"),
        ("jumpscare", "Jump-scare hit", "jumpscare sound"),
    ]},
}


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", (name or "").lower().replace(" ", "_"))[:40] or "cue"


def _dir(game: str):
    return get_settings().data_dir / "game_cues" / _safe(game)


def install_cue(game: str, event: str, src_path: str) -> None:
    """Normalise any audio file into a 16 kHz mono cue at <game>/<event>.wav."""
    dest = _dir(game) / f"{_safe(event)}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(["-i", str(src_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(dest)])


def install_cue_from_url(game: str, event: str, url: str) -> None:
    """Download a sound (e.g. a MyInstants/soundboard link) and install it.

    Local single-user tool: we fetch a user-supplied URL on their own machine.
    """
    if not (url or "").lower().startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are supported")
    with tempfile.TemporaryDirectory() as tmp:
        dl = Path(tmp) / "in"
        req = urllib.request.Request(url, headers={"User-Agent": "ClipForge/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dl, "wb") as f:
            shutil.copyfileobj(r, f)
        install_cue(game, event, str(dl))


def remove_cue(game: str, event: str) -> bool:
    d = _dir(game)
    removed = False
    if d.is_dir():
        for p in d.glob(f"{_safe(event)}.*"):
            if p.suffix.lower() in CUE_EXTS:
                p.unlink(missing_ok=True)
                removed = True
    return removed


def pack_status() -> dict:
    """Per game: which events have a cue file present."""
    out: dict[str, dict] = {}
    for game, pack in PACKS.items():
        d = _dir(game)
        present = {p.stem.lower() for p in d.glob("*") if p.suffix.lower() in CUE_EXTS} if d.is_dir() else set()
        events = [{"name": n, "desc": desc, "hint": hint, "configured": n.lower() in present}
                  for (n, desc, hint) in pack["events"]]
        out[game] = {
            "label": pack["label"],
            "events": events,
            "configured": sum(1 for e in events if e["configured"]),
            "total": len(events),
        }
    return out
