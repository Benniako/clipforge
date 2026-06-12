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
import urllib.parse
import urllib.request
from pathlib import Path

from .config import get_settings
from .media import ffmpeg
from .providers.detect_cues import CUE_EXTS

PACKS: dict[str, dict] = {
    "valorant": {"label": "Valorant", "events": [
        ("kill", "Kill banner ‘ding’ (repeats = multi-kill, scores higher)", "valorant kill sound"),
        ("double_kill", "2nd-kill banner tone", "valorant double kill sound"),
        ("triple_kill", "3rd-kill banner tone", "valorant triple kill sound"),
        ("quad_kill", "4th-kill banner tone", "valorant quadra kill sound"),
        ("ace", "Ace announcer line", "valorant ace sound"),
        ("clutch", "Clutch announcer line", "valorant clutch sound"),
        ("spike_plant", "Spike planted", "valorant spike planted sound"),
        ("spike_defuse", "Spike defused", "valorant defuse sound"),
    ]},
    "cs2": {"label": "CS2", "events": [
        ("kill", "Kill confirm (repeats = multi-kill, scores higher)", "cs2 kill sound"),
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


# Audio file URL inside an HTML page (absolute, or quoted relative path like
# MyInstants' "/media/sounds/x.mp3").
_AUDIO_EXT = r"\.(?:mp3|wav|ogg|m4a|aac|flac)"
_ABS_AUDIO_RE = re.compile(r"https?://[^\"'\s<>]+" + _AUDIO_EXT, re.IGNORECASE)
_REL_AUDIO_RE = re.compile(r"[\"']([^\"'\s<>]+" + _AUDIO_EXT + r")[\"']", re.IGNORECASE)


def audio_url_from_html(html: str, base_url: str) -> str | None:
    """First audio-file URL referenced by an HTML page (absolute), or None.

    Lets users paste a soundboard *page* link (the natural thing to copy)
    instead of hunting for the raw .mp3 address.
    """
    m = _ABS_AUDIO_RE.search(html)
    if m:
        return m.group(0)
    m = _REL_AUDIO_RE.search(html)
    if m:
        return urllib.parse.urljoin(base_url, m.group(1))
    return None


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ClipForge/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def install_cue_from_url(game: str, event: str, url: str) -> None:
    """Download a sound (e.g. a MyInstants/soundboard link) and install it.

    Accepts a direct audio URL or an HTML page that references one (the
    sound link is then extracted and fetched). Local single-user tool: we
    fetch a user-supplied URL on their own machine.
    """
    if not (url or "").lower().startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are supported")
    with tempfile.TemporaryDirectory() as tmp:
        dl = Path(tmp) / "in"
        _download(url, dl)
        head = dl.read_bytes()[:256].lstrip()
        if head.startswith((b"<", b"\xef\xbb\xbf<")):  # an HTML page, not audio
            found = audio_url_from_html(
                dl.read_text(encoding="utf-8", errors="ignore"), url)
            if not found:
                raise ValueError(
                    "that page has no direct audio link — right-click the "
                    "sound's download button and paste the copied address")
            _download(found, dl)
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
