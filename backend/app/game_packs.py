"""Cue packs - the canonical "what sounds to add" per game.

ClipForge matches reference game sounds (see app/providers/detect_cues.py) to
pinpoint events. We can't ship the copyrighted audio, but we ship the structure:
which events each supported game has, a search hint for finding the sound, and a
status check for which cues you've added. Drop files (any audio format) named
``<event>.<ext>`` into ``<data>/game_cues/<game>/``.
"""
from __future__ import annotations

import re
import tempfile
import urllib.parse
from pathlib import Path

from ._util import http_download
from .config import get_settings
from .media import ffmpeg
from .providers.detect_cues import CUE_EXTS

PACKS: dict[str, dict] = {
    "valorant": {"label": "Valorant", "events": [
        ("kill", "Kill-Banner-Sound (mehrfach = Multikill, zählt stärker)", "valorant kill sound"),
        ("double_kill", "Sound für den zweiten Kill", "valorant double kill sound"),
        ("triple_kill", "Sound für den dritten Kill", "valorant triple kill sound"),
        ("quad_kill", "Sound für den vierten Kill", "valorant quadra kill sound"),
        ("ace", "Ace-Ansage", "valorant ace sound"),
        ("clutch", "Clutch-Ansage", "valorant clutch sound"),
        ("spike_plant", "Spike platziert", "valorant spike planted sound"),
        ("spike_defuse", "Spike entschärft", "valorant defuse sound"),
    ]},
    "cs2": {"label": "CS2", "events": [
        ("kill", "Kill-Bestätigung (mehrfach = Multikill, zählt stärker)", "cs2 kill sound"),
        ("headshot", "Headshot-Sound", "cs2 headshot sound"),
        ("bomb_plant", "Bombe gelegt", "cs2 bomb planted sound"),
        ("bomb_defuse", "Bombe entschärft", "cs2 defuse sound"),
    ]},
    "eafc": {"label": "EA FC / FIFA", "events": [
        ("goal", "Tor plus Kommentar", "ea fc goal sound"),
        ("whistle", "Schiedsrichterpfiff", "fc 26 referee whistle"),
        ("crowd_roar", "Publikumsjubel", "football crowd roar"),
    ]},
    "rocketleague": {"label": "Rocket League", "events": [
        ("goal", "Tor-Explosion", "rocket league goal explosion sound"),
        ("demolition", "Demolition", "rocket league demolition sound"),
        ("save", "Parade", "rocket league save sound"),
    ]},
    "horror": {"label": "Horror", "events": [
        ("stinger", "Schock-Stinger", "horror stinger sound"),
        ("scream", "Schrei", "scream sound effect"),
        ("jumpscare", "Jumpscare-Impact", "jumpscare sound"),
    ]},
    # Cross-game sounds: imported here, they're matched for every game profile
    # (the detector scans <data>/game_cues/common/ alongside the active game).
    "common": {"label": "Allgemein (alle Spiele)", "events": [
        ("airhorn", "Airhorn / Hype-Stoss", "airhorn sound"),
        ("hype", "Hype / Let's-go-Shout", "lets go hype sound"),
        ("laugh", "Lach-Explosion", "laugh sound effect"),
        ("applause", "Applaus / Jubel", "applause sound"),
        ("bruh", "Bruh / Fail-Moment", "bruh sound effect"),
        ("wow", "Wow / Schock-Reaktion", "wow sound effect"),
    ]},
}

# Folder scanned for cues that apply regardless of game - kept out of the
# per-game cue dir so it's reused everywhere (see detect_gameplay._cue_events).
COMMON_PACK = "common"


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9äöüß_-]", "", (name or "").lower().replace(" ", "_"))[:40] or "cue"


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

    Lets users paste a soundboard page link (the natural thing to copy)
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
    http_download(url, dest)


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
        if head.startswith((b"<", b"\xef\xbb\xbf<")):
            found = audio_url_from_html(
                dl.read_text(encoding="utf-8", errors="ignore"), url)
            if not found:
                raise ValueError(
                    "diese Seite enthält keinen direkten Audio-Link - kopiere "
                    "die Adresse des Download-Buttons und füge sie hier ein")
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
