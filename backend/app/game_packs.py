"""Cue packs — the canonical "what sounds and graphics to add" per game.

ClipForge matches reference game sounds (app/providers/detect_cues.py) and
on-screen graphics (app/providers/detect_visual_cues.py) to pinpoint events. We
can't ship the copyrighted assets, but we ship the *structure*: which events each
supported game has, a search hint for finding the asset, and a status check for
which cues you've added. Drop audio files named ``<event>.<ext>`` into
``<data>/game_cues/<game>/`` and reference images (cropped from a screenshot)
into ``<data>/game_cues/<game>/visual/``.
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
from .providers.detect_visual_cues import IMG_EXTS

# Each pack: "events" are audio cues (reference sounds), "visual_events" are
# on-screen graphics (a cropped reference image — crop just the stable UI
# element from a screenshot, not the whole frame).
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
    ], "visual_events": [
        ("kill_banner", "Kill banner graphic (crop from a screenshot)", "valorant kill banner png"),
        ("ace_banner", "ACE splash graphic", "valorant ace banner png"),
        ("clutch_banner", "CLUTCH splash graphic", "valorant clutch banner png"),
    ]},
    "cs2": {"label": "CS2", "events": [
        ("kill", "Kill confirm (repeats = multi-kill, scores higher)", "cs2 kill sound"),
        ("headshot", "Headshot ‘ding’", "cs2 headshot sound"),
        ("bomb_plant", "Bomb planted", "cs2 bomb planted sound"),
        ("bomb_defuse", "Bomb defused", "cs2 defuse sound"),
    ], "visual_events": [
        ("kill_icon", "Kill-feed skull icon (crop from a screenshot)", "cs2 killfeed skull icon png"),
        ("mvp_banner", "Round MVP banner", "cs2 mvp banner png"),
    ]},
    "eafc": {"label": "EA FC / FIFA", "events": [
        ("goal", "Goal net + commentary", "ea fc goal sound"),
        ("whistle", "Referee whistle", "fc 26 referee whistle"),
        ("crowd_roar", "Crowd roar", "football crowd roar"),
    ], "visual_events": [
        ("goal_banner", "GOAL overlay graphic (crop from a screenshot)", "ea fc goal overlay png"),
        ("replay_wipe", "Replay transition graphic", "ea fc replay wipe png"),
    ]},
    "rocketleague": {"label": "Rocket League", "events": [
        ("goal", "Goal explosion", "rocket league goal explosion sound"),
        ("demolition", "Demolition", "rocket league demolition sound"),
        ("save", "Save", "rocket league save sound"),
    ], "visual_events": [
        ("goal_banner", "GOAL! banner graphic (crop from a screenshot)", "rocket league goal banner png"),
        ("mvp_banner", "MVP banner", "rocket league mvp png"),
    ]},
    "horror": {"label": "Horror", "events": [
        ("stinger", "Musical sting", "horror stinger sound"),
        ("scream", "Scream", "scream sound effect"),
        ("jumpscare", "Jump-scare hit", "jumpscare sound"),
    ], "visual_events": [
        ("death_screen", "Death / game-over screen (crop from a screenshot)", "you died screen png"),
    ]},
}


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", (name or "").lower().replace(" ", "_"))[:40] or "cue"


def _dir(game: str):
    return get_settings().data_dir / "game_cues" / _safe(game)


# Magic bytes of the image formats we accept as visual cues.
_IMG_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"BM", b"GIF8")


def is_image_file(path: str | Path) -> bool:
    try:
        head = Path(path).open("rb").read(16)
    except OSError:
        return False
    if head.startswith(_IMG_MAGIC):
        return True
    return head[:4] == b"RIFF" and head[8:12] == b"WEBP"


def install_cue(game: str, event: str, src_path: str) -> None:
    """Install an audio or image cue (auto-detected from the file's content).

    Audio is normalised to a 16 kHz mono <game>/<event>.wav; an image becomes a
    visual cue at <game>/visual/<event>.png.
    """
    if is_image_file(src_path):
        dest = _dir(game) / "visual" / f"{_safe(event)}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.run(["-i", str(src_path), "-frames:v", "1", str(dest)])
        return
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
    removed = False
    for d, exts in ((_dir(game), CUE_EXTS), (_dir(game) / "visual", IMG_EXTS)):
        if d.is_dir():
            for p in d.glob(f"{_safe(event)}.*"):
                if p.suffix.lower() in exts:
                    p.unlink(missing_ok=True)
                    removed = True
    return removed


def _present(d: Path, exts: set[str]) -> set[str]:
    return {p.stem.lower() for p in d.glob("*") if p.suffix.lower() in exts} if d.is_dir() else set()


def pack_status() -> dict:
    """Per game: which events have a cue file present."""
    out: dict[str, dict] = {}
    for game, pack in PACKS.items():
        d = _dir(game)
        audio = _present(d, CUE_EXTS)
        visual = _present(d / "visual", IMG_EXTS)
        events = [{"name": n, "desc": desc, "hint": hint, "kind": "audio",
                   "configured": n.lower() in audio}
                  for (n, desc, hint) in pack["events"]]
        events += [{"name": n, "desc": desc, "hint": hint, "kind": "visual",
                    "configured": n.lower() in visual}
                   for (n, desc, hint) in pack.get("visual_events", [])]
        out[game] = {
            "label": pack["label"],
            "events": events,
            "configured": sum(1 for e in events if e["configured"]),
            "total": len(events),
        }
    return out
