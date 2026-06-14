"""On-screen text detection (OCR) — find viral moments the audio misses.

A kill-feed entry, a giant **VICTORY** / **DEFEAT** banner, **ELIMINATED**, a
**GOAL!** caption, **WINNER WINNER** — these are the exact instants a highlight
should open on, and they're printed right on the screen even when the audio is
ambiguous. This provider samples frames, reads them with whatever OCR engine is
installed, and matches the text against a per-game lexicon of viral markers.

Backends, best-accuracy first (all optional — none installed ⇒ this returns
nothing and the audio/cue path still works):

  PaddleOCR (PP-OCRv5) → EasyOCR (strong on noisy game overlays) → Tesseract.

Only ~1 frame every couple of seconds is read, downscaled, so even a long VOD
stays cheap. Pure helpers (keyword matching, frame-time sampling, de-dup) carry
no OCR dependency so they're unit-tested without a backend.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo

log = logging.getLogger("clipforge.ocr")


@dataclass
class OcrEvent:
    t: float          # timestamp in the source (s)
    label: str        # canonical marker, e.g. "victory", "kill", "goal"
    text: str         # the raw on-screen text that matched
    confidence: float  # 0..1


# Viral on-screen markers. Keys are canonical labels; values are the phrases
# (lowercased, whole-ish) that, found in OCR text, mean that event. Matched as
# normalized substrings, so "you have been eliminated" still hits "eliminated".
_GENERIC: dict[str, tuple[str, ...]] = {
    "victory": ("victory", "you win", "winner winner", "winner", "match won", "mission complete"),
    "defeat": ("defeat", "you lose", "you died", "you are dead", "game over", "wasted", "mission failed"),
    "eliminated": ("eliminated", "knocked", "knockout", "k.o", "ko", "finish him"),
    "kill": ("double kill", "triple kill", "multi kill", "ultra kill", "rampage",
             "killing spree", "first blood", "headshot"),
    "ace": ("ace", "team ace", "flawless"),
    "clutch": ("clutch", "1v5", "1v4", "1v3"),
    "record": ("new record", "personal best", "high score", "level up"),
}

# Per-profile extra markers, merged over the generic set.
_PROFILE_EXTRA: dict[str, dict[str, tuple[str, ...]]] = {
    "valorant": {
        "ace": ("ace", "team ace", "flawless", "thrifty", "flawless victory"),
        "clutch": ("clutch", "1v5", "1v4", "1v3", "1v2"),
        "spike": ("spike planted", "spike defused", "defusing"),
    },
    "cs2": {
        "bomb": ("bomb has been planted", "bomb planted", "bomb defused"),
        "win": ("counter-terrorists win", "terrorists win", "ct win"),
    },
    "eafc": {
        "goal": ("goal", "go!", "full time", "half time", "penalty", "red card"),
    },
    "rocketleague": {
        "goal": ("goal", "what a save", "save", "demolished", "epic save"),
    },
    "horror": {
        "defeat": ("you died", "game over", "you are dead", "wasted"),
    },
}
_ALIAS = {"auto": "generic", "cs": "cs2", "fifa": "eafc"}

_NORM_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Lowercase and collapse to alphanumerics + single spaces for matching."""
    return _WS_RE.sub(" ", _NORM_RE.sub(" ", (text or "").lower())).strip()


def lexicon(profile: str | None) -> dict[str, tuple[str, ...]]:
    """The marker lexicon for a game profile (generic ∪ profile extras)."""
    name = _ALIAS.get((profile or "generic").lower().replace(" ", ""),
                      (profile or "generic").lower().replace(" ", ""))
    merged: dict[str, tuple[str, ...]] = {k: v for k, v in _GENERIC.items()}
    for label, phrases in _PROFILE_EXTRA.get(name, {}).items():
        merged[label] = tuple(dict.fromkeys(merged.get(label, ()) + phrases))
    return merged


def match_keywords(text: str, profile: str | None) -> list[tuple[str, str]]:
    """Return [(label, matched_phrase)] for every viral marker in ``text``.

    Pure: no OCR dependency, so the lexicon is unit-testable. Longer phrases win
    over substrings of themselves (so "double kill" reports once, as a kill).
    """
    norm = _norm(text)
    if not norm:
        return []
    padded = f" {norm} "
    out: list[tuple[str, str]] = []
    for label, phrases in lexicon(profile).items():
        best: str | None = None
        for ph in phrases:
            p = _norm(ph)
            # word-boundary-ish match (padded spaces) avoids "ko" inside "took".
            if f" {p} " in padded and (best is None or len(p) > len(best)):
                best = p
        if best is not None:
            out.append((label, best))
    return out


def sample_frame_times(duration: float, *, every: float = 2.0,
                       max_frames: int = 400) -> list[float]:
    """Evenly-spaced timestamps to OCR, capped so long VODs stay bounded."""
    if duration <= 0:
        return []
    step = max(every, duration / max_frames)
    times: list[float] = []
    t = step / 2.0
    while t < duration and len(times) < max_frames:
        times.append(round(t, 3))
        t += step
    return times


def dedupe_events(events: list[OcrEvent], *, min_gap: float = 4.0) -> list[OcrEvent]:
    """Collapse repeats of the same label that persist across sampled frames
    (a banner shows for several seconds → one event at its first sighting)."""
    events = sorted(events, key=lambda e: (e.t, e.label))
    kept: list[OcrEvent] = []
    last: dict[str, float] = {}
    for e in events:
        if e.t - last.get(e.label, -1e9) >= min_gap:
            kept.append(e)
        last[e.label] = e.t
    kept.sort(key=lambda e: e.t)
    return kept


# --------------------------------------------------------------------------- #
# OCR backends (lazy, optional)
# --------------------------------------------------------------------------- #
_reader = None  # cached backend instance


def _get_reader(engine: str):
    global _reader
    if _reader is not None:
        return _reader
    s = get_settings()
    gpu = s.device == "cuda"
    if engine == "paddleocr":
        from paddleocr import PaddleOCR

        _reader = ("paddleocr", PaddleOCR(use_angle_cls=False, lang="en",
                                          show_log=False, use_gpu=gpu))
    elif engine == "easyocr":
        import easyocr

        _reader = ("easyocr", easyocr.Reader(["en"], gpu=gpu, verbose=False))
    elif engine == "tesseract":
        import pytesseract  # noqa: F401

        _reader = ("tesseract", None)
    else:
        _reader = ("", None)
    return _reader


def _ocr_image(path: str, engine: str) -> str:
    """Read all text from one image with the active backend → one string."""
    kind, reader = _get_reader(engine)
    try:
        if kind == "paddleocr":
            res = reader.ocr(path, cls=False) or []
            lines = []
            for page in res:
                for entry in (page or []):
                    txt = entry[1][0] if entry and len(entry) > 1 else ""
                    if txt:
                        lines.append(txt)
            return " ".join(lines)
        if kind == "easyocr":
            return " ".join(reader.readtext(path, detail=0) or [])
        if kind == "tesseract":
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(path))
    except Exception as e:  # one bad frame mustn't sink detection
        log.warning("ocr read failed for %s: %s", path, e)
    return ""


def find_text_events(src_path: str, info: MediaInfo,
                     settings, *, every: float = 2.0) -> list[OcrEvent]:
    """Sample frames and return viral on-screen-text events. [] if OCR is off
    or the source has no video."""
    s = get_settings()
    if not s.has_ocr or not info.has_video or info.duration <= 0:
        return []
    times = sample_frame_times(info.duration, every=every)
    if not times:
        return []
    engine = s.ocr_engine
    profile = getattr(settings, "game_profile", "generic")
    events: list[OcrEvent] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = Path(tmp)
            for i, t in enumerate(times):
                frame = tmpd / f"f{i}.png"
                try:
                    # Downscale to 720p wide — plenty for banner/feed text, fast.
                    ffmpeg.run(["-ss", f"{t:.3f}", "-i", src_path, "-frames:v", "1",
                                "-vf", "scale=1280:-2", str(frame)], timeout=30)
                except Exception as e:
                    log.warning("ocr frame grab failed at %.1fs: %s", t, e)
                    continue
                text = _ocr_image(str(frame), engine)
                for label, matched in match_keywords(text, profile):
                    events.append(OcrEvent(t=round(t, 3), label=label,
                                           text=matched, confidence=0.8))
    except Exception as e:
        log.warning("ocr detection aborted: %s", e)
        return []
    events = dedupe_events(events)
    log.info("ocr: %d on-screen events via %s", len(events), engine)
    return events
