"""On-screen text detection (OCR) — find viral moments the audio misses.

A kill-feed entry, a giant **VICTORY** / **DEFEAT** banner, **ELIMINATED**, a
**GOAL!** caption, **WINNER WINNER** — these are the exact instants a highlight
should open on, and they're printed right on the screen even when the audio is
ambiguous. This provider samples frames, reads them with whatever OCR engine is
installed, and matches the text against a per-game lexicon of viral markers.

Backends, best-accuracy first (all optional — none installed ⇒ this returns
nothing and the audio/cue path still works):

  PaddleOCR (PP-OCRv6 -> PP-OCRv5) → EasyOCR (strong on noisy game overlays) → Tesseract.

Only ~1 frame every couple of seconds is read, downscaled, so even a long VOD
stays cheap. Pure helpers (keyword matching, frame-time sampling, de-dup) carry
no OCR dependency so they're unit-tested without a backend.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from .. import visual_cues

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
    "victory": ("victory", "you win", "winner winner", "winner", "match won",
                "mission complete", "sieg", "gewonnen", "runde gewonnen",
                "spiel gewonnen", "auftrag abgeschlossen"),
    "defeat": ("defeat", "you lose", "you died", "you are dead", "game over",
               "wasted", "mission failed", "niederlage", "verloren",
               "du bist tot", "mission fehlgeschlagen"),
    "eliminated": ("eliminated", "knocked", "knockout", "k.o", "ko",
                   "finish him", "eliminiert", "ausgeschaltet",
                   "niedergeschlagen", "besiegt"),
    "kill": ("double kill", "triple kill", "multi kill", "ultra kill", "rampage",
             "killing spree", "first blood", "headshot", "doppelkill",
             "dreifachkill", "mehrfachkill", "kopfschuss"),
    "ace": ("ace", "team ace", "flawless"),
    "clutch": ("clutch", "1v5", "1v4", "1v3"),
    "record": ("new record", "personal best", "high score", "level up",
               "neuer rekord", "persoenlicher rekord", "level aufstieg"),
    "menu": ("settings", "inventory", "main menu", "menu", "lobby",
             "store", "shop", "collection", "loadout", "agent select",
             "matchmaking", "einstellungen", "inventar", "hauptmenu",
             "hauptmenue", "hauptmenü", "menue", "menü", "laden",
             "sammlung", "ausruestung", "ausrüstung", "agenten"),
}

# Per-profile extra markers, merged over the generic set.
_PROFILE_EXTRA: dict[str, dict[str, tuple[str, ...]]] = {
    "valorant": {
        "ace": ("ace", "team ace", "flawless", "thrifty", "flawless victory",
                "team ass"),
        "clutch": ("clutch", "1v5", "1v4", "1v3", "1v2", "letzter spieler",
                   "letzte spielerin", "last player standing"),
        "kill": ("headshot", "kopfschuss", "abgeschossen", "eliminiert",
                 "enemy killed", "gegner ausgeschaltet", "gegner ubrig",
                 "gegner übrig"),
        "spike": ("spike planted", "spike defused", "defusing",
                  "spike platziert", "spike entscharft", "spike entschärft"),
        "round_win": ("round won", "runde gewonnen", "victory"),
    },
    "cs2": {
        "bomb": ("bomb has been planted", "bomb planted", "bomb defused",
                 "bombe wurde gelegt", "bombe gelegt", "bombe platziert",
                 "bombe entscharft", "bombe entschaerft"),
        "win": ("counter-terrorists win", "terrorists win", "ct win",
                "terroristen gewinnen", "antiterroreinheit gewinnt",
                "counter terrorists gewinnen", "runde gewonnen", "sieg"),
    },
    "eafc": {
        "goal": ("goal", "go!", "full time", "half time", "penalty", "red card",
                 "tor", "halbzeit", "abpfiff", "elfmeter", "rote karte",
                 "gelbe karte", "freistoss", "freistos"),
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
    folded = unicodedata.normalize("NFKD", text or "")
    folded = folded.encode("ascii", "ignore").decode("ascii")
    return _WS_RE.sub(" ", _NORM_RE.sub(" ", folded.lower())).strip()


def lexicon(profile: str | None) -> dict[str, tuple[str, ...]]:
    """The marker lexicon for a game profile (generic ∪ profile extras)."""
    name = _ALIAS.get((profile or "generic").lower().replace(" ", ""),
                      (profile or "generic").lower().replace(" ", ""))
    merged: dict[str, tuple[str, ...]] = {k: v for k, v in _GENERIC.items()}
    for label, phrases in _PROFILE_EXTRA.get(name, {}).items():
        merged[label] = tuple(dict.fromkeys(merged.get(label, ()) + phrases))
    for label, phrases in visual_cues.lexicon_extra(name).items():
        merged[label] = tuple(dict.fromkeys(merged.get(label, ()) + phrases))
    return merged


def _fuzzy_contains(phrase: str, words: list[str], *, threshold: int) -> bool:
    """True if any n-gram window of ``words`` ~matches ``phrase`` (OCR-typo safe).

    No-op (returns False) when rapidfuzz isn't installed, so exact matching still
    works. Only used as a fallback after exact matching misses, to avoid drifting
    the match rate up on clean text.
    """
    try:
        from rapidfuzz import fuzz
    except Exception:
        return False
    ptoks = phrase.split()
    n = len(ptoks)
    if n == 0 or len(words) < n:
        return False
    # Single short tokens fuzz-match too loosely ("ace" ≈ "are"); require length.
    if n == 1 and len(phrase) < 5:
        return False
    for i in range(len(words) - n + 1):
        window = " ".join(words[i:i + n])
        if fuzz.ratio(window, phrase) >= threshold:
            return True
    return False


def match_keywords(text: str, profile: str | None, *,
                   fuzzy: bool = True, threshold: int = 86) -> list[tuple[str, str]]:
    """Return [(label, matched_phrase)] for every viral marker in ``text``.

    Pure: no OCR dependency, so the lexicon is unit-testable. Longer phrases win
    over substrings of themselves (so "double kill" reports once, as a kill).
    When ``fuzzy`` and rapidfuzz is installed, a phrase the exact pass missed is
    retried with edit-distance matching, so stylized/garbled game text
    ("VICT0RY", "ELiMiNATED", "HEADSHOTI") still resolves to its marker.
    """
    norm = _norm(text)
    if not norm:
        return []
    padded = f" {norm} "
    toks = norm.split()
    out: list[tuple[str, str]] = []
    for label, phrases in lexicon(profile).items():
        best: str | None = None
        for ph in phrases:
            p = _norm(ph)
            # word-boundary-ish match (padded spaces) avoids "ko" inside "took".
            exact = f" {p} " in padded
            if not exact and fuzzy:
                exact = _fuzzy_contains(p, toks, threshold=threshold)
            if exact and (best is None or len(p) > len(best)):
                best = p
        if best is not None and not visual_cues.is_false_positive(profile, label, text):
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


def focused_frame_times(duration: float, focus_times: list[float] | None, *,
                        every: float = 2.0, max_frames: int = 260) -> list[float]:
    """OCR around likely moments, with a light safety sweep across the VOD."""
    if not focus_times:
        return sample_frame_times(duration, every=every, max_frames=max_frames)
    out: list[float] = []
    seen: set[int] = set()
    for t in sorted(float(x) for x in focus_times if 0.0 <= float(x) <= duration):
        for off in (-1.0, 0.0, 1.5, 3.0):
            tt = min(max(t + off, 0.25), max(duration - 0.25, 0.25))
            bucket = int(round(tt / 0.75))
            if bucket not in seen:
                seen.add(bucket)
                out.append(round(tt, 3))
    for t in sample_frame_times(duration, every=max(every * 5.0, 12.0),
                                max_frames=max_frames // 4):
        bucket = int(round(t / 0.75))
        if bucket not in seen:
            seen.add(bucket)
            out.append(t)
    out.sort()
    if len(out) > max_frames:
        step = len(out) / max_frames
        out = [out[int(i * step)] for i in range(max_frames)]
    return out


def scene_frame_times(src_path: str, duration: float, *,
                      max_frames: int = 120) -> list[float]:
    """Frames just after hard cuts, so OCR reads stable new shots."""
    if duration <= 0 or not get_settings().has_scenedetect:
        return []
    try:
        from . import scenes

        cuts = scenes.scene_cuts(src_path, 0.0, duration)
    except Exception as e:
        log.debug("scene OCR keyframes unavailable: %s", e)
        return []
    times = [min(max(c + 0.15, 0.25), max(duration - 0.25, 0.25))
             for c in cuts if 0.0 <= c <= duration]
    if len(times) > max_frames:
        step = len(times) / max_frames
        times = [times[int(i * step)] for i in range(max_frames)]
    return [round(t, 3) for t in times]


def dedupe_events(events: list[OcrEvent], *, min_gap: float = 4.0) -> list[OcrEvent]:
    """Collapse repeats of the same label that persist across sampled frames
    (a banner shows for several seconds → one event at its first sighting)."""
    # Strongest-first within a tie so a high-confidence ROI crop survives over a
    # weaker full-frame hit at the same instant; then by time for the gap walk.
    events = sorted(events, key=lambda e: (e.t, e.label, -e.confidence))
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


def _make_paddle(gpu: bool):
    """Construct a PaddleOCR reader across the 2.x and 3.x APIs.

    PaddleOCR 3.0 (2025) is a non-backwards-compatible rewrite: it dropped the
    ``show_log`` / ``use_angle_cls`` / ``use_gpu`` constructor kwargs (now
    ``use_textline_orientation`` and ``device``) and passing the old ones raises.
    We try the modern signature first and fall back to the legacy one, so a clip
    farm on either major version reads on-screen text instead of silently
    returning nothing.
    """
    # PaddlePaddle 3.3.x can crash in oneDNN/PIR CPU inference on Windows.
    # Disabling the default MKLDNN path keeps OCR usable while preserving the
    # GPU/transformers attempts where the local runtime supports them.
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    from paddleocr import PaddleOCR

    device = "gpu" if gpu else "cpu"
    # PP-OCRv6 on PaddleOCR 3.2+ with the transformers engine. Older installs
    # fall through to the existing PP-OCRv5/legacy constructor cascade.
    try:
        return PaddleOCR(
            text_detection_model_name="PP-OCRv6_medium_det",
            text_recognition_model_name="PP-OCRv6_medium_rec",
            engine="transformers",
            lang="en",
            use_textline_orientation=False,
            device=device,
        )
    except Exception as e:
        log.info("PaddleOCR v6 unavailable, falling back (%s)", e)
    for kwargs in (
        # 3.x: angle classifier off (we only read horizontal banners), pick device.
        {"lang": "en", "use_textline_orientation": False, "device": device},
        {"lang": "en", "use_textline_orientation": False},
        # 2.x: legacy flags.
        {"lang": "en", "use_angle_cls": False, "show_log": False, "use_gpu": gpu},
        {"lang": "en"},
    ):
        try:
            return PaddleOCR(**kwargs)
        except Exception as e:
            log.info("PaddleOCR constructor fallback failed (%s): %s", kwargs, e)
    return PaddleOCR(lang="en")  # last resort — let a real error surface


def _make_easyocr(gpu: bool):
    import easyocr

    return easyocr.Reader(["en"], gpu=gpu, verbose=False)


def _get_reader(engine: str):
    global _reader
    if _reader is not None:
        return _reader
    s = get_settings()
    gpu = s.device == "cuda"
    attempts = []
    if engine == "paddleocr":
        attempts.append(("paddleocr", lambda: _make_paddle(gpu)))
        attempts.append(("easyocr", lambda: _make_easyocr(gpu)))
    elif engine == "easyocr":
        attempts.append(("easyocr", lambda: _make_easyocr(gpu)))
    elif engine == "tesseract":
        attempts.append(("tesseract", lambda: None))
    for kind, make in attempts:
        try:
            _reader = (kind, make())
            return _reader
        except Exception as e:
            log.warning("%s OCR unavailable, trying fallback if present: %s", kind, e)
    _reader = ("", None)
    return _reader


def _paddle_read(reader, path: str) -> tuple[str, float]:
    """Read text + a mean recognition confidence from one image.

    Parses both PaddleOCR 2.x (``[[ [box, (text, conf)], ... ]]``) and 3.x
    (``OCRResult`` with ``rec_texts``/``rec_scores``). Confidence is the mean of
    the recognized lines' scores, or 0.0 when the backend doesn't report any.
    """
    # 3.x prefers .predict(); .ocr() still exists but warns. Use whichever runs.
    res = None
    try:
        res = reader.predict(path)
    except (AttributeError, TypeError):
        try:
            res = reader.ocr(path, cls=False)
        except TypeError:
            res = reader.ocr(path)  # 3.x .ocr() dropped the cls kwarg
    if not res:
        return "", 0.0
    lines: list[str] = []
    scores: list[float] = []
    for page in res:
        # 3.x: dict-like result with parallel rec_texts / rec_scores lists.
        rec_texts = None
        rec_scores = None
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts")
            rec_scores = page.get("rec_scores")
        else:
            rec_texts = getattr(page, "rec_texts", None)
            rec_scores = getattr(page, "rec_scores", None)
            if rec_texts is None:
                try:
                    rec_texts = page["rec_texts"]
                    rec_scores = page["rec_scores"]
                except (TypeError, KeyError, IndexError):
                    rec_texts = None
        if rec_texts:
            for i, t in enumerate(rec_texts):
                if not t:
                    continue
                lines.append(t)
                try:
                    scores.append(float(rec_scores[i]))
                except (TypeError, IndexError, ValueError):
                    pass
            continue
        # 2.x: list of [box, (text, conf)] entries.
        for entry in (page or []):
            try:
                txt = entry[1][0]
                conf = entry[1][1]
            except (TypeError, IndexError, KeyError):
                txt, conf = "", None
            if txt:
                lines.append(txt)
                try:
                    scores.append(float(conf))
                except (TypeError, ValueError):
                    pass
    mean = sum(scores) / len(scores) if scores else 0.0
    return " ".join(lines), max(0.0, min(1.0, mean))


def _paddle_text(reader, path: str) -> str:
    return _paddle_read(reader, path)[0]


def _ocr_image(path: str, engine: str) -> str:
    """Read all text from one image with the active backend → one string."""
    return _ocr_image_conf(path, engine)[0]


def _ocr_image_conf(path: str, engine: str) -> tuple[str, float]:
    """Read text and a 0..1 recognition confidence (0.0 when unknown)."""
    kind, reader = _get_reader(engine)
    try:
        if kind == "paddleocr":
            return _paddle_read(reader, path)
        if kind == "easyocr":
            return _easyocr_read(reader, path)
        if kind == "tesseract":
            import pytesseract
            from PIL import Image

            return pytesseract.image_to_string(Image.open(path)), 0.0
    except Exception as e:  # one bad frame mustn't sink detection
        log.warning("ocr read failed for %s: %s", path, e)
    return "", 0.0


def _easyocr_text(reader, path: str) -> str:
    return _easyocr_read(reader, path)[0]


def _easyocr_read(reader, path: str) -> tuple[str, float]:
    """Read EasyOCR output + mean confidence without assuming one return shape."""
    try:
        rows = reader.readtext(path, detail=1, paragraph=False) or []
        lines: list[str] = []
        scores: list[float] = []
        for row in rows:
            txt = ""
            if isinstance(row, str):
                txt = row
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                txt = row[1][0] if isinstance(row[1], (list, tuple)) else row[1]
                if len(row) >= 3:
                    try:
                        scores.append(float(row[2]))
                    except (TypeError, ValueError):
                        pass
            if txt:
                lines.append(str(txt))
        if lines:
            mean = sum(scores) / len(scores) if scores else 0.0
            return " ".join(lines), max(0.0, min(1.0, mean))
    except Exception as e:
        log.debug("easyocr detail read failed for %s: %s", path, e)
    try:
        rows = reader.readtext(path, detail=0, paragraph=False) or []
    except TypeError:
        rows = reader.readtext(path, detail=0) or []
    return " ".join(str(x) for x in rows if x), 0.0


def _ocr_frame_images(frame: Path, tmpd: Path, idx: int,
                      profile: str | None,
                      extra_regions: list | tuple | None = None
                      ) -> list[tuple[str, Path]]:
    """Return full frame plus game-specific ROIs that carry useful text."""
    name = _ALIAS.get((profile or "generic").lower().replace(" ", ""),
                      (profile or "generic").lower().replace(" ", ""))
    # Skip the PIL decode entirely when this profile can't produce any ROI crop
    # (e.g. "generic" with no saved/manual regions) — on a long VOD that avoids
    # hundreds of full-frame decodes whose result is just the frame path again.
    saved_regions = visual_cues.regions_extra(name)
    if name not in {"valorant", "cs2"} and not saved_regions and not extra_regions:
        return [("full", frame)]
    try:
        from PIL import Image, ImageOps

        im = Image.open(frame).convert("RGB")
        w, h = im.size
        specs: list[tuple[str, tuple[int, int, int, int]]] = []
        if name in {"valorant", "cs2"}:
            specs.extend([
                ("killfeed", (int(w * 0.55), 0, w, int(h * 0.36))),
                ("top_banner", (int(w * 0.20), 0, int(w * 0.80), int(h * 0.28))),
                ("center_banner", (int(w * 0.16), int(h * 0.22),
                                   int(w * 0.84), int(h * 0.72))),
            ])
        for label, regions in saved_regions.items():
            for r_i, region in enumerate(regions):
                try:
                    x0 = int(float(region.get("x", 0.0)) * w)
                    y0 = int(float(region.get("y", 0.0)) * h)
                    x1 = int((float(region.get("x", 0.0)) + float(region.get("w", 1.0))) * w)
                    y1 = int((float(region.get("y", 0.0)) + float(region.get("h", 1.0))) * h)
                except (TypeError, ValueError):
                    continue
                x0 = max(0, min(w - 1, x0))
                y0 = max(0, min(h - 1, y0))
                x1 = max(x0 + 1, min(w, x1))
                y1 = max(y0 + 1, min(h, y1))
                specs.append((f"saved_{label}_{r_i}", (x0, y0, x1, y1)))
        for r_i, region in enumerate(extra_regions or []):
            try:
                if hasattr(region, "model_dump"):
                    region = region.model_dump()
                x0 = int(float(region.get("x", 0.0)) * w)
                y0 = int(float(region.get("y", 0.0)) * h)
                x1 = int((float(region.get("x", 0.0)) + float(region.get("w", 1.0))) * w)
                y1 = int((float(region.get("y", 0.0)) + float(region.get("h", 1.0))) * h)
            except (AttributeError, TypeError, ValueError):
                continue
            x0 = max(0, min(w - 1, x0))
            y0 = max(0, min(h - 1, y0))
            x1 = max(x0 + 1, min(w, x1))
            y1 = max(y0 + 1, min(h, y1))
            specs.append((f"manual_roi_{r_i}", (x0, y0, x1, y1)))
        out: list[tuple[str, Path]] = []
        if idx % 5 == 0:
            out.append(("full", frame))
        if not specs:
            return out or [("full", frame)]
        for roi, box in specs:
            crop = im.crop(box)
            if crop.width < 32 or crop.height < 24:
                continue
            scale = 2 if crop.width < 900 else 1
            if scale > 1:
                crop = crop.resize((crop.width * scale, crop.height * scale))
            crop = ImageOps.autocontrast(crop)
            p = tmpd / f"f{idx}_{roi}.png"
            crop.save(p)
            out.append((roi, p))
        return out
    except Exception as e:
        log.warning("ocr ROI crop failed for %s: %s", frame, e)
        return [("full", frame)]


def _ocr_evidence(matched: str, raw_text: str) -> str:
    text = " ".join((raw_text or "").split())
    if not text:
        return matched
    if len(text) > 140:
        text = text[:137] + "..."
    return f"{matched} | {text}"


def _manual_matches(text: str, cues: list[str] | tuple[str, ...] | None
                    ) -> list[tuple[str, str]]:
    norm = _norm(text)
    if not norm:
        return []
    padded = f" {norm} "
    out: list[tuple[str, str]] = []
    for cue in cues or ():
        p = _norm(cue)
        if p and f" {p} " in padded:
            out.append(("manual_visual", p))
    return out


def find_text_events(src_path: str, info: MediaInfo,
                     settings, *, every: float = 2.0,
                     focus_times: list[float] | None = None) -> list[OcrEvent]:
    """Sample frames and return viral on-screen-text events. [] if OCR is off
    or the source has no video."""
    s = get_settings()
    if not s.has_ocr or not info.has_video or info.duration <= 0:
        return []
    cfg = getattr(settings, "game_config", None)
    scene_times = scene_frame_times(src_path, info.duration)
    focused = list(focus_times or []) + scene_times
    times = focused_frame_times(info.duration, focused or None, every=every)
    if not times:
        return []
    engine = s.ocr_engine
    profile = getattr(settings, "game_profile", "generic")
    extra_regions = getattr(cfg, "visual_rois", []) if cfg is not None else []
    manual_cues = getattr(cfg, "visual_text_cues", []) if cfg is not None else []
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
                roi_texts = []
                for roi, img in _ocr_frame_images(
                        frame, tmpd, i, profile, extra_regions=extra_regions):
                    text, rconf = _ocr_image_conf(str(img), engine)
                    if text:
                        roi_texts.append((roi, text, rconf))
                for roi, text, rconf in roi_texts:
                    # Prefer the engine's real recognition confidence; fall back
                    # to a ROI prior only when the backend doesn't report one
                    # (e.g. tesseract). ROI crops read a tight banner, so they
                    # keep a small reliability edge over a full-frame sweep.
                    if rconf > 0.0:
                        conf = round(min(1.0, rconf * (1.0 if roi != "full" else 0.97)), 4)
                    else:
                        conf = 0.9 if roi != "full" else 0.8
                    matches = match_keywords(text, profile)
                    matches.extend(_manual_matches(text, manual_cues))
                    for label, matched in matches:
                        events.append(OcrEvent(t=round(t, 3), label=label,
                                               text=_ocr_evidence(matched, text),
                                               confidence=conf))
    except Exception as e:
        # Re-raise so the caller (_find_ocr_events) records a UI warning instead
        # of silently degrading to zero on-screen events.
        log.warning("ocr detection aborted: %s", e)
        raise
    events = dedupe_events(events)
    log.info("ocr: %d on-screen events via %s", len(events), engine)
    return events
