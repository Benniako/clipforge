"""Cue-pack management endpoints — add/remove reference game sounds from the UI.

Lets the user paste a sound URL (e.g. a MyInstants link) or upload a file and
have it installed as a matching cue — no command line. All local.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import game_packs, visual_cues
from ..config import get_settings
from ..media import ffmpeg
from ..providers import detect_cues, detect_ocr

router = APIRouter(prefix="/api/cues", tags=["cues"])


@router.get("")
def list_cues() -> dict:
    out = game_packs.pack_status()
    visual = visual_cues.list_visual_cues()
    for game, cues in visual.items():
        out.setdefault(game, {"label": game.title(), "events": [], "configured": 0, "total": 0})
        out[game]["visual"] = cues
    return out


@router.get("/visual")
def list_visual() -> dict:
    return visual_cues.list_visual_cues()


@router.get("/visual-meta")
def list_visual_meta() -> dict:
    return visual_cues.list_visual_meta()


@router.post("/visual/{game}/{label}")
def add_visual(game: str, label: str, phrase: str = Form(...)) -> dict:
    try:
        visual_cues.add_visual_cue(game, label, phrase)
    except Exception as e:
        raise HTTPException(400, f"could not save visual cue: {e}")
    return visual_cues.list_visual_cues()


@router.post("/visual/{game}/{label}/region")
def add_visual_region(
    game: str,
    label: str,
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
    name: str | None = Form(None),
    phrase: str | None = Form(None),
) -> dict:
    try:
        if phrase and phrase.strip():
            visual_cues.add_visual_cue(game, label, phrase)
        visual_cues.add_visual_region(game, label, {"x": x, "y": y, "w": w, "h": h}, name)
    except Exception as e:
        raise HTTPException(400, f"could not save visual cue region: {e}")
    return visual_cues.list_visual_meta()


@router.post("/visual/{game}/{label}/false")
def add_false_visual(game: str, label: str, phrase: str = Form(...)) -> dict:
    try:
        visual_cues.add_false_visual_cue(game, label, phrase)
    except Exception as e:
        raise HTTPException(400, f"could not save false recognition: {e}")
    return visual_cues.list_visual_meta()


@router.delete("/visual/{game}/{label}")
def delete_visual(game: str, label: str, phrase: str | None = None) -> dict:
    visual_cues.remove_visual_cue(game, label, phrase)
    return visual_cues.list_visual_cues()


def _tmp_upload(file: UploadFile, suffix: str | None = None) -> Path:
    ext = suffix or Path(file.filename or "upload").suffix or ".bin"
    fd, tmp_name = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    return Path(tmp_name)


@router.post("/lab/ocr")
async def test_ocr_box(
    game: str = Form("auto"),
    label: str | None = Form(None),
    x: float = Form(0.0),
    y: float = Form(0.0),
    w: float = Form(1.0),
    h: float = Form(1.0),
    save: bool = Form(False),
    file: UploadFile = File(...),
) -> dict:
    """OCR an uploaded screenshot crop and optionally save the text as a visual cue.

    Box coordinates are normalized 0..1 relative to the image.
    """
    settings = get_settings()
    if not settings.has_ocr:
        raise HTTPException(400, "OCR is not available in this setup")
    tmp = _tmp_upload(file)
    crop_path: Path | None = None
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        from PIL import Image, ImageOps

        im = Image.open(tmp).convert("RGB")
        iw, ih = im.size
        x0 = max(0, min(iw - 1, int(x * iw)))
        y0 = max(0, min(ih - 1, int(y * ih)))
        x1 = max(x0 + 1, min(iw, int((x + w) * iw)))
        y1 = max(y0 + 1, min(ih, int((y + h) * ih)))
        crop = ImageOps.autocontrast(im.crop((x0, y0, x1, y1)))
        crop_path = tmp.with_suffix(".crop.png")
        crop.save(crop_path)
        text = detect_ocr._ocr_image(str(crop_path), settings.ocr_engine)
        matches = [
            {"label": lab, "phrase": phrase}
            for lab, phrase in detect_ocr.match_keywords(text, game)
        ]
        saved = False
        if save:
            cue_label = label or (matches[0]["label"] if matches else "visual_cue")
            visual_cues.add_visual_cue(game, cue_label, text)
            visual_cues.add_visual_region(
                game,
                cue_label,
                {"x": x0 / iw, "y": y0 / ih, "w": (x1 - x0) / iw, "h": (y1 - y0) / ih},
                cue_label,
            )
            saved = True
        return {
            "text": text,
            "matches": matches,
            "box": {"x": x0 / iw, "y": y0 / ih, "w": (x1 - x0) / iw, "h": (y1 - y0) / ih},
            "saved": saved,
            "visual": visual_cues.list_visual_cues(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"OCR test failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)
        if crop_path is not None:
            crop_path.unlink(missing_ok=True)


@router.post("/lab/audio")
async def test_audio_cues(
    game: str = Form("auto"),
    file: UploadFile = File(...),
) -> dict:
    """Match installed custom sound cues against an uploaded audio/video sample."""
    tmp = _tmp_upload(file)
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        base = get_settings().data_dir / "game_cues"
        games = {game_packs.COMMON_PACK, game_packs._safe(game)}
        events = []
        for sub in games:
            for ev in detect_cues.find_events(str(tmp), base / sub):
                events.append({
                    "t": ev.t,
                    "label": ev.label,
                    "similarity": ev.similarity,
                    "source": sub,
                })
        events.sort(key=lambda e: e["t"])
        return {"events": events, "count": len(events)}
    except Exception as e:
        raise HTTPException(400, f"audio cue test failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/lab/audio-window")
async def test_audio_window(
    game: str = Form("auto"),
    label: str | None = Form(None),
    start: float = Form(0.0),
    duration: float = Form(2.5),
    save: bool = Form(False),
    file: UploadFile = File(...),
) -> dict:
    """Extract a short audio window from an uploaded video/audio file.

    This is for creating clean reference cues from the imported gameplay clip
    without saving the entire match as a cue.
    """
    tmp = _tmp_upload(file)
    window = tmp.with_suffix(".cue.wav")
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        start = max(0.0, float(start))
        duration = max(0.25, min(8.0, float(duration)))
        ffmpeg.run([
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(tmp),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(window),
        ], timeout=120)
        base = get_settings().data_dir / "game_cues"
        games = {game_packs.COMMON_PACK, game_packs._safe(game)}
        events = []
        for sub in games:
            for ev in detect_cues.find_events(str(window), base / sub):
                events.append({
                    "t": round(start + ev.t, 3),
                    "label": ev.label,
                    "similarity": ev.similarity,
                    "source": sub,
                })
        events.sort(key=lambda e: e["t"])
        saved = False
        if save:
            cue_label = (label or "").strip()
            if not cue_label:
                raise HTTPException(400, "name the audio cue before saving")
            await run_in_threadpool(game_packs.install_cue, game, cue_label, str(window))
            saved = True
        return {"events": events, "count": len(events), "saved": saved, "start": start, "duration": duration}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"audio window failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)
        window.unlink(missing_ok=True)


@router.post("/{game}/{event}")
async def add_cue(game: str, event: str,
                  url: str | None = Form(None),
                  file: UploadFile | None = File(None)) -> dict:
    """Install a cue for <game>/<event> from a URL or an uploaded file."""
    if not url and not file:
        raise HTTPException(400, "provide a sound url or file")
    try:
        if file is not None:
            suffix = Path(file.filename or "cue").suffix or ".bin"
            # Close the fd mkstemp opens, or Windows refuses the unlink below
            # while the handle is held ([WinError 32]).
            fd, tmp_name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                with open(tmp, "wb") as out:
                    while chunk := await file.read(1 << 20):
                        out.write(chunk)
                await run_in_threadpool(game_packs.install_cue, game, event, str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        else:
            await run_in_threadpool(game_packs.install_cue_from_url, game, event, url)
    except Exception as e:
        raise HTTPException(400, f"could not install cue: {e}")
    return list_cues()


@router.delete("/{game}/{event}")
def delete_cue(game: str, event: str) -> dict:
    game_packs.remove_cue(game, event)
    return list_cues()
