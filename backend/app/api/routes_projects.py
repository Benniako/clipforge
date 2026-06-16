"""Project endpoints: import, list, status polling, retrieval, export, delete."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import asyncio
import threading

from fastapi import (APIRouter, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .. import store
from ..config import get_settings
from ..models import (ASPECTS, ContentType, ImportSettings, Platform, PowerMode,
                      Project, ProjectStatus, ProjectSummary)
from ..pipeline import ingest
from ..pipeline.captions import build_srt
from ..pipeline.orchestrator import engine
from ..providers.detect_gameplay import KNOWN_PROFILES

log = logging.getLogger("clipforge.api")
router = APIRouter(prefix="/api/projects", tags=["projects"])
_SYSTEM_SAMPLE: tuple[float, dict] | None = None


def _safe_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()
    return (keep or "clip")[:60]


def _auto_length_range(platform: Platform, content_type: ContentType) -> tuple[float, float]:
    """Short-form defaults when Auto length is enabled."""
    if content_type == ContentType.gameplay:
        return 12.0, 35.0
    if platform in (Platform.tiktok, Platform.reels):
        return 12.0, 42.0
    if platform == Platform.shorts:
        return 18.0, 55.0
    return 15.0, 60.0


def _clamp_lengths(min_len: float, max_len: float) -> tuple[float, float]:
    return max(3.0, min(min_len, max_len)), max(min_len, max_len)


def _clamp_pad(v: float | None) -> float | None:
    if v is None:
        return None
    return round(max(0.0, min(float(v), 60.0)), 3)


def _system_usage() -> dict:
    """Best-effort CPU/GPU use for the live processing UI.

    All fields are nullable so a missing psutil/nvidia-smi never breaks polling.
    Cached briefly because the status endpoint is called often while rendering.
    """
    global _SYSTEM_SAMPLE
    now = time.time()
    if _SYSTEM_SAMPLE and now - _SYSTEM_SAMPLE[0] < 2.0:
        return _SYSTEM_SAMPLE[1]

    sample = {
        "cpu_pct": None,
        "gpu_pct": None,
        "gpu_mem_mb": None,
        "gpu_mem_total_mb": None,
    }
    try:
        import psutil  # type: ignore

        sample["cpu_pct"] = float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=0.8,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = [p.strip() for p in proc.stdout.splitlines()[0].split(",")]
            if len(parts) >= 3:
                sample["gpu_pct"] = float(parts[0])
                sample["gpu_mem_mb"] = float(parts[1])
                sample["gpu_mem_total_mb"] = float(parts[2])
    except Exception:
        pass

    _SYSTEM_SAMPLE = (now, sample)
    return sample


@router.post("")
async def create_project(
    name: str = Form("Untitled"),
    url: str | None = Form(None),
    platform: str = Form("generic"),
    power_mode: str = Form("balanced"),
    min_len: float = Form(15.0),
    max_len: float = Form(60.0),
    target_clips: int = Form(10),
    style_id: str = Form("bold-pop"),
    language: str = Form("de"),
    content_type: str = Form("auto"),
    aspect: str = Form("9:16"),
    burn_captions: bool = Form(True),
    game_profile: str = Form("auto"),
    tighten: bool = Form(False),
    denoise: bool = Form(False),
    motion: str = Form("none"),
    facecam_layout: str = Form("auto"),
    use_ocr: bool = Form(True),
    use_vlm: bool = Form(True),
    use_audio_events: bool = Form(True),
    cue_learning: bool = Form(True),
    auto_length: bool = Form(False),
    lead_seconds: float | None = Form(None),
    tail_seconds: float | None = Form(None),
    file: UploadFile | None = File(None),
) -> Project:
    if not file and not url:
        raise HTTPException(400, "provide either a file upload or a url")
    try:
        plat = Platform(platform)
    except ValueError:
        plat = Platform.generic
    try:
        pmode = PowerMode(power_mode)
    except ValueError:
        pmode = PowerMode.balanced

    try:
        ctype = ContentType(content_type)
    except ValueError:
        ctype = ContentType.auto
    clean_min, clean_max = (_auto_length_range(plat, ctype) if auto_length
                            else _clamp_lengths(min_len, max_len))
    settings = ImportSettings(
        platform=plat,
        power_mode=pmode,
        min_len=clean_min,
        max_len=clean_max,
        target_clips=max(1, min(target_clips, 30)),
        default_style_id=style_id,
        language=language if language in ("auto", "en", "de") else "de",
        content_type=ctype,
        aspect=aspect if aspect in ASPECTS else "9:16",
        burn_captions=burn_captions,
        game_profile=game_profile if game_profile in KNOWN_PROFILES else "auto",
        tighten=tighten,
        denoise=denoise,
        motion=motion if motion in ("none", "push") else "none",
        facecam_layout=(facecam_layout
                        if facecam_layout in ("auto", "off", "split", "framed")
                        else "auto"),
        use_ocr=use_ocr,
        use_vlm=use_vlm,
        use_audio_events=use_audio_events,
        cue_learning=cue_learning,
        auto_length=auto_length,
        lead_seconds=_clamp_pad(lead_seconds),
        tail_seconds=_clamp_pad(tail_seconds),
    )
    project = Project(name=name or "Untitled", settings=settings,
                      status=ProjectStatus.created)
    store.save(project)

    tmp: Path | None = None

    def _discard() -> None:
        # Roll back everything a failed import may have left behind: the DB
        # row, the project's media directory, and the upload temp file.
        store.delete(project.id)
        shutil.rmtree(get_settings().media_dir / project.id, ignore_errors=True)
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    try:
        if file is not None:
            # Close the fd mkstemp opens, or Windows won't let us move the file
            # afterwards ([WinError 32] file used by another process).
            fd, tmp_name = tempfile.mkstemp(suffix=Path(file.filename or "v.mp4").suffix)
            os.close(fd)
            tmp = Path(tmp_name)
            cap = get_settings().upload_cap_bytes
            size = 0
            with open(tmp, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if cap is not None and size > cap:
                        raise HTTPException(
                            413, f"file exceeds the {get_settings().max_upload_mb} MB "
                                 "upload limit (CLIPFORGE_MAX_UPLOAD_MB; 0 = unlimited)")
                    out.write(chunk)
            src = await run_in_threadpool(ingest.attach_source_file, project,
                                          tmp, file.filename or "upload.mp4")
        else:
            src = await run_in_threadpool(ingest.attach_source_url, project, url)
    except HTTPException:
        _discard()
        raise
    except Exception as e:
        _discard()
        raise HTTPException(400, f"could not import source: {e}")

    try:
        with store.mutate(project.id) as p:
            p.source = src
            if not name or name == "Untitled":
                p.name = Path(src.filename).stem[:60] or "Untitled"
        engine.enqueue(project.id)
    except Exception as e:
        _discard()
        raise HTTPException(500, f"could not start processing: {e}")
    return store.get(project.id)


@router.get("", response_model=list[ProjectSummary])
def list_projects() -> list[ProjectSummary]:
    return store.list_summaries()


@router.get("/{project_id}", response_model=Project)
def get_project(project_id: str) -> Project:
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    return p


def _status_payload(project_id: str) -> dict | None:
    p = store.get(project_id)
    if not p:
        return None
    return {
        "id": p.id,
        "status": p.status,
        "error": p.error,
        "warnings": p.warnings,
        "content_type": p.content_type,
        "settings": {
            "power_mode": p.settings.power_mode,
            "aspect": p.settings.aspect,
        },
        "system": _system_usage(),
        "progress": p.progress.model_dump(),
        "clips": [
            {
                "id": c.id, "title": c.title, "score": c.score, "kind": c.kind,
                "status": c.status,
                "duration": round(c.tightened_duration or c.duration, 2),
                "thumb_url": c.thumb_url, "export_url": c.export_url,
            }
            for c in p.clips
        ],
    }


@router.get("/{project_id}/status")
def project_status(project_id: str) -> dict:
    """Lightweight polling payload for the processing + grid views."""
    payload = _status_payload(project_id)
    if payload is None:
        raise HTTPException(404, "project not found")
    return payload


@router.websocket("/{project_id}/ws")
async def project_ws(ws: WebSocket, project_id: str) -> None:
    """Push status updates while a project processes (UI falls back to
    polling when this isn't available). Sends only on change, closes once
    the project reaches a terminal state."""
    await ws.accept()
    last: dict | None = None
    try:
        while True:
            payload = await run_in_threadpool(_status_payload, project_id)
            if payload is None:
                await ws.send_json({"error": "project not found"})
                break
            encoded = jsonable_encoder(payload)
            if encoded != last:
                await ws.send_json(encoded)
                last = encoded
            if payload["status"] in ("ready", "failed"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    try:
        await ws.close()
    except Exception:
        pass


@router.delete("/{project_id}")
def delete_project(project_id: str) -> dict:
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    shutil.rmtree(get_settings().media_dir / project_id, ignore_errors=True)
    store.delete(project_id)
    return {"deleted": project_id}


class Reprocess(BaseModel):
    """Optional setting overrides applied before re-running the pipeline."""
    platform: str | None = None
    power_mode: str | None = None
    content_type: str | None = None
    aspect: str | None = None
    min_len: float | None = None
    max_len: float | None = None
    target_clips: int | None = None
    style_id: str | None = None
    language: str | None = None
    burn_captions: bool | None = None
    game_profile: str | None = None
    tighten: bool | None = None
    denoise: bool | None = None
    motion: str | None = None
    facecam_layout: str | None = None
    use_ocr: bool | None = None
    use_vlm: bool | None = None
    use_audio_events: bool | None = None
    cue_learning: bool | None = None
    auto_length: bool | None = None
    lead_seconds: float | None = None
    tail_seconds: float | None = None


@router.post("/{project_id}/reprocess", response_model=Project)
def reprocess(project_id: str, body: Reprocess | None = None) -> Project:
    """Re-run the whole pipeline on the stored source — applies new ratings,
    cues, or any overridden settings, without re-uploading."""
    p = store.get(project_id)
    if not p or not p.source:
        raise HTTPException(404, "project or source missing")
    if p.status in (ProjectStatus.queued, ProjectStatus.processing):
        # A second concurrent run would fight the first over the same clip
        # files and clobber its store writes.
        raise HTTPException(409, "project is still processing — wait for it to finish")
    s = p.settings
    if body:
        if body.platform:
            try:
                s.platform = Platform(body.platform)
            except ValueError:
                pass
        if body.power_mode:
            try:
                s.power_mode = PowerMode(body.power_mode)
            except ValueError:
                pass
        if body.content_type:
            try:
                s.content_type = ContentType(body.content_type)
            except ValueError:
                pass
        if body.aspect in ASPECTS:
            s.aspect = body.aspect
        if body.language in ("auto", "en", "de"):
            s.language = body.language
        if body.style_id:
            s.default_style_id = body.style_id
        if body.game_profile in KNOWN_PROFILES:
            s.game_profile = body.game_profile
        if body.burn_captions is not None:
            s.burn_captions = body.burn_captions
        if body.tighten is not None:
            s.tighten = body.tighten
        if body.denoise is not None:
            s.denoise = body.denoise
        if body.motion in ("none", "push"):
            s.motion = body.motion
        if body.facecam_layout in ("auto", "off", "split", "framed"):
            s.facecam_layout = body.facecam_layout
        if body.use_ocr is not None:
            s.use_ocr = body.use_ocr
        if body.use_vlm is not None:
            s.use_vlm = body.use_vlm
        if body.use_audio_events is not None:
            s.use_audio_events = body.use_audio_events
        if body.cue_learning is not None:
            s.cue_learning = body.cue_learning
        if body.auto_length is not None:
            s.auto_length = body.auto_length
        if "lead_seconds" in body.model_fields_set:
            s.lead_seconds = _clamp_pad(body.lead_seconds)
        if "tail_seconds" in body.model_fields_set:
            s.tail_seconds = _clamp_pad(body.tail_seconds)
        if body.min_len is not None and body.max_len is not None:
            s.min_len, s.max_len = _clamp_lengths(body.min_len, body.max_len)
        if body.target_clips is not None:
            s.target_clips = max(1, min(body.target_clips, 30))
        if s.auto_length:
            s.min_len, s.max_len = _auto_length_range(s.platform, s.content_type)

    # Drop previous outputs (files + records); the source media is kept.
    pdir = get_settings().media_dir / project_id
    shutil.rmtree(pdir / "clips", ignore_errors=True)
    shutil.rmtree(pdir / "montages", ignore_errors=True)
    with store.mutate(project_id) as proj:
        proj.settings = s
        proj.clips = []
        proj.montages = []
        proj.content_type = None
        proj.facecam = None      # re-detected on the next run
        proj.warnings = []
        proj.error = None
        proj.status = ProjectStatus.created
    engine.enqueue(project_id)
    return store.get(project_id)


class AspectBody(BaseModel):
    aspect: str


@router.post("/{project_id}/aspect", response_model=Project)
def set_aspect(project_id: str, body: AspectBody) -> Project:
    """Change the output format AFTER processing: re-renders every clip in the
    new aspect without re-running transcription/detection/scoring. Per-clip
    aspect overrides made in the editor keep winning."""
    if body.aspect not in ASPECTS:
        raise HTTPException(400, f"unknown aspect '{body.aspect}'")
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    if p.status in (ProjectStatus.queued, ProjectStatus.processing):
        raise HTTPException(409, "project is still processing — wait for it to finish")
    if not p.clips:
        raise HTTPException(409, "no clips to re-render yet")
    with store.mutate(project_id) as proj:
        proj.settings.aspect = body.aspect
    threading.Thread(target=engine.rerender_all, args=(project_id,),
                     daemon=True).start()
    return store.get(project_id)


@router.get("/{project_id}/export")
def export_batch(project_id: str):
    """Zip every ready clip for one-click batch download."""
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    settings = get_settings()
    ready = [c for c in p.clips if c.export_url]
    if not ready:
        raise HTTPException(409, "no rendered clips to export yet")

    fd, zip_name = tempfile.mkstemp(suffix=".zip")
    os.close(fd)  # release the handle before zipfile reopens it (Windows-safe)
    tmp_zip = Path(zip_name)
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as z:
        for i, c in enumerate(sorted(ready, key=lambda c: c.score, reverse=True), 1):
            path = settings.media_dir / c.export_url.removeprefix("/media/")
            if path.exists():
                stem = f"{i:02d}_{c.score:02d}_{_safe_name(c.title)}"
                z.write(path, f"{stem}.mp4")
                # caption sidecar (.srt) for editing in a desktop NLE
                if c.captions.words:
                    z.writestr(f"{stem}.srt", build_srt(c.captions))
    fname = f"{_safe_name(p.name)}_clips.zip"
    return FileResponse(tmp_zip, media_type="application/zip", filename=fname,
                        background=BackgroundTask(lambda: tmp_zip.unlink(missing_ok=True)))
