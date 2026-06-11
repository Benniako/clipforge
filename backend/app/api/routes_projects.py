"""Project endpoints: import, list, status polling, retrieval, export, delete."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .. import store
from ..config import get_settings
from ..models import (ASPECTS, ContentType, ImportSettings, Platform, Project,
                      ProjectStatus, ProjectSummary)
from ..pipeline import ingest
from ..pipeline.captions import build_srt
from ..pipeline.orchestrator import engine
from ..providers.detect_gameplay import KNOWN_PROFILES

log = logging.getLogger("clipforge.api")
router = APIRouter(prefix="/api/projects", tags=["projects"])


def _safe_name(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in s).strip()
    return (keep or "clip")[:60]


@router.post("")
async def create_project(
    name: str = Form("Untitled"),
    url: str | None = Form(None),
    platform: str = Form("generic"),
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
    motion: str = Form("none"),
    facecam_layout: str = Form("auto"),
    file: UploadFile | None = File(None),
) -> Project:
    if not file and not url:
        raise HTTPException(400, "provide either a file upload or a url")
    try:
        plat = Platform(platform)
    except ValueError:
        plat = Platform.generic

    try:
        ctype = ContentType(content_type)
    except ValueError:
        ctype = ContentType.auto
    settings = ImportSettings(
        platform=plat,
        min_len=max(3.0, min(min_len, max_len)),
        max_len=max(min_len, max_len),
        target_clips=max(1, min(target_clips, 30)),
        default_style_id=style_id,
        language=language if language in ("auto", "en", "de") else "de",
        content_type=ctype,
        aspect=aspect if aspect in ASPECTS else "9:16",
        burn_captions=burn_captions,
        game_profile=game_profile if game_profile in KNOWN_PROFILES else "auto",
        tighten=tighten,
        motion=motion if motion in ("none", "push") else "none",
        facecam_layout=(facecam_layout
                        if facecam_layout in ("auto", "off", "split", "framed")
                        else "auto"),
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
            cap = get_settings().max_upload_mb * 1024 * 1024
            size = 0
            with open(tmp, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > cap:
                        raise HTTPException(413, "file exceeds upload limit")
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


@router.get("/{project_id}/status")
def project_status(project_id: str) -> dict:
    """Lightweight polling payload for the processing + grid views."""
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    return {
        "id": p.id,
        "status": p.status,
        "error": p.error,
        "warnings": p.warnings,
        "content_type": p.content_type,
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
    motion: str | None = None
    facecam_layout: str | None = None


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
        if body.motion in ("none", "push"):
            s.motion = body.motion
        if body.facecam_layout in ("auto", "off", "split", "framed"):
            s.facecam_layout = body.facecam_layout
        if body.min_len is not None and body.max_len is not None:
            s.min_len, s.max_len = max(3.0, min(body.min_len, body.max_len)), max(body.min_len, body.max_len)
        if body.target_clips is not None:
            s.target_clips = max(1, min(body.target_clips, 30))

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
