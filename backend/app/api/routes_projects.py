"""Project endpoints: import, list, status polling, retrieval, export, delete."""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import asyncio
import json
import threading

from fastapi import (APIRouter, File, Form, Header, HTTPException, Query,
                     Request, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
from urllib.parse import unquote

from .. import store
from ..config import get_settings
from ..models import (ASPECTS, AiBoostSettings, ContentType, GameProfileConfig,
                      ImportSettings, Platform, PowerMode, Project,
                      ProjectStatus, ProjectSummary)
from ..pipeline import ingest
from ..pipeline.captions import build_srt
from ..pipeline.nle_export import build_cmx3600, ready_clips_for_edl
from ..pipeline.orchestrator import engine
from ..providers.detect_gameplay import KNOWN_PROFILES

log = logging.getLogger("clipforge.api")

# Optional psutil — imported once at module level, not per-request.
try:
    import psutil as _psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False
    _psutil = None
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
    value = float(v)
    if not math.isfinite(value):
        return None
    return round(max(0.0, min(value, 60.0)), 3)


def _friendly_import_error(exc: Exception, url: str | None = None) -> str:
    """Map an import exception to a user-friendly, actionable message."""
    msg = str(exc).lower()
    if url:
        if "404" in msg or "not found" in msg:
            return f"URL not found or not accessible: '{url}'. Check the link is correct and the video is public."
        if "403" in msg or "forbidden" in msg:
            return f"Access denied for URL: '{url}'. The video may be private or geo-blocked."
        if "no such host" in msg or "resolve" in msg:
            return f"Could not reach '{url}'. Check your internet connection and that the URL is correct."
        if "ssl" in msg or "certificate" in msg:
            return f"SSL error while fetching '{url}'. Try downloading the file manually and uploading it instead."
    if "no space" in msg or "disk" in msg:
        return "Your disk is full. Free up space or set CLIPFORGE_DATA_DIR to a drive with more room."
    if "permission" in msg:
        return "Permission denied writing to the data directory. Check folder permissions."
    if "codec" in msg or "container" in msg or "moov" in msg:
        return ("The video file couldn't be read. It may be corrupted or in an unsupported format. "
                "Try re-encoding to H.264 MP4 with HandBrake first.")
    if "memory" in msg:
        return "ClipForge ran out of memory during import. Try a shorter video or close other applications."
    # Fallback: keep it short and remove file paths
    clean = str(exc).replace("\\", "/")
    clean = re.sub(r"[A-Za-z]:/[^\s,)]+", "[path]", clean)
    clean = re.sub(r"/tmp/[^\s,)]+", "[path]", clean)
    return f"Could not import the source: {clean[:200]}"


def _split_values(text: str | None) -> list[str]:
    raw = (text or "").replace("\r", "\n")
    if not raw.strip():
        return []
    chunks: list[str] = []
    for part in raw.replace(";", "\n").split("\n"):
        if "," in part and len(part) < 160:
            chunks.extend(part.split(","))
        else:
            chunks.append(part)
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        value = " ".join(chunk.split())
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out[:80]


def _parse_roi_json(text: str | None) -> list[dict]:
    if not text or not text.strip():
        return []
    try:
        raw = json.loads(text)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append({
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "w": float(item.get("w", 1.0)),
                "h": float(item.get("h", 1.0)),
            })
        except (TypeError, ValueError):
            continue
    return out[:40]


def _game_config_from_form(*, detection_mode: str, visual_rois_json: str,
                           visual_text_cues: str, reference_audio_files: str,
                           vlm_visual_prompts: str, audio_prompts: str,
                           audio_negative_prompts: str) -> GameProfileConfig:
    mode = detection_mode if detection_mode in {"zero_shot", "manual", "hybrid"} else "zero_shot"
    data = {
        "detection_mode": mode,
        "visual_rois": _parse_roi_json(visual_rois_json),
        "visual_text_cues": _split_values(visual_text_cues),
        "reference_audio_files": _split_values(reference_audio_files),
        "vlm_visual_prompts": _split_values(vlm_visual_prompts) or GameProfileConfig().vlm_visual_prompts,
        "audio_prompts": _split_values(audio_prompts),
        "audio_negative_prompts": (
            _split_values(audio_negative_prompts)
            or GameProfileConfig().audio_negative_prompts
        ),
    }
    cfg = GameProfileConfig.model_validate(data)
    cfg.visual_rois = [r.clamped() for r in cfg.visual_rois]
    return cfg


def _bool_value(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _float_value(value, default: float | None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str_value(value, default: str) -> str:
    return str(value) if value is not None else default


def _game_config_from_meta(value) -> GameProfileConfig:
    if not isinstance(value, dict):
        return _game_config_from_form(
            detection_mode="zero_shot",
            visual_rois_json="",
            visual_text_cues="",
            reference_audio_files="",
            vlm_visual_prompts="",
            audio_prompts="",
            audio_negative_prompts="",
        )
    data = dict(value)
    data["detection_mode"] = data.get("detection_mode", "zero_shot")
    data["visual_rois"] = data.get("visual_rois") or []
    data["visual_text_cues"] = data.get("visual_text_cues") or []
    data["reference_audio_files"] = data.get("reference_audio_files") or []
    data["vlm_visual_prompts"] = (
        data.get("vlm_visual_prompts")
        or GameProfileConfig().vlm_visual_prompts
    )
    data["audio_prompts"] = data.get("audio_prompts") or []
    data["audio_negative_prompts"] = (
        data.get("audio_negative_prompts")
        or GameProfileConfig().audio_negative_prompts
    )
    cfg = GameProfileConfig.model_validate(data)
    cfg.visual_rois = [r.clamped() for r in cfg.visual_rois]
    return cfg


def _build_import_settings(*, platform: str = "generic",
                           power_mode: str = "balanced",
                           min_len: float = 15.0,
                           max_len: float = 60.0,
                           target_clips: int = 10,
                           style_id: str = "bold-pop",
                           language: str = "de",
                           content_type: str = "auto",
                           aspect: str = "9:16",
                           burn_captions: bool = True,
                           game_profile: str = "auto",
                           tighten: bool = False,
                           denoise: bool = False,
                           motion: str = "none",
                           facecam_layout: str = "auto",
                           use_ocr: bool = True,
                           use_vlm: bool = True,
                           use_cues: bool = True,
                           use_audio_events: bool = True,
                           cue_learning: bool = True,
                           auto_length: bool = False,
                           lead_seconds: float | None = None,
                           tail_seconds: float | None = None,
                           ai_boost: AiBoostSettings | None = None,
                           game_config: GameProfileConfig | None = None
                           ) -> ImportSettings:
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
    return ImportSettings(
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
        use_cues=use_cues,
        use_audio_events=use_audio_events,
        cue_learning=cue_learning,
        auto_length=auto_length,
        ai_boost=ai_boost or AiBoostSettings(),
        lead_seconds=_clamp_pad(lead_seconds),
        tail_seconds=_clamp_pad(tail_seconds),
        game_config=game_config or _game_config_from_meta(None),
    )


def _settings_from_meta(data: dict) -> ImportSettings:
    boost = data.get("ai_boost") if isinstance(data.get("ai_boost"), dict) else {}
    return _build_import_settings(
        platform=_str_value(data.get("platform"), "generic"),
        power_mode=_str_value(data.get("power_mode"), "balanced"),
        min_len=_float_value(data.get("min_len"), 15.0) or 15.0,
        max_len=_float_value(data.get("max_len"), 60.0) or 60.0,
        target_clips=_int_value(data.get("target_clips"), 10),
        style_id=_str_value(data.get("style_id"), "bold-pop"),
        language=_str_value(data.get("language"), "de"),
        content_type=_str_value(data.get("content_type"), "auto"),
        aspect=_str_value(data.get("aspect"), "9:16"),
        burn_captions=_bool_value(data.get("burn_captions"), True),
        game_profile=_str_value(data.get("game_profile"), "auto"),
        tighten=_bool_value(data.get("tighten"), False),
        denoise=_bool_value(data.get("denoise"), False),
        motion=_str_value(data.get("motion"), "none"),
        facecam_layout=_str_value(data.get("facecam_layout"), "auto"),
        use_ocr=_bool_value(data.get("use_ocr"), True),
        use_vlm=_bool_value(data.get("use_vlm"), True),
        use_cues=_bool_value(data.get("use_cues"), True),
        use_audio_events=_bool_value(data.get("use_audio_events"), True),
        cue_learning=_bool_value(data.get("cue_learning"), True),
        auto_length=_bool_value(data.get("auto_length"), False),
        lead_seconds=_float_value(data.get("lead_seconds"), None),
        tail_seconds=_float_value(data.get("tail_seconds"), None),
        ai_boost=AiBoostSettings(
            emphasis=_bool_value(boost.get("emphasis"), True),
            emoji=_bool_value(boost.get("emoji"), True),
            speakerColors=_bool_value(boost.get("speakerColors"), True),
            autoZoom=_bool_value(boost.get("autoZoom"), True),
            broll=_bool_value(boost.get("broll"), False),
            hookCheck=_bool_value(boost.get("hookCheck"), True),
        ),
        game_config=_game_config_from_meta(data.get("game_config")),
    )


def _parse_upload_settings(raw: str) -> dict:
    try:
        data = json.loads(unquote(raw))
    except Exception:
        raise HTTPException(400, "upload settings were invalid JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "upload settings must be a JSON object")
    return data


async def _raw_upload_stream_parts(request: Request, settings_header: str | None
                                   ) -> tuple[dict, object, bytes]:
    """Return settings metadata, remaining body stream, and first file bytes.

    New clients send ``CFMETA <json-byte-count>\n<json><file-bytes>`` as one raw
    octet stream. Header metadata is still accepted for older dev builds.
    """
    stream = request.stream()
    if settings_header:
        return _parse_upload_settings(settings_header), stream, b""

    buf = b""
    async for chunk in stream:
        if chunk:
            buf += chunk
            break
    if not buf:
        return {}, stream, b""
    if not buf.startswith(b"CFMETA "):
        return {}, stream, buf

    while b"\n" not in buf:
        if len(buf) > 4096:
            raise HTTPException(400, "upload settings prefix is too large")
        try:
            chunk = await anext(stream)
        except StopAsyncIteration:
            raise HTTPException(400, "upload settings prefix was incomplete")
        buf += chunk
    line, rest = buf.split(b"\n", 1)
    try:
        meta_len = int(line[len(b"CFMETA "):].strip())
    except ValueError:
        raise HTTPException(400, "upload settings prefix was invalid")
    if meta_len < 0 or meta_len > 1_000_000:
        raise HTTPException(400, "upload settings metadata is too large")
    while len(rest) < meta_len:
        try:
            chunk = await anext(stream)
        except StopAsyncIteration:
            raise HTTPException(400, "upload settings metadata was incomplete")
        rest += chunk
    raw_meta = rest[:meta_len]
    first_file_bytes = rest[meta_len:]
    try:
        data = json.loads(raw_meta.decode("utf-8"))
    except Exception:
        raise HTTPException(400, "upload settings were invalid JSON")
    if not isinstance(data, dict):
        raise HTTPException(400, "upload settings must be a JSON object")
    return data, stream, first_file_bytes


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
        if _HAS_PSUTIL and _psutil is not None:
            sample["cpu_pct"] = float(_psutil.cpu_percent(interval=None))
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
    use_cues: bool = Form(True),
    use_audio_events: bool = Form(True),
    cue_learning: bool = Form(True),
    auto_length: bool = Form(False),
    lead_seconds: float | None = Form(None),
    tail_seconds: float | None = Form(None),
    detection_mode: str = Form("zero_shot"),
    visual_rois_json: str = Form(""),
    visual_text_cues: str = Form(""),
    reference_audio_files: str = Form(""),
    vlm_visual_prompts: str = Form(""),
    audio_prompts: str = Form(""),
    audio_negative_prompts: str = Form(""),
    # AI Boost toggles (production-value passes, each on by default).
    ai_boost_emphasis: bool = Form(True),
    ai_boost_emoji: bool = Form(True),
    ai_boost_speaker_colors: bool = Form(True),
    ai_boost_auto_zoom: bool = Form(True),
    ai_boost_broll: bool = Form(False),
    ai_boost_hook_check: bool = Form(True),
    file: UploadFile | None = File(None),
) -> Project:
    if not file and not url:
        raise HTTPException(400, "provide either a file upload or a url")
    settings = _build_import_settings(
        platform=platform,
        power_mode=power_mode,
        min_len=min_len,
        max_len=max_len,
        target_clips=target_clips,
        style_id=style_id,
        language=language,
        content_type=content_type,
        aspect=aspect,
        burn_captions=burn_captions,
        game_profile=game_profile,
        tighten=tighten,
        denoise=denoise,
        motion=motion,
        facecam_layout=facecam_layout,
        use_ocr=use_ocr,
        use_vlm=use_vlm,
        use_cues=use_cues,
        use_audio_events=use_audio_events,
        cue_learning=cue_learning,
        auto_length=auto_length,
        lead_seconds=lead_seconds,
        tail_seconds=tail_seconds,
        ai_boost=AiBoostSettings(
            emphasis=ai_boost_emphasis,
            emoji=ai_boost_emoji,
            speakerColors=ai_boost_speaker_colors,
            autoZoom=ai_boost_auto_zoom,
            broll=ai_boost_broll,
            hookCheck=ai_boost_hook_check,
        ),
        game_config=_game_config_from_form(
            detection_mode=detection_mode,
            visual_rois_json=visual_rois_json,
            visual_text_cues=visual_text_cues,
            reference_audio_files=reference_audio_files,
            vlm_visual_prompts=vlm_visual_prompts,
            audio_prompts=audio_prompts,
            audio_negative_prompts=audio_negative_prompts,
        ),
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
        error_msg = _friendly_import_error(e, url)
        raise HTTPException(400, error_msg)

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


@router.post("/raw-upload")
async def create_project_raw_upload(
    request: Request,
    filename: str = Query("upload.mp4"),
    x_clipforge_settings: str | None = Header(None, alias="X-ClipForge-Settings"),
) -> Project:
    """Create a project from a raw file body.

    Large videos should not go through multipart parsing: Starlette may spool the
    whole body before our handler runs, and parser/temp-disk failures surface as
    the unhelpful "error parsing the body". This route streams bytes directly to
    our temp file and uses a small JSON settings header for project options.
    """
    data, body_stream, first_file_bytes = await _raw_upload_stream_parts(
        request, x_clipforge_settings)
    upload_name = Path(filename or "upload.mp4").name or "upload.mp4"
    try:
        settings = _settings_from_meta(data)
    except Exception as e:
        raise HTTPException(400, f"upload settings were invalid: {e}")
    project = Project(name=_str_value(data.get("name"), "Untitled") or "Untitled",
                      settings=settings, status=ProjectStatus.created)
    store.save(project)

    tmp: Path | None = None

    def _discard() -> None:
        store.delete(project.id)
        shutil.rmtree(get_settings().media_dir / project.id, ignore_errors=True)
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    try:
        cap = get_settings().upload_cap_bytes
        fd, tmp_name = tempfile.mkstemp(suffix=Path(upload_name).suffix or ".mp4")
        os.close(fd)
        tmp = Path(tmp_name)
        size = 0
        with open(tmp, "wb") as out:
            if first_file_bytes:
                size += len(first_file_bytes)
                if cap is not None and size > cap:
                    raise HTTPException(
                        413, f"file exceeds the {get_settings().max_upload_mb} MB "
                             "upload limit (CLIPFORGE_MAX_UPLOAD_MB; 0 = unlimited)")
                out.write(first_file_bytes)
            async for chunk in body_stream:
                if not chunk:
                    continue
                size += len(chunk)
                if cap is not None and size > cap:
                    raise HTTPException(
                        413, f"file exceeds the {get_settings().max_upload_mb} MB "
                             "upload limit (CLIPFORGE_MAX_UPLOAD_MB; 0 = unlimited)")
                out.write(chunk)
        if size <= 0:
            raise HTTPException(400, "uploaded file was empty")
        src = await run_in_threadpool(ingest.attach_source_file, project,
                                      tmp, upload_name)
        tmp = None
    except HTTPException:
        _discard()
        raise
    except Exception as e:
        _discard()
        error_msg = _friendly_import_error(e)
        raise HTTPException(400, error_msg)

    try:
        with store.mutate(project.id) as p:
            p.source = src
            if not data.get("name") or data.get("name") == "Untitled":
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


def _progress_timing(p) -> dict:
    """Elapsed + estimated-remaining seconds for the live processing UI.

    ETA extrapolates from how long it took to reach the current percent — it is
    self-correcting as the run proceeds and stays None until there's enough
    signal (a few percent in) to avoid wild early guesses.

    The frontend also receives ``rendered_count`` and ``target_clips`` so it
    can compute a render-specific ETA from clip throughput when needed.
    """
    import time
    prog = p.progress
    started = prog.started_at
    processing = str(getattr(p.status, "value", p.status)) == "processing"
    elapsed = round(time.time() - started, 1) if started else None
    eta = None
    if processing and started and elapsed is not None and prog.pct >= 3.0:
        remaining = elapsed * (100.0 - prog.pct) / max(prog.pct, 1e-6)
        eta = round(max(0.0, remaining), 1)
    return {
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "source_duration": round(p.source.duration, 1) if p.source else None,
    }


def _status_payload(project_id: str) -> dict | None:
    p = store.get(project_id)
    if not p:
        return None
    rendered = sum(1 for c in p.clips if c.thumb_url)
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
        "timing": _progress_timing(p),
        "target_clips": p.settings.target_clips,
        "rendered_count": rendered,
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


@router.post("/{project_id}/pause")
def pause_project(project_id: str) -> dict:
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    if p.status not in (ProjectStatus.queued, ProjectStatus.processing, ProjectStatus.paused):
        raise HTTPException(409, "only queued or processing projects can be paused")
    engine.pause(project_id)
    return project_status(project_id)


@router.post("/{project_id}/resume")
def resume_project(project_id: str) -> dict:
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    if p.status != ProjectStatus.paused:
        raise HTTPException(409, "project is not paused")
    engine.resume(project_id)
    return project_status(project_id)


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

@router.delete("/{project_id}/purge")
def purge_project(project_id: str) -> dict:
    """Irreversibly erase every trace of a project: database row, media files,
    temp artifacts, and any cached data. Unlike DELETE this is a full forensic
    wipe — use it for Privacy Mode compliance (e.g. when a user asks to have
    their content removed from the system).

    Returns {"ok": true} regardless of whether the project existed, so
    callers can safely purge without a preliminary existence check.
    """
    p = store.get(project_id)
    if p:
        shutil.rmtree(get_settings().media_dir / project_id, ignore_errors=True)
        tmp_root = Path(tempfile.gettempdir())
        for item in tmp_root.glob(f"*{project_id}*"):
            try:
                item.unlink(missing_ok=True)
            except Exception:
                pass
        store.delete(project_id)
    return {"ok": True}


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
    use_cues: bool | None = None
    use_audio_events: bool | None = None
    cue_learning: bool | None = None
    auto_length: bool | None = None
    lead_seconds: float | None = None
    tail_seconds: float | None = None
    game_config: GameProfileConfig | None = None
    # AI Boost overrides — let users toggle production-value passes after import
    # without re-uploading. Each field is Optional so the caller can send a subset.
    ai_boost_emphasis: bool | None = None
    ai_boost_emoji: bool | None = None
    ai_boost_speaker_colors: bool | None = None
    ai_boost_auto_zoom: bool | None = None
    ai_boost_broll: bool | None = None
    ai_boost_hook_check: bool | None = None


@router.post("/{project_id}/reprocess", response_model=Project)
def reprocess(project_id: str, body: Reprocess | None = None) -> Project:
    """Re-run the whole pipeline on the stored source — applies new ratings,
    cues, or any overridden settings, without re-uploading."""
    p = store.get(project_id)
    if not p or not p.source:
        raise HTTPException(404, "project or source missing")
    if p.status in (ProjectStatus.queued, ProjectStatus.processing, ProjectStatus.paused):
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
        if body.use_cues is not None:
            s.use_cues = body.use_cues
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
        if body.game_config is not None:
            body.game_config.visual_rois = [r.clamped() for r in body.game_config.visual_rois]
            s.game_config = body.game_config
        if body.min_len is not None and body.max_len is not None:
            s.min_len, s.max_len = _clamp_lengths(body.min_len, body.max_len)
        # AI Boost overrides — merge whichever fields the caller sent.
        if body.ai_boost_emphasis is not None:
            s.ai_boost.emphasis = body.ai_boost_emphasis
        if body.ai_boost_emoji is not None:
            s.ai_boost.emoji = body.ai_boost_emoji
        if body.ai_boost_speaker_colors is not None:
            s.ai_boost.speakerColors = body.ai_boost_speaker_colors
        if body.ai_boost_auto_zoom is not None:
            s.ai_boost.autoZoom = body.ai_boost_auto_zoom
        if body.ai_boost_broll is not None:
            s.ai_boost.broll = body.ai_boost_broll
        if body.ai_boost_hook_check is not None:
            s.ai_boost.hookCheck = body.ai_boost_hook_check
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
        proj.events = []
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
    if p.status in (ProjectStatus.queued, ProjectStatus.processing, ProjectStatus.paused):
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
            path = settings.media_dir / (c.export_url[7:] if c.export_url.startswith("/media/") else c.export_url)
            if path.exists():
                stem = f"{i:02d}_{c.score:02d}_{_safe_name(c.title)}"
                z.write(path, f"{stem}.mp4")
                # caption sidecar (.srt) for editing in a desktop NLE
                if c.captions.words:
                    z.writestr(f"{stem}.srt", build_srt(c.captions))
    fname = f"{_safe_name(p.name)}_clips.zip"
    return FileResponse(tmp_zip, media_type="application/zip", filename=fname,
                        background=BackgroundTask(lambda: tmp_zip.unlink(missing_ok=True)))


@router.get("/{project_id}/export/premiere")
def export_premiere(project_id: str):
    """Zip a source-based CMX 3600 EDL plus SRT sidecars for desktop editing."""
    p = store.get(project_id)
    if not p:
        raise HTTPException(404, "project not found")
    if not p.source:
        raise HTTPException(409, "project source is missing")
    ready = ready_clips_for_edl(p)
    if not ready:
        raise HTTPException(409, "no rendered clips to export yet")

    settings = get_settings()
    source_file = settings.media_dir / p.source.path
    fd, zip_name = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    tmp_zip = Path(zip_name)
    edl_name = f"{_safe_name(p.name)}.edl"
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as z:
        z.writestr(edl_name, build_cmx3600(p, source_file=source_file, clips=ready))
        for i, c in enumerate(ready, 1):
            if c.captions.words:
                stem = f"{i:02d}_{c.score:02d}_{_safe_name(c.title)}"
                z.writestr(f"captions/{stem}.srt", build_srt(c.captions))
    fname = f"{_safe_name(p.name)}_premiere_edl.zip"
    return FileResponse(tmp_zip, media_type="application/zip", filename=fname,
                        background=BackgroundTask(lambda: tmp_zip.unlink(missing_ok=True)))


@router.post("/{project_id}/mark-highlight")
def mark_highlight(project_id: str, timestamp: float, duration: float = 30.0):
    """Streamer.bot webhook: mark a highlight moment during a live stream.

    When called with a source timestamp and duration, creates a clip at that
    position and queues it for rendering. Streamer.bot can call this via its
    HTTP request action:
    ``POST http://localhost:8000/api/projects/{project_id}/mark-highlight?timestamp=1234.5&duration=30``
    """
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    if not project.source:
        raise HTTPException(409, "project has no source video")
    if not project.transcript:
        raise HTTPException(409, "project hasn't been transcribed yet")

    from ..models import Clip, ClipStatus, CaptionSet, Reframe, LayoutType, ReframeKeyframe
    clip = Clip(
        id=uuid.uuid4().hex[:12],
        start=round(max(timestamp - duration / 2, 0), 3),
        end=round(min(timestamp + duration / 2, project.source.duration), 3),
        title=f"Highlight @ {time.strftime('%H:%M:%S', time.gmtime(timestamp))}",
        kind="talking", status=ClipStatus.pending,
        captions=CaptionSet(),
        reframe=Reframe(layout=LayoutType.fill, keyframes=[ReframeKeyframe(t=0.0, cx=0.5)]),
    )
    dur = clip.end - clip.start
    if dur < 5:
        clip.start = max(clip.end - 10, 0)
    if clip.end - clip.start > 120:
        clip.end = min(clip.start + 120, project.source.duration)

    with store.mutate(project_id) as p:
        p.clips.append(clip)
    threading.Thread(target=engine.rerender_clip, args=(project_id, clip.id),
                     daemon=True).start()
    return {"ok": True, "clip_id": clip.id, "start": clip.start,
            "end": clip.end, "duration": round(clip.end - clip.start, 1)}
