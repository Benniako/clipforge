"""Ingest stage — get a source video into the system and describe it.

Accepts either an uploaded file or a pasted URL. URLs are fetched with yt-dlp
when available (handles YouTube and many hosts); otherwise a direct media URL is
downloaded over HTTP. Every source is probed so the rest of the pipeline knows
its duration, dimensions, and whether it carries audio.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

from ..config import get_settings
from ..media import ffmpeg
from ..models import Project, SourceMedia

log = logging.getLogger("clipforge.ingest")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".flv"}


def project_dir(project_id: str) -> Path:
    d = get_settings().media_dir / project_id
    (d / "clips").mkdir(parents=True, exist_ok=True)
    return d


def attach_source_file(project: Project, tmp_path: str | Path, filename: str) -> SourceMedia:
    """Move an uploaded temp file into the project and probe it."""
    ext = Path(filename).suffix.lower() or ".mp4"
    if ext not in VIDEO_EXTS:
        raise ValueError(f"unsupported file type '{ext}'")
    dest = project_dir(project.id) / f"source{ext}"
    shutil.move(str(tmp_path), dest)
    return _finalize(project, dest, filename=filename, url=None)


def attach_source_url(project: Project, url: str) -> SourceMedia:
    if not (url or "").lower().startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs can be imported")
    dest_stem = project_dir(project.id) / "source"
    settings = get_settings()
    if settings.has_ytdlp:
        dest = _download_ytdlp(url, dest_stem)
    else:
        dest = _download_http(url, dest_stem)
    return _finalize(project, dest, filename=dest.name, url=url)


def _download_ytdlp(url: str, dest_stem: Path) -> Path:
    import yt_dlp

    opts = {
        "outtmpl": str(dest_stem) + ".%(ext)s",
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    if get_settings().ffmpeg:
        opts["ffmpeg_location"] = str(Path(get_settings().ffmpeg).parent)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():  # merged file may carry a different ext
        cands = sorted(dest_stem.parent.glob(dest_stem.name + ".*"))
        if not cands:
            raise RuntimeError("yt-dlp produced no output file")
        path = cands[0]
    return path


def _download_http(url: str, dest_stem: Path) -> Path:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in VIDEO_EXTS:
        ext = ".mp4"
    dest = dest_stem.with_suffix(ext)
    # Same cap as file uploads — a direct-URL import shouldn't be the one
    # path that can fill the disk unbounded.
    cap = get_settings().max_upload_mb * 1024 * 1024
    size = 0
    req = urllib.request.Request(url, headers={"User-Agent": "ClipForge/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            size += len(chunk)
            if size > cap:
                raise ValueError("download exceeds the upload size limit")
            f.write(chunk)
    return dest


def _finalize(project: Project, path: Path, *, filename: str, url: str | None) -> SourceMedia:
    info = ffmpeg.probe(path)
    if not info.has_video or info.duration <= 0:
        raise ValueError("file does not appear to be a playable video")
    # A poster frame for the project / upload card.
    try:
        ffmpeg.make_thumbnail(path, project_dir(project.id) / "source.jpg",
                              at=min(info.duration * 0.1, 3.0), width=640)
    except Exception as e:
        log.warning("source thumbnail failed: %s", e)
    rel = path.relative_to(get_settings().media_dir)
    return SourceMedia(
        filename=filename, path=str(rel), url=url,
        duration=info.duration, width=info.width, height=info.height,
        fps=info.fps, size_bytes=path.stat().st_size,
    )
