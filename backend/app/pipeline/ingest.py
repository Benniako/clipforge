"""Ingest stage — get a source video into the system and describe it.

Accepts either an uploaded file or a pasted URL. URLs are fetched with yt-dlp
when available (handles YouTube and many hosts); otherwise a direct media URL is
downloaded over HTTP. Every source is probed so the rest of the pipeline knows
its duration, dimensions, and whether it carries audio.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .._util import http_download
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
    # Proactively warn when the YouTube path is known-degraded (no deno → the
    # JS runtime yt-dlp needs to read YouTube's player). The download still
    # works but may cap at 360p. We log it here; the API layer can also call
    # detect_ytdlp_warnings() to surface it in the project's warnings list.
    if "youtu" in (url or "").lower():
        warn = detect_ytdlp_warnings()
        if warn:
            log.warning("YouTube import degraded: %s", warn)
            try:
                project.add_warning(warn, severity="warning", code="yt_no_deno")
            except Exception:
                pass  # warning is best-effort; never block the import
    dest_stem = project_dir(project.id) / "source"
    settings = get_settings()
    if settings.has_ytdlp:
        dest = _download_ytdlp(url, dest_stem)
    else:
        dest = _download_http(url, dest_stem)
    return _finalize(project, dest, filename=dest.name, url=url)


def _download_ytdlp(url: str, dest_stem: Path) -> Path:
    import yt_dlp

    # Format selection: MERGED (DASH) path first. Putting 'best' or a progressive
    # format first is a trap on YouTube — the only progressive stream with a
    # working URL is usually 360p (format 18), so 'best[height<=1080]' silently
    # resolves to 360p and the higher-res DASH formats never get tried.
    # bv*+ba asks for the best separate video + audio and merges with ffmpeg,
    # which gets 1080p. 'b' is the all-in-one fallback when no merge is possible.
    fmt_merged_first = "bv*[height<=1080]+ba/b[height<=1080]/best/b"
    opts = {
        "outtmpl": str(dest_stem) + ".%(ext)s",
        "format": fmt_merged_first,
        "merge_output_format": "mp4",
        # NOTE: do NOT pin extractor_args.player_client. Forcing android/web
        # triggers YouTube's SABR experiment to strip URLs from the adaptive
        # formats (yt-dlp #12482), collapsing the available set to format 18
        # (360p). Letting yt-dlp negotiate its own clients yields the full set.
        "noplaylist": True,           # never silently grab a whole playlist
        "retries": 5,                 # transient network/HTTP errors
        "fragment_retries": 5,        # DASH/HLS segment fetches
        "concurrent_fragment_downloads": 4,
        "http_chunk_size": 10485760,  # 10 MB — dodges the 503 throttle wall
        # Surface real errors so the UI can show "age-restricted" instead of
        # "didn't work". We keep noprogress to avoid log spam.
        "noprogress": True,
        "no_warnings": False,
        "ignoreerrors": False,
    }
    if get_settings().ffmpeg:
        opts["ffmpeg_location"] = str(Path(get_settings().ffmpeg).parent)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as e:
        # yt-dlp nests the real cause; unwrap it so the caller's error message is
        # actually useful ("Sign in to confirm you're not a bot", "Video unavailable",
        # "Private video", etc.) rather than a bare DownloadError.
        cause = e
        while cause.__cause__ is not None and isinstance(cause.__cause__, Exception):
            cause = cause.__cause__
        msg = str(cause).strip() or str(e)
        raise RuntimeError(f"YouTube/import failed: {msg}") from e
    if not path.exists():  # merged file may carry a different ext than prepare_filename guessed
        cands = sorted(
            (p for p in dest_stem.parent.glob(dest_stem.name + ".*")
             if p.suffix.lower() in VIDEO_EXTS and not p.name.endswith(".part")),
            key=lambda p: p.stat().st_size, reverse=True,
        )
        if not cands:
            raise RuntimeError("yt-dlp produced no output file")
        path = cands[0]  # largest real video file (skip .part fragments)
    return path


def detect_ytdlp_warnings() -> str | None:
    """Return a human-readable warning if the YouTube download path is degraded.

    yt-dlp 2026.x requires a JavaScript runtime (deno) to decode YouTube's player
    JS and enumerate all formats. Without it, downloads still work but the
    available format set collapses to the one stream that needs no JS decoding
    (typically 360p). We surface this proactively so the user knows to install
    deno, rather than getting silent 360p. Returns None when all clear.
    """
    import shutil

    if shutil.which("deno"):
        return None
    # yt-dlp only uses deno for YouTube-family extractors, so the warning is
    # relevant specifically to URL imports.
    return (
        "YouTube imports may be capped at 360p because deno isn't installed. "
        "yt-dlp now needs a JavaScript runtime (deno) to read YouTube's player "
        "and get higher resolutions. Install deno (https://deno.land) and "
        "restart ClipForge for 1080p imports."
    )


def _download_http(url: str, dest_stem: Path) -> Path:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in VIDEO_EXTS:
        ext = ".mp4"
    dest = dest_stem.with_suffix(ext)
    cap = get_settings().upload_cap_bytes
    http_download(url, dest, cap_bytes=cap)
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
        filename=filename, path=rel.as_posix(), url=url,
        duration=info.duration, width=info.width, height=info.height,
        fps=info.fps, size_bytes=path.stat().st_size,
    )
