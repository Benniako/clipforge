"""Thin, well-behaved wrappers around ffmpeg and ffprobe.

Everything that shells out to the media engine goes through here so we have a
single place for binary resolution, error surfacing, and logging.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import get_settings


_COMMON_ERRORS: dict[str, tuple[str, str]] = {
    "Invalid data found when processing": (
        "corrupt or non-video file",
        "The input file isn't a valid video. Check it plays in a regular player.",
    ),
    "No such file or directory": (
        "file not found",
        "The input path doesn't exist. Check the file path and try again.",
    ),
    "Unknown encoder": (
        "encoder not available",
        "Your ffmpeg build doesn't include this encoder. Try 'libx264' instead of hardware encoding, "
        "or install a ffmpeg build with the required encoder.",
    ),
    "encoder not found": (
        "encoder not available",
        "Your ffmpeg build doesn't include this encoder. Try 'libx264' instead of hardware encoding, "
        "or install a ffmpeg build with the required encoder.",
    ),
    "Connection refused": (
        "network error",
        "Could not connect to the remote server. Check your internet connection and the URL.",
    ),
    "Protocol not found": (
        "unsupported protocol",
        "The input URL scheme isn't supported. Use a direct video URL or download the file first.",
    ),
    "Error while decoding": (
        "decoding error",
        "ffmpeg couldn't decode a part of this video. The file may be corrupted. Try re-encoding it first.",
    ),
    "Invalid argument": (
        "invalid argument",
        "An ffmpeg flag or value was invalid. This is likely a bug — check the command.",
    ),
    "Permission denied": (
        "permission denied",
        "Can't read the input file or write to the output. Check file permissions.",
    ),
    "codec not supported": (
        "unsupported codec",
        "This video uses a codec ffmpeg can't handle. Try re-encoding to H.264 first.",
    ),
}


def _categorize_ffmpeg_error(stderr: str) -> str:
    """Try to match common ffmpeg errors to human-readable messages."""
    for pattern, (category, hint) in _COMMON_ERRORS.items():
        if pattern.lower() in stderr.lower():
            return f"{category}: {hint}"
    # Generic fallback — try to extract the most relevant line.
    lines = [l.strip() for l in stderr.splitlines() if l.strip()]
    # Skip informational lines, find the actual error.
    for line in lines:
        if any(kw in line.lower() for kw in ("error", "unable", "failed", "cannot")):
            return line[:200]
    # Last resort: show the last non-empty line.
    return (lines[-1][:200] if lines else "unknown ffmpeg error")


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe exits non-zero. Carries a human-readable error."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        self.category = _categorize_ffmpeg_error(stderr)
        # Build a concise message: category + last 3 lines of stderr for context.
        tail_lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()][-3:]
        tail = "\n".join(tail_lines) if tail_lines else "(no details)"
        super().__init__(f"ffmpeg: {self.category}\n{tail}")


@dataclass
class MediaInfo:
    duration: float          # seconds
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool
    codec: str | None

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def _ffmpeg_bin() -> str:
    s = get_settings()
    if not s.ffmpeg:
        raise FFmpegError(["ffmpeg"], 127, "ffmpeg binary not found in this environment")
    return s.ffmpeg


def run(args: list[str], *, timeout: int | None = 600,
        cwd: str | Path | None = None) -> str:
    """Run ffmpeg with the given args (the binary is prepended). Returns stderr.

    ffmpeg writes progress/info to stderr, so callers that need output read it
    from the return value. Raises :class:`FFmpegError` on failure or timeout —
    every call gets a timeout (default 10 min) so a hung ffmpeg on a corrupt or
    network-backed file can never wedge a pipeline worker thread.

    ``cwd`` lets callers run from a directory so filtergraph file references
    (e.g. ``ass=f=cap.ass``) can be bare filenames — avoiding Windows path
    escaping issues with drive-letter colons and backslashes.
    """
    cmd = [_ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(cwd) if cwd else None)
    except subprocess.TimeoutExpired:
        raise FFmpegError(cmd, -1, f"timed out after {timeout}s")
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, proc.stderr)
    return proc.stderr


@lru_cache(maxsize=64)
def probe(path: str | Path) -> MediaInfo:
    """Return basic stream info for a media file.

    Uses ffprobe when available; otherwise parses an ffmpeg ``-i`` probe pass so
    the system still works with only an ffmpeg binary present.

    Cached per path (max 64 entries) — during a project run the same source file
    is probed multiple times (once in _process, again per rerender_all/clips/one).
    Source files are read-only after import, so the cache is always valid.
    """
    s = get_settings()
    path = str(path)
    if s.ffprobe:
        cmd = [
            s.ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            raise FFmpegError(cmd, -1, "ffprobe timed out after 60s")
        if proc.returncode != 0:
            raise FFmpegError(cmd, proc.returncode, proc.stderr)
        return _parse_ffprobe(json.loads(proc.stdout))
    return _probe_with_ffmpeg(path)


def _parse_ffprobe(data: dict) -> MediaInfo:
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration", 0) or 0)
    width = int(v.get("width", 0)) if v else 0
    height = int(v.get("height", 0)) if v else 0
    fps = 0.0
    if v and v.get("avg_frame_rate", "0/0") not in ("0/0", "0"):
        num, _, den = v["avg_frame_rate"].partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
    if v and not duration:
        duration = float(v.get("duration", 0) or 0)
    return MediaInfo(
        duration=duration, width=width, height=height, fps=fps or 30.0,
        has_audio=a is not None, has_video=v is not None,
        codec=v.get("codec_name") if v else None,
    )


def _probe_with_ffmpeg(path: str) -> MediaInfo:
    cmd = [_ffmpeg_bin(), "-hide_banner", "-i", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise FFmpegError(cmd, -1, "ffmpeg probe timed out after 60s")
    text = proc.stderr  # ffmpeg prints stream info to stderr, exits 1 with no output file
    dur = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    w = h = 0
    fps = 30.0
    vm = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", text)
    if vm:
        w, h = int(vm.group(1)), int(vm.group(2))
    fm = re.search(r"(\d+(?:\.\d+)?) fps", text)
    if fm:
        fps = float(fm.group(1))
    return MediaInfo(
        duration=dur, width=w, height=h, fps=fps,
        has_audio="Audio:" in text, has_video="Video:" in text,
        codec=None,
    )


def extract_audio_wav(src: str | Path, dst: str | Path, *, sample_rate: int = 16000) -> Path:
    """Extract mono 16 kHz PCM audio — the format Whisper expects."""
    dst = Path(dst)
    run(["-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(dst)], timeout=1800)  # multi-hour VODs are fine
    return dst


def grab_frame(src: str | Path, dst: str | Path, *, t: float = 0.0,
               width: int | None = None, quality: int | None = None,
               fps: float | None = None, timeout: float = 60.0) -> Path:
    """Extract one frame at time ``t`` seconds, scaled to ``width`` (keeps AR).

    Single source of truth for the "grab a frame" ffmpeg filtergraph that was
    previously inlined ~6× across the pipeline (classify, facecam, reframe,
    detect_ocr, vlm, make_thumbnail). ``quality`` sets JPEG quality (``-q:v``);
    ``fps`` adds an ``fps=`` filter (used by sampling paths that want a fixed
    frame rate before the scale). Returns the destination path.
    """
    dst = Path(dst)

    # Shared LRU frame cache: check before decoding so VLM, facecam, OCR,
    # and reframe all share the same frame bytes instead of running N
    # independent ffmpeg calls for the same timestamp.
    from .frame_cache import get as _cache_get, put as _cache_put

    cached = _cache_get(str(src), t)
    if cached is not None:
        dst.write_bytes(cached)
        return str(dst)

    vf = []
    if fps is not None:
        vf.append(f"fps={fps}")
    if width is not None:
        vf.append(f"scale={width}:-2")
    args = ["-ss", f"{max(t, 0):.3f}", "-i", str(src), "-frames:v", "1"]
    if vf:
        args += ["-vf", ",".join(vf)]
    if quality is not None:
        args += ["-q:v", str(quality)]
    # GPU-accelerated decode for frame extraction: offloads H.264/HEVC decode
    # to NVDEC hardware, reducing CPU load. Only when an NVIDIA GPU is present
    # (the same condition render.py uses for its hwaccel path).
    try:
        from ..config import get_settings
        s = get_settings()
        if s.use_nvenc or s.has_nvidia:
            args = ["-hwaccel", "cuda"] + args
    except Exception:
        pass  # settings not available yet (e.g. during import) — safe fallback
    args.append(str(dst))
    run(args, timeout=timeout)

    # Populate cache so subsequent callers skip the ffmpeg call entirely.
    try:
        _cache_put(str(src), t, dst.read_bytes())
    except Exception:
        pass

    return str(dst)


def make_thumbnail(src: str | Path, dst: str | Path, *, at: float = 0.0,
                   width: int = 540) -> Path:
    """Grab a single frame at ``at`` seconds, scaled to ``width`` (keeps AR)."""
    return grab_frame(src, dst, t=at, width=width)
