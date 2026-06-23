"""Thin, well-behaved wrappers around ffmpeg and ffprobe.

Everything that shells out to the media engine goes through here so we have a
single place for binary resolution, error surfacing, and logging.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import get_settings


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe exits non-zero. Carries the tail of stderr."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-12:])
        super().__init__(f"ffmpeg exited {returncode}\n{tail}")


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


def probe(path: str | Path) -> MediaInfo:
    """Return basic stream info for a media file.

    Uses ffprobe when available; otherwise parses an ffmpeg ``-i`` probe pass so
    the system still works with only an ffmpeg binary present.
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
    import re

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


def make_thumbnail(src: str | Path, dst: str | Path, *, at: float = 0.0,
                   width: int = 540) -> Path:
    """Grab a single frame at ``at`` seconds, scaled to ``width`` (keeps AR)."""
    dst = Path(dst)
    run(["-ss", f"{max(at, 0):.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", f"scale={width}:-2", str(dst)])
    return dst
