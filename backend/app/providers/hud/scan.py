"""Decode only the scoreboard region of a video — and only the frames needed.

Two modes share one generator:

* **keyframes** — demux and decode keyframes only. A scoreboard holds its
  value for minutes, so one sample every few seconds is plenty, and skipping
  everything else turns a four-hour 1440p recording into a ~15 second read.
* **fps** — a normal decode of a short window at a fixed rate, used to pin
  down the exact moment a digit changed.

Frames come back as ``(seconds, gray_image)`` where the image is just the
requested region. Timestamps are taken from ffmpeg's own ``showinfo`` output
for each frame, never inferred from a frame count, so variable-frame-rate
sources are handled. In fps mode they are exact. In keyframe mode ffmpeg
reports them up to about three frame durations off (it reorders timestamps
without the packets that were skipped): 0.05 s at 60 fps, 0.1 s at 30 fps —
irrelevant for samples seconds apart, and the reason exact timing is always
taken from an fps-mode window.
"""
from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from collections import deque
from typing import Iterator

log = logging.getLogger("clipforge.hud.scan")

_PTS = re.compile(rb"pts_time:\s*(-?[0-9.]+)")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class HudScanError(RuntimeError):
    """ffmpeg could not produce frames for the scoreboard region."""


def _ffmpeg_bin() -> str:
    from ...config import get_settings

    exe = get_settings().ffmpeg
    if not exe:
        raise HudScanError("ffmpeg binary not found in this environment")
    return exe


def build_command(exe: str, src: str, region: tuple[int, int, int, int], *,
                  keyframes: bool, fps: float | None, start: float | None,
                  duration: float | None, hwaccel: bool, passthrough: tuple[str, str],
                  pix_fmt: str = "gray",
                  stack: tuple[int, int, int, int] | None = None) -> list[str]:
    x, y, w, h = region
    cmd = [exe, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
    if keyframes:
        # -discard drops non-key packets in the demuxer (no disk read for
        # them on MP4/MKV); -skip_frame covers containers that ignore it.
        cmd += ["-discard:v", "nokey", "-skip_frame", "nokey"]
    elif hwaccel:
        cmd += _hw_args(src)
    if start is not None:
        cmd += ["-ss", f"{max(float(start), 0.0):.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{max(float(duration), 0.05):.3f}"]
    vf = []
    if fps and not keyframes:
        vf.append(f"fps={float(fps):g}")
    if stack is None:
        vf.insert(0, f"crop={w}:{h}:{x}:{y}:exact=1")
    else:
        # A second region from the same decode, padded to the first one's
        # width and stacked under it: one pass over the video reads both.
        extra = _stack_list(stack)
        n = len(extra) + 1
        taps = "".join(f"[s{i}]" for i in range(n))
        parts = [f"split={n}{taps}", f"[s0]crop={w}:{h}:{x}:{y}:exact=1[c0]"]
        for i, (sx, sy, sw, sh) in enumerate(extra, 1):
            parts.append(f"[s{i}]crop={sw}:{sh}:{sx}:{sy}:exact=1,pad={w}:{sh}:0:0[c{i}]")
        vf.append(";".join(parts) + ";" + "".join(f"[c{i}]" for i in range(n))
                  + f"vstack=inputs={n}")
    vf.append("showinfo")
    cmd += ["-an", "-sn", "-dn", "-vf", ",".join(vf), *passthrough,
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    return cmd


def _stack_list(stack) -> list[tuple[int, int, int, int]]:
    """``stack`` as a list of regions (it may be one region or several)."""
    if not stack:
        return []
    return [tuple(stack)] if isinstance(stack[0], int) else [tuple(s) for s in stack]


def _hw_args(src: str) -> list[str]:
    """GPU decode flags; names the NVDEC decoder when the codec is known."""
    try:
        from ...media import ffmpeg as _ff

        return _ff.hw_decode_args(src) or ["-hwaccel", "cuda"]
    except Exception:
        return ["-hwaccel", "cuda"]


def iter_frames(src, region: tuple[int, int, int, int], *, keyframes: bool = True,
                fps: float | None = None, start: float | None = None,
                duration: float | None = None, hwaccel: bool = False,
                idle_timeout: float = 120.0, ffmpeg_bin: str | None = None,
                pix_fmt: str = "gray",
                stack: tuple[int, int, int, int] | None = None
                ) -> Iterator[tuple[float, object]]:
    """Yield ``(t, gray)`` for the ``(x, y, w, h)`` pixel region of ``src``.

    ``pix_fmt="bgr24"`` yields colour frames of shape ``(h, w, 3)`` instead.
    ``t`` is on the source timeline (seconds from the start of the file).
    Raises :class:`HudScanError` when ffmpeg fails before producing a frame.
    """
    exe = ffmpeg_bin or _ffmpeg_bin()
    # ffmpeg >= 5.1 spells frame passthrough "-fps_mode"; older builds only
    # know "-vsync". Without either, the raw-video muxer pads to a constant
    # frame rate and would duplicate every keyframe a few hundred times.
    for attempt, passthrough in enumerate((("-fps_mode", "passthrough"), ("-vsync", "0"))):
        cmd = build_command(exe, str(src), region, keyframes=keyframes, fps=fps, start=start,
                            duration=duration, hwaccel=hwaccel, passthrough=passthrough,
                            pix_fmt=pix_fmt, stack=stack)
        produced = 0
        out_region = (0, 0, region[2], region[3] + sum(s[3] for s in _stack_list(stack)))
        try:
            for item in _run(cmd, out_region, float(start or 0.0), idle_timeout,
                             channels=3 if pix_fmt == "bgr24" else 1):
                produced += 1
                yield item
            return
        except HudScanError as e:
            old_ffmpeg = attempt == 0 and produced == 0 and "fps_mode" in str(e)
            if not old_ffmpeg:
                raise
            log.info("hud scan: ffmpeg has no -fps_mode, retrying with -vsync")


def _run(cmd: list[str], region: tuple[int, int, int, int], base_t: float,
         idle_timeout: float, channels: int = 1) -> Iterator[tuple[float, object]]:
    import numpy as np

    _x, _y, w, h = region
    frame_bytes = w * h * channels
    shape = (h, w) if channels == 1 else (h, w, channels)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=_NO_WINDOW)
    except OSError as e:
        raise HudScanError(f"could not start ffmpeg: {e}") from e

    pts: deque[float] = deque()
    tail: deque[bytes] = deque(maxlen=25)
    state = {"last": time.monotonic(), "done": False, "killed": False}

    def _drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            m = _PTS.search(line) if b"showinfo" in line else None
            if m:
                pts.append(float(m.group(1)))
            elif line.strip():
                tail.append(line.strip())

    def _watchdog() -> None:
        while not state["done"]:
            time.sleep(1.0)
            if not state["done"] and time.monotonic() - state["last"] > idle_timeout:
                state["killed"] = True
                try:
                    proc.kill()
                except OSError:
                    pass
                return

    reader = threading.Thread(target=_drain_stderr, daemon=True)
    watchdog = threading.Thread(target=_watchdog, daemon=True)
    reader.start()
    watchdog.start()
    produced = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)  # type: ignore[union-attr]
            if not buf or len(buf) < frame_bytes:
                break
            state["last"] = time.monotonic()
            # The showinfo line for a frame is written before the frame leaves
            # the filter graph, but our stderr thread may lag a few ms.
            waited = 0.0
            while not pts and reader.is_alive() and waited < 5.0:
                time.sleep(0.002)
                waited += 0.002
            if not pts:
                raise HudScanError("ffmpeg produced a frame without a timestamp")
            t = base_t + pts.popleft()
            produced += 1
            yield t, np.frombuffer(buf, np.uint8).reshape(shape)
    finally:
        state["done"] = True
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        reader.join(timeout=5)

    if state["killed"]:
        raise HudScanError(f"ffmpeg stalled for more than {idle_timeout:.0f}s and was stopped")
    if produced == 0:
        detail = b" | ".join(tail).decode("utf-8", "replace")[-600:]
        raise HudScanError(f"ffmpeg produced no frames: {detail}")
