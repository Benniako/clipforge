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
                  stack: tuple[int, int, int, int] | None = None,
                  skip_noref: bool = False) -> list[str]:
    x, y, w, h = region
    cmd = [exe, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info"]
    gpu_frames = False
    if keyframes:
        # -discard drops non-key packets in the demuxer (no disk read for
        # them on MP4/MKV); -skip_frame covers containers that ignore it.
        cmd += ["-discard:v", "nokey", "-skip_frame", "nokey"]
    elif hwaccel:
        hw = _hw_args(src)
        cmd += hw
        # With a named NVDEC decoder and a frame-rate reduction, frames stay on
        # the GPU until after the fps filter: only the kept ones are copied to
        # system memory (measured 1.7x faster at 2 fps from 1440p60).
        if fps and "-c:v" in hw:
            cmd += ["-hwaccel_output_format", "cuda"]
            gpu_frames = True
    elif skip_noref:
        # Software decode that skips frames nothing else refers to (B-frames):
        # 1.5x faster, and a sampled frame is at most a few frames late.
        cmd += ["-skip_frame", "noref"]
    if start is not None:
        cmd += ["-ss", f"{max(float(start), 0.0):.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{max(float(duration), 0.05):.3f}"]
    vf = []
    if fps and not keyframes:
        vf.append(f"fps={float(fps):g}")
        if gpu_frames:
            vf.append("hwdownload,format=nv12")
    if stack is None:
        # Cropping first is cheapest in software; GPU frames must be
        # downloaded before they can be cropped.
        vf.insert(len(vf) if gpu_frames else 0, f"crop={w}:{h}:{x}:{y}:exact=1")
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
                stack: tuple[int, int, int, int] | None = None,
                skip_noref: bool = False) -> Iterator[tuple[float, object]]:
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
                            pix_fmt=pix_fmt, stack=stack, skip_noref=skip_noref)
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
            if old_ffmpeg:
                log.info("hud scan: ffmpeg has no -fps_mode, retrying with -vsync")
                continue
            if hwaccel and not keyframes and produced == 0:
                # e.g. a 10-bit source the NVDEC path here cannot hand over.
                log.info("hud scan: GPU decode gave no frames, retrying in software")
                yield from iter_frames(src, region, keyframes=False, fps=fps, start=start,
                                       duration=duration, hwaccel=False,
                                       idle_timeout=idle_timeout, ffmpeg_bin=exe,
                                       pix_fmt=pix_fmt, stack=stack, skip_noref=bool(fps))
                return
            raise


def plan_workers(hwaccel: bool, cpus: int | None = None) -> list[str]:
    """Decoder mix for :func:`map_frames`: ``"gpu"`` and/or ``"cpu"`` entries.

    One GPU decoder is all the card gives (two or four in parallel reach the
    same total, measured on an RTX 5060 Ti), but software decoders add to it:
    one GPU plus two CPU workers read 1440p60 at about 43x realtime against
    26x for the GPU alone.
    """
    import os

    cpus = cpus or os.cpu_count() or 4
    if hwaccel:
        return ["gpu"] + ["cpu"] * max(0, min(3, (cpus - 2) // 4))
    return ["cpu"] * max(1, min(4, cpus // 4))


def map_frames(src, region: tuple[int, int, int, int], fn, *, fps: float,
               start: float = 0.0, duration: float, hwaccel: bool = False,
               pix_fmt: str = "gray", stack=None, chunk_s: float = 60.0,
               workers: list[str] | None = None,
               ffmpeg_bin: str | None = None) -> list[tuple[float, object]]:
    """``[(t, fn(frame)), ...]`` in time order for a dense scan at ``fps``.

    The time range is cut into chunks that several ffmpeg processes decode at
    once (see :func:`plan_workers`); ``fn`` runs on the worker threads, so it
    must not keep state between frames. A chunk the GPU decoder fails on is
    read again in software. Raises :class:`HudScanError` when nothing at all
    could be read.
    """
    import queue

    step = 1.0 / float(fps)
    # Chunks start on the sample grid so every worker samples the same instants.
    chunk = max(step, round(chunk_s / step) * step)
    todo: queue.Queue = queue.Queue()
    n = 0
    while n * chunk < duration - 1e-6:
        todo.put((start + n * chunk, min(chunk, duration - n * chunk)))
        n += 1
    out: dict[int, object] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def read(a: float, length: float, kind: str) -> list[tuple[float, object]]:
        got = []
        for t, frame in iter_frames(src, region, keyframes=False, fps=fps, start=a,
                                    duration=length, hwaccel=kind == "gpu", pix_fmt=pix_fmt,
                                    stack=stack, skip_noref=kind == "cpu", ffmpeg_bin=ffmpeg_bin):
            if t < a + length - 1e-3:
                got.append((t, fn(frame)))
        return got

    def work(kind: str) -> None:
        while True:
            try:
                a, length = todo.get_nowait()
            except queue.Empty:
                return
            try:
                got = read(a, length, kind)
            except HudScanError as e:
                if kind != "gpu":
                    with lock:
                        errors.append(str(e))
                    continue
                log.info("hud scan: GPU decode failed at %.0fs (%s); software for this worker",
                         a, str(e)[-160:])
                kind = "cpu"
                try:
                    got = read(a, length, kind)
                except HudScanError as e2:
                    with lock:
                        errors.append(str(e2))
                    continue
            with lock:
                for t, value in got:
                    out.setdefault(int(round(t / step)), (t, value))

    plan = workers if workers is not None else plan_workers(hwaccel)
    threads = [threading.Thread(target=work, args=(kind,), daemon=True) for kind in plan]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if not out:
        raise HudScanError(errors[0] if errors else "no frames could be read")
    if errors:
        log.warning("hud scan: %d chunk(s) could not be read: %s", len(errors), errors[0][-200:])
    return [out[k] for k in sorted(out)]


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
