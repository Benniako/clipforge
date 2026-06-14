"""Scene-cut detection — snap clip boundaries off hard cuts.

A gameplay clip that opens mid-cut (killcam transition, replay wipe, map
change) reads as a glitch. ffmpeg's ``scene`` score finds the hard cuts cheaply;
we snap a clip's start to the nearest cut when one is close, so the clip opens
on a fresh shot instead of half a frame into the previous one.

Talking content is left alone on purpose: its boundaries come from speech
edges, and snapping those to visual cuts could chop a word.
"""
from __future__ import annotations

import logging
import re

from ..config import get_settings
from ..media import ffmpeg

log = logging.getLogger("clipforge.scenes")

# showinfo lines look like: "... n:  3 pts:  12345 pts_time:4.115 ..."
_PTS = re.compile(r"pts_time:\s*(\d+(?:\.\d+)?)")


def parse_showinfo_times(stderr: str) -> list[float]:
    """Frame timestamps from ffmpeg ``showinfo`` filter output (stderr)."""
    return [float(m) for m in _PTS.findall(stderr)]


def scene_cuts(src: str, t0: float, t1: float, *,
               threshold: float = 0.4) -> list[float]:
    """Hard-cut timestamps (absolute source time) inside [t0, t1].

    Uses PySceneDetect's adaptive detector when installed (more robust to camera
    motion / lighting than a raw frame-difference score), falling back to
    ffmpeg's ``scene`` filter otherwise. Decodes only the requested span
    (input-seeked), so probing a few seconds around a boundary is cheap even on
    an hour-long VOD.
    """
    dur = t1 - t0
    if dur <= 0:
        return []
    if get_settings().has_scenedetect:
        cuts = _scenedetect_cuts(src, max(t0, 0.0), dur)
        if cuts is not None:
            return cuts
    err = ffmpeg.run(["-ss", f"{max(t0, 0.0):.3f}", "-i", src, "-t", f"{dur:.3f}",
                      "-vf", f"select='gt(scene,{threshold})',showinfo",
                      "-f", "null", "-"], timeout=120)
    return [round(max(t0, 0.0) + t, 3) for t in parse_showinfo_times(err)]


def _scenedetect_cuts(src: str, t0: float, dur: float) -> list[float] | None:
    """PySceneDetect adaptive-detector cut times (absolute), or None on failure."""
    try:  # pragma: no cover - exercised only when scenedetect is installed
        from scenedetect import AdaptiveDetector, open_video, SceneManager

        video = open_video(src)
        try:
            video.seek(t0)
        except Exception:
            pass
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector())
        sm.detect_scenes(video, end_time=t0 + dur)
        scenes = sm.get_scene_list()
        # Each scene's start (after the first) is a hard cut.
        return [round(s[0].get_seconds(), 3) for s in scenes[1:]]
    except Exception as e:
        log.warning("scenedetect failed (%s); using ffmpeg scene score", e)
        return None


def snap(t: float, cuts: list[float], *, window: float) -> float:
    """Nearest cut to ``t`` within ±``window``, or ``t`` unchanged.

    A cut timestamp is the first frame of the new shot, so starting a clip
    exactly on it opens clean.
    """
    best: float | None = None
    for c in cuts:
        if abs(c - t) <= window and (best is None or abs(c - t) < abs(best - t)):
            best = c
    return best if best is not None else t
