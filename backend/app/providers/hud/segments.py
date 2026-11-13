"""Which game is on screen when: one recording can hold several games.

A stream session recorded to one file often moves from game to game. Each HUD
tracker must then only judge the stretch that shows its own game:

* the scoreboard tracker does this itself (spans whose score never reads are
  dropped, see :func:`detect._track_layout`) and reports ``game_spans``;
* Valorant is found here by its top bar. Left of the round timer sits a green
  band with the own team's agents, right of it a red band with the enemy's.
  Both are translucent, so the test compares the two sides instead of looking
  for fixed colours.

Measured on a 96-minute session (Valorant, EA FC, Fortnite, 1392 keyframes):
the top-bar test passes on 17-29 of 29 samples per two minutes of Valorant and
on at most 5 of 29 anywhere else, so a per-minute share separates them cleanly.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterator

log = logging.getLogger("clipforge.hud.segments")

_M = 1440.0
TOPBAR_Y = 40 / _M
TOPBAR_H = 56 / _M
TOPBAR_W = 1000 / _M
# The stretch between the last agent portrait and the score plate, as offsets
# from the frame centre: pure band colour, no portrait in the way.
BAND_NEAR = 240 / _M
BAND_FAR = 285 / _M
BAND_MIN_DIFF = 30.0       # (R-G) right minus (R-G) left; Valorant reads 70-90, other games ~0

BIN_S = 60.0
MIN_SHARE = 0.4            # share of a minute's samples that must pass
MIN_SAMPLES = 3
MIN_SPAN_S = 120.0
PAD_S = 45.0               # agent select and the end screen have no top bar
MAX_KEYFRAME_GAP_S = 15.0
FALLBACK_FPS = 0.2

Span = tuple[float, float]
FrameSource = Callable[..., Iterator[tuple[float, object]]]


def topbar_region(width: int, height: int) -> tuple[int, int, int, int]:
    w = min(int(TOPBAR_W * height), width)
    w -= w % 2
    h = max(int(TOPBAR_H * height), 2)
    h -= h % 2
    x = (width - w) // 2
    x -= x % 2
    y = int(TOPBAR_Y * height)
    y -= y % 2
    return x, y, w, h


def topbar_present(bgr, height: int) -> bool:
    """True when ``bgr`` (the :func:`topbar_region` strip) shows Valorant's top bar."""
    import numpy as np

    f = np.asarray(bgr).astype(np.int16)
    rg = f[..., 2] - f[..., 1]
    c = f.shape[1] // 2
    near, far = int(BAND_NEAR * height), int(BAND_FAR * height)
    if far <= near or c - far < 0:
        return False
    left = float(rg[:, c - far:c - near].mean())
    right = float(rg[:, c + near:c + far].mean())
    return right - left >= BAND_MIN_DIFF and left < 5.0 and right > 5.0


def spans_from_flags(samples: list[tuple[float, bool]], duration: float, *,
                     bin_s: float = BIN_S, min_share: float = MIN_SHARE,
                     min_span_s: float = MIN_SPAN_S, pad_s: float = PAD_S) -> list[Span]:
    """Time spans in which most samples are flagged, judged minute by minute."""
    bins: dict[int, list[int]] = {}
    for t, ok in samples:
        b = bins.setdefault(int(t // bin_s), [0, 0])
        b[0] += 1
        b[1] += bool(ok)
    on = sorted(i for i, (n, hits) in bins.items() if n >= MIN_SAMPLES and hits >= min_share * n)
    runs: list[list[int]] = []
    for i in on:
        if runs and i - runs[-1][1] <= 2:          # one weak minute does not end a match
            runs[-1][1] = i
        else:
            runs.append([i, i])
    out: list[Span] = []
    for a, b in runs:
        start, end = a * bin_s, (b + 1) * bin_s
        if end - start >= min_span_s:
            out.append((max(start - pad_s, 0.0), min(end + pad_s, duration)))
    return out


def valorant_spans(src_path: str, info, *, frame_source: FrameSource | None = None
                   ) -> list[Span] | None:
    """Spans of ``src_path`` that show Valorant; ``None`` when it cannot be read.

    One keyframe-only pass over a thin strip: a few seconds for a recording of
    more than an hour.
    """
    width, height = int(getattr(info, "width", 0) or 0), int(getattr(info, "height", 0) or 0)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if width <= 0 or height <= 0 or duration <= 0:
        return None
    started = time.monotonic()
    region = topbar_region(width, height)
    if frame_source is None:
        from . import scan

        def frame_source(reg, **kw):
            return scan.iter_frames(src_path, reg, **kw)

    def read(**kw) -> list[tuple[float, bool]]:
        return [(float(t), topbar_present(fr, height))
                for t, fr in frame_source(region, pix_fmt="bgr24", **kw)]

    try:
        samples = read(keyframes=True)
        gaps = sorted(b[0] - a[0] for a, b in zip(samples, samples[1:]))
        if len(samples) < 3 or gaps[len(gaps) // 2] > MAX_KEYFRAME_GAP_S:
            samples = read(keyframes=False, fps=FALLBACK_FPS)
    except Exception as e:
        log.info("game segments: top-bar scan failed: %s", e)
        return None
    spans = spans_from_flags(samples, duration)
    log.info("game segments: Valorant on screen in %d span(s) %s (%d samples, %.0fs)",
             len(spans), [(round(a), round(b)) for a, b in spans], len(samples),
             time.monotonic() - started)
    return spans


def cluster_spans(spans: list[Span], duration: float, *, bridge_s: float = 600.0,
                  pad_s: float = 180.0) -> list[Span]:
    """Join spans less than ``bridge_s`` apart and pad the result.

    Turns "scoreboard on screen" spans into "this game is being played" spans:
    menus and replays between two matches belong to the game as well.
    """
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= bridge_s:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(max(a - pad_s, 0.0), min(b + pad_s, duration) if duration > 0 else b + pad_s)
            for a, b in out]


def track_killfeed_spans(src_path: str, info, spans: list[Span]):
    """Run the kill tracker on each span and return one combined result."""
    from . import killfeed as K

    parts = []
    for a, b in spans:
        res = K.track_killfeed(src_path, info, start=a, duration=b - a)
        if res is not None:
            parts.append(res)
    if not parts:
        return None
    events = sorted((e for p in parts for e in p.events), key=lambda e: e.t)
    rounds = sorted((r for p in parts for r in p.rounds), key=lambda r: r.t_first)
    for i, rnd in enumerate(rounds, 1):
        rnd.index = i
        for ev in rnd.kills:
            ev.round_index = i
    return K.KillfeedResult(
        events=events, rounds=rounds, samples=sum(p.samples for p in parts),
        sample_interval=parts[0].sample_interval,
        active_s=round(sum(p.active_s for p in parts), 1),
        healthy=any(p.healthy for p in parts),
        diagnostics={
            "spans": [[round(a, 1), round(b, 1)] for a, b in spans],
            "round_starts": [s for p in parts for s in p.diagnostics.get("round_starts", [])],
            "banner_sightings": sum(p.diagnostics.get("banner_sightings", 0) for p in parts),
            "own_row_sightings": sum(p.diagnostics.get("own_row_sightings", 0) for p in parts),
            "headshot_icon_rows": {
                k: sum(p.diagnostics.get("headshot_icon_rows", {}).get(k, 0) for p in parts)
                for k in ("no", "unsure", "yes")},
            "seconds": round(sum(p.diagnostics.get("seconds", 0.0) for p in parts), 1),
        })
