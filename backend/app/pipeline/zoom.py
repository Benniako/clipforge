"""Auto zoom / punch-in — the "alive frame" production-value pass.

A short, subtle zoom on a power word or a scene cut is the single biggest thing
that makes a static talking-head clip feel *edited* rather than *recorded*. The
zoom is centred on the existing reframe crop centre (so the speaker stays
framed) and rides a smooth ease curve: quick punch-in (~150 ms), hold, ease out.

This module produces a time-varying scale factor ``z(t)`` as an ffmpeg
expression, plus a helper that injects it into a crop/scale filter chain. Pure
functions only — the expression math is unit-tested without ffmpeg.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoomSpike:
    """One punch-in event centred at ``t`` with peak scale ``peak``.

    ``duration`` is the full in-out span; the spike is an isosceles triangle
    (linear ease) which is simple to express in ffmpeg and reads as a clean
    "bump" rather than a nauseating lurch. Peak 1.12 = 12% zoom-in.
    """
    t: float
    peak: float = 1.12
    duration: float = 0.45   # ~150ms in, ~150ms hold-ish, ~150ms out


def spikes_from_emphasis(words, *, peak: float = 1.12,
                         duration: float = 0.45,
                         min_gap: float = 0.6) -> list[ZoomSpike]:
    """Build zoom spikes at each emphasised power word.

    Capped by ``min_gap`` so a rapid-fire emphatic line doesn't strobe the zoom
    (two spikes closer than min_gap collapse to the first). Pure.
    """
    spikes: list[ZoomSpike] = []
    for w in words:
        if not getattr(w, "emphasis", False):
            continue
        t = float(getattr(w, "t", 0.0))
        if spikes and t - spikes[-1].t < min_gap:
            continue
        spikes.append(ZoomSpike(t=t, peak=peak, duration=duration))
    return spikes


def spikes_from_cuts(cuts: list[float], *, peak: float = 1.08,
                     duration: float = 0.30) -> list[ZoomSpike]:
    """Build gentler, shorter spikes on scene-cut boundaries.

    Scene cuts already carry motion; a heavy zoom would be too much. Smaller
    peak (8%) and shorter span keeps it as a punctuation mark, not a whip-pan.
    """
    return [ZoomSpike(t=float(t), peak=peak, duration=duration) for t in cuts]


def merge_spikes(*spike_lists: list[ZoomSpike]) -> list[ZoomSpike]:
    """Combine spike sources, sorted by time. Pure."""
    out: list[ZoomSpike] = []
    for lst in spike_lists:
        out.extend(lst)
    out.sort(key=lambda s: s.t)
    return out


def zoom_expr(spikes: list[ZoomSpike], base: float = 1.0) -> str:
    """ffmpeg ``z(t)`` expression: a baseline scale plus a sum of triangle bumps.

    Each spike contributes ``amp * max(0, 1 - 2*|t - t0|/duration)`` — a tent
    function that's 0 outside its window and ``amp`` at its centre. ffmpeg's
    ``if(...)`` nesting builds the piecewise sum. When no spikes are present,
    returns the constant ``base`` so the filter chain is a no-op.
    """
    if not spikes:
        return repr(base)
    expr = repr(base)
    for s in spikes:
        amp = s.peak - base
        if amp <= 0:
            continue
        half = (s.duration / 2.0) or 1e-6
        # tent(t) = amp * max(0, 1 - |t - t0|/half)  — triangular bump centred at t0
        tent = f"{amp:.4f}*max(0,1-abs(t-{s.t:.3f})/{half:.4f})"
        expr = f"({expr})+{tent}"
    return expr


def build_zoom_filter(spikes: list[ZoomSpike], out_w: int, out_h: int,
                      base: float = 1.0) -> str | None:
    """Return a ``scale,crop`` filter pair that realises the time-varying zoom,
    or None when there are no spikes (caller skips the filter entirely).

    Implementation: scale the frame up by z(t) (so the crop centre stays put),
    then crop back to out_w×out_h centred on the scaled frame's centre. ffmpeg
    evaluates z(t) per frame via the ``between``/``if`` expression from zoom_expr.
    """
    if not spikes:
        return None
    z = zoom_expr(spikes, base=base)
    # Over-scale by the max peak so the crop never runs out of pixels.
    max_z = max((s.peak for s in spikes), default=base)
    sw = int(out_w * max_z) | 1   # odd dimensions keep libswscale happy
    sh = int(out_h * max_z) | 1
    return (
        f"scale={sw}:{sh}:eval=frame,"
        f"crop={out_w}:{out_h}:'(iw-{out_w})/2':'(ih-{out_h})/2'"
    )
