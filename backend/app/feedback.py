"""Local, no-API learning loop — ClipForge gets better from your feedback.

Two things are learned, both stored in the same SQLite db and applied on the
next run (so it improves across sessions):

1. **Personalised scoring** — which explainable features predict the clips *you*
   keep. We weight the existing named features (hook, emotion, …), so the score
   and its reasons stay transparent. Cold-start safe: learned weights are blended
   with the per-platform defaults by a confidence factor α = n/(n+K), so a few
   ratings barely move anything and trust builds gradually.

2. **Boundary correction** — how far you trim from the detector's raw boundaries.
   We store the detector's *raw* start/end on each clip, learn the median offset
   to where you actually cut, and shift future detections — damped and clamped so
   it converges instead of drifting.

Signals: explicit 👍/👎 (weight 1.0), a trim (boundary sample + 👍 0.5), a
download (👍 0.5). 👎 gives the contrast that makes the learner discriminative.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import threading
from contextlib import contextmanager

from .config import get_settings

_lock = threading.RLock()

# Tunables
_CONF_K = 8.0          # ratings needed before learned weights count for ~half
_POS_ONLY_BASELINE = 0.4   # feature "presence" threshold when only 👍 exist
_LOGREG_MIN = 30.0     # weighted samples (with both classes) before the
                       # logistic learner takes over from mean-difference
_BOUND_MIN = 3         # trims needed before correcting boundaries
_BOUND_DAMP = 0.8      # damping toward the observed median offset
_BOUND_MAX = 3.0       # max seconds we'll shift a boundary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    clip_id  TEXT NOT NULL,
    source   TEXT NOT NULL,            -- 'explicit' | 'download' | 'trim'
    scope    TEXT NOT NULL,
    label    REAL NOT NULL,            -- 1.0 keep, 0.0 reject
    weight   REAL NOT NULL,
    features TEXT NOT NULL,            -- JSON {feature: value}
    ts       REAL NOT NULL,
    PRIMARY KEY (clip_id, source)
);
CREATE INDEX IF NOT EXISTS idx_feedback_scope ON feedback(scope);
CREATE TABLE IF NOT EXISTS trims (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    ds    REAL NOT NULL,
    de    REAL NOT NULL,
    ts    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trims_scope ON trims(scope);
"""


_initialized = False


def init_db() -> None:
    with _connect():
        pass  # schema is ensured on first connect below


@contextmanager
def _connect():
    global _initialized
    con = sqlite3.connect(get_settings().db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    # Double-checked under the lock so two threads racing the first connect
    # (API pool + background worker share this DB) can't let a reader hit a
    # table before the schema DDL runs → "no such table". The DDL is idempotent.
    if not _initialized:
        with _lock:
            if not _initialized:
                con.executescript(_SCHEMA)
                _initialized = True
    try:
        yield con
        con.commit()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Scopes — keep feature sets consistent within a learner
# --------------------------------------------------------------------------- #
def score_scope(content_type: str, platform: str) -> str:
    return f"score:{content_type}:{platform}"


def bound_scope(content_type: str, key: str) -> str:
    return f"bound:{content_type}:{key}"


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record_rating(clip_id: str, scope: str, label: float, features: dict,
                  *, source: str = "explicit", weight: float = 1.0) -> None:
    import time
    with _lock, _connect() as con:
        con.execute(
            "INSERT INTO feedback (clip_id, source, scope, label, weight, features, ts) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(clip_id, source) DO UPDATE SET "
            "scope=excluded.scope, label=excluded.label, weight=excluded.weight, "
            "features=excluded.features, ts=excluded.ts",
            (clip_id, source, scope, float(label), float(weight),
             json.dumps(features or {}), time.time()),
        )


def remove_rating(clip_id: str, source: str = "explicit") -> None:
    with _lock, _connect() as con:
        con.execute("DELETE FROM feedback WHERE clip_id=? AND source=?", (clip_id, source))


def record_trim(scope: str, ds: float, de: float) -> None:
    import time
    # ignore absurd deltas (e.g. a full re-pick) so one outlier can't skew things
    if abs(ds) > 30 or abs(de) > 30:
        return
    with _lock, _connect() as con:
        con.execute("INSERT INTO trims (scope, ds, de, ts) VALUES (?,?,?,?)",
                    (scope, float(ds), float(de), time.time()))


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
def _rows(scope: str):
    with _connect() as con:
        return [(r[0], r[1], json.loads(r[2]))
                for r in con.execute(
                    "SELECT label, weight, features FROM feedback WHERE scope=?", (scope,))]


def _wmean(rows, key: str) -> float | None:
    tot = sum(w for _, w, _ in rows)
    if tot <= 0:
        return None
    return sum(w * float(f.get(key, 0.0)) for _, w, f in rows) / tot


def _logistic_importance(rows, keys: list[str]) -> dict[str, float] | None:
    """L2-regularised logistic regression (deterministic GD); returns positive-
    part weights as importances, or None if numpy is missing or the fit degenerates.

    Upgrades on mean-difference once there is enough data: it accounts for
    correlated features (e.g. hooky clips also being fast-paced) instead of
    crediting every co-occurring feature equally.
    """
    try:
        import numpy as np
    except Exception:
        return None
    X = np.array([[float(f.get(k, 0.0)) for k in keys] for _, _, f in rows])
    y = np.array([1.0 if lbl >= 0.5 else 0.0 for lbl, _, _ in rows])
    sw = np.array([w for _, w, _ in rows])
    w = np.zeros(len(keys))
    b = 0.0
    lam, lr = 0.5, 0.5
    sw_total = max(float(sw.sum()), 1e-9)
    for _ in range(300):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (sw * (p - y))
        # Regularization and gradient share the same weight-total denominator so
        # the L2 strength tracks the *evidence* (sum of sample weights) rather
        # than the raw row count — otherwise mixing 0.5-weight downloads with
        # 1.0-weight explicit ratings silently shifts the regularization.
        gw = X.T @ g / sw_total + lam * w / sw_total
        gb = float(g.sum() / sw_total)
        w -= lr * gw
        b -= lr * gb
    imp = {k: max(float(wi), 0.0) for k, wi in zip(keys, w)}
    return imp if sum(imp.values()) > 1e-6 else None


def learned_weights(scope: str, base: dict[str, float]) -> dict[str, float]:
    """Blend personalised feature weights with the defaults (cold-start safe)."""
    rows = _rows(scope)
    n = sum(w for _, w, _ in rows)
    if n < 1:
        return dict(base)
    pos = [r for r in rows if r[0] >= 0.5]
    neg = [r for r in rows if r[0] < 0.5]

    imp: dict[str, float] | None = None
    if pos and neg and n >= _LOGREG_MIN:  # enough data: proper discriminative fit
        imp = _logistic_importance(rows, list(base))
    if imp is None:
        imp = {}
        if pos and neg:                   # discriminative: kept vs rejected
            for k in base:
                p, q = _wmean(pos, k), _wmean(neg, k)
                imp[k] = max((p or 0.0) - (q or 0.0), 0.0)
        elif pos:                         # positives only: features present in kept
            for k in base:
                imp[k] = max((_wmean(pos, k) or 0.0) - _POS_ONLY_BASELINE, 0.0)
        else:
            return dict(base)

    s = sum(imp.values())
    if s <= 1e-9:
        return dict(base)
    learned = {k: v / s for k, v in imp.items()}

    alpha = n / (n + _CONF_K)
    blended = {k: (1 - alpha) * base[k] + alpha * learned.get(k, 0.0) for k in base}
    # keep the total equal to base so the calibration range is unchanged
    bs, cur = sum(base.values()), sum(blended.values())
    if cur > 0:
        blended = {k: v * bs / cur for k, v in blended.items()}
    return blended


def boundary_correction(scope: str) -> tuple[float, float]:
    """(start_shift, end_shift) seconds to apply to raw detector boundaries."""
    with _connect() as con:
        rows = con.execute("SELECT ds, de FROM trims WHERE scope=?", (scope,)).fetchall()
    if len(rows) < _BOUND_MIN:
        return 0.0, 0.0
    ds = statistics.median(r[0] for r in rows)
    de = statistics.median(r[1] for r in rows)
    clamp = lambda x: max(-_BOUND_MAX, min(_BOUND_MAX, _BOUND_DAMP * x))
    return round(clamp(ds), 3), round(clamp(de), 3)


# --------------------------------------------------------------------------- #
# Transparency / management
# --------------------------------------------------------------------------- #
def overview() -> dict:
    """Summary for the UI: counts + what the scorer has learned per scope."""
    with _connect() as con:
        fb = con.execute("SELECT scope, label, weight FROM feedback").fetchall()
        tr = con.execute("SELECT scope FROM trims").fetchall()
    scopes: dict[str, dict] = {}
    for scope, label, weight in fb:
        s = scopes.setdefault(scope, {"up": 0, "down": 0})
        s["up" if label >= 0.5 else "down"] += 1
    n_pos = sum(1 for _, l, _ in fb if l >= 0.5)
    n_neg = sum(1 for _, l, _ in fb if l < 0.5)

    learned: dict[str, dict] = {}
    from .providers.detect_gameplay import audio_weights
    from .providers.score import BASE_WEIGHTS
    from .models import Platform
    for scope in {s for s, _, _ in fb if s.startswith("score:")}:
        # Gameplay scopes learn over audio features (intensity/sustain/…), not
        # the talking-content features — using the wrong base here made this
        # endpoint report the defaults instead of what was actually learned.
        if scope.startswith("score:gameplay:"):
            base = audio_weights(scope.rsplit(":", 1)[-1])
        else:
            base = BASE_WEIGHTS[Platform.generic]
        w = learned_weights(scope, base)
        top = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:3]
        learned[scope] = {k: round(v, 3) for k, v in top}

    return {
        "total_ratings": len(fb),
        "likes": n_pos,
        "dislikes": n_neg,
        "trims": len(tr),
        "personalized": n_pos + n_neg >= 1,
        "learned_top_features": learned,
    }


def reset(scope: str | None = None) -> None:
    with _lock, _connect() as con:
        if scope:
            con.execute("DELETE FROM feedback WHERE scope=?", (scope,))
            con.execute("DELETE FROM trims WHERE scope=?", (scope,))
        else:
            con.execute("DELETE FROM feedback")
            con.execute("DELETE FROM trims")
