"""Active-speaker detection (LR-ASD / Light-ASD) — optional, audio-visual.

LR-ASD answers "which visible face is speaking?" by fusing audio with face
motion. ClipForge uses it conservatively: when the optional checkout is present
we let it drive talking-head crop keyframes, and when anything is missing or
low-confidence we fall back to the built-in face/Yolo tracker.
"""
from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..media import ffmpeg
from ..models import Word

log = logging.getLogger("clipforge.asd")

ASD_FPS = 25.0
MIN_SPEAKING_SCORE = 0.0


def available() -> bool:
    """True only after the LR-ASD inference adapter is fully wired."""
    return get_settings().has_asd


def _asd_dir() -> Path | None:
    s = get_settings()
    candidates = []
    env = os.environ.get("CLIPFORGE_ASD_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(s.data_dir / "models" / "LR-ASD")
    for path in candidates:
        if (path / "Columbia_test.py").exists() and (path / "weight" / "pretrain_AVA.model").exists():
            return path
    return None


def _add_numpy_int_compat(env: dict[str, str], shim_root: Path) -> None:
    """Give old LR-ASD scripts the NumPy alias removed in NumPy 2.x."""
    try:
        import numpy as np

        needs_shim = not hasattr(np, "int")
    except Exception:
        needs_shim = False
    if not needs_shim:
        return
    shim_root.mkdir(parents=True, exist_ok=True)
    (shim_root / "sitecustomize.py").write_text(
        "try:\n"
        "    import numpy as _np\n"
        "    if not hasattr(_np, 'int'):\n"
        "        _np.int = int\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(shim_root) + (os.pathsep + current if current else "")


def _write_lrasd_runner(tmp_dir: Path) -> Path:
    """Run LR-ASD with in-process compatibility patches for old demos."""
    runner = tmp_dir / "run_lrasd.py"
    runner.write_text(
        "import os\n"
        "import runpy\n"
        "import sys\n"
        "try:\n"
        "    import numpy as _np\n"
        "    if not hasattr(_np, 'int'):\n"
        "        _np.int = int\n"
        "except Exception:\n"
        "    pass\n"
        "sys.path.insert(0, os.getcwd())\n"
        "sys.argv = ['Columbia_test.py', *sys.argv[1:]]\n"
        "runpy.run_path('Columbia_test.py', run_name='__main__')\n",
        encoding="utf-8",
    )
    return runner


def track_centers(src_path: str, start: float, end: float) -> list[tuple[float, float]] | None:
    """Return clip-relative ``(time, cx)`` centers for the active speaking face.

    The upstream LR-ASD entrypoint is a demo script that writes ``tracks.pckl``
    and ``scores.pckl``. This wrapper runs it on a short trimmed clip and parses
    those files into the same center series used by the regular reframe logic.
    """
    if not available():
        return None
    asd_dir = _asd_dir()
    if asd_dir is None:
        return None

    dur = max(end - start, 0.1)
    with tempfile.TemporaryDirectory(prefix="clipforge-asd-") as tmp:
        tmp_dir = Path(tmp)
        clip_path = tmp_dir / "clip.mp4"
        try:
            ffmpeg.run([
                "-ss", f"{max(start, 0.0):.3f}", "-i", src_path,
                "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-b:a", "128k",
                str(clip_path),
            ], timeout=max(180, int(dur * 12)))
        except Exception as e:
            log.warning("LR-ASD trim failed (%s); using fallback reframe", e)
            return None

        runner = _write_lrasd_runner(tmp_dir)
        cmd = [
            sys.executable, str(runner),
            "--videoName", "clip",
            "--videoFolder", str(tmp_dir),
            "--pretrainModel", str(asd_dir / "weight" / "pretrain_AVA.model"),
            "--nDataLoaderThread", str(max(2, min(os.cpu_count() or 4, 8))),
        ]
        env = os.environ.copy()
        if get_settings().ffmpeg:
            env["PATH"] = str(Path(get_settings().ffmpeg).parent) + os.pathsep + env.get("PATH", "")
        _add_numpy_int_compat(env, tmp_dir / "_pycompat")

        from .._util import run_subprocess

        try:
            proc = run_subprocess(
                cmd, cwd=str(asd_dir), env=env, check=False,
                timeout=max(300, int(dur * 30)),
                log_label="LR-ASD",
            )
        except subprocess.TimeoutExpired:
            log.warning("LR-ASD timed out after %.1fs; using fallback reframe", dur)
            return None
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-8:])
            log.warning("LR-ASD failed (%s); using fallback reframe\n%s", proc.returncode, tail)
            return None

        work = tmp_dir / "clip" / "pywork"
        frames = tmp_dir / "clip" / "pyframes"
        try:
            with open(work / "tracks.pckl", "rb") as f:
                tracks = pickle.load(f)
            with open(work / "scores.pckl", "rb") as f:
                scores = pickle.load(f)
            frame_width = _frame_width(frames)
            return _centers_from_tracks(tracks, scores, frame_width)
        except Exception as e:
            log.warning("LR-ASD output parse failed (%s); using fallback reframe", e)
            return None


def _frame_width(frames_dir: Path) -> int:
    import cv2

    first = next(iter(sorted(frames_dir.glob("*.jpg"))), None)
    if first is None:
        return 0
    img = cv2.imread(str(first))
    return int(img.shape[1]) if img is not None else 0


def _as_float_array(values: Any) -> list[float]:
    try:
        import numpy as np

        return [float(v) for v in np.asarray(values, dtype=float).reshape(-1).tolist()]
    except Exception:
        try:
            return [float(v) for v in values]
        except Exception:
            return []


def _score_at(scores: list[float], idx: int, n_frames: int) -> float:
    if not scores:
        return float("-inf")
    if len(scores) == n_frames:
        return scores[min(idx, len(scores) - 1)]
    pos = round(idx * (len(scores) - 1) / max(n_frames - 1, 1))
    return scores[max(0, min(pos, len(scores) - 1))]


def _bbox_center_x(track: dict[str, Any], idx: int, frame_width: int) -> float | None:
    proc = track.get("proc_track") or {}
    xs = proc.get("x", [])
    if xs is None:
        xs = []
    if len(xs) > idx and frame_width > 0:
        return float(xs[idx]) / frame_width

    inner = track.get("track") or track
    boxes = inner.get("bbox", [])
    if boxes is None:
        boxes = []
    if len(boxes) <= idx or frame_width <= 0:
        return None
    box = boxes[idx]
    return (float(box[0]) + float(box[2])) / (2 * frame_width)


def _centers_from_tracks(
    tracks: list[dict[str, Any]],
    scores: list[Any],
    frame_width: int,
    *,
    fps: float = ASD_FPS,
    min_score: float = MIN_SPEAKING_SCORE,
) -> list[tuple[float, float]] | None:
    """Parse LR-ASD pickles into active-speaker crop centers.

    For each frame, keep the visible track with the highest positive speaking
    score. If every score is negative/absent, return ``None`` so the normal
    reframe path handles the clip.
    """
    if frame_width <= 0 or not tracks:
        return None

    by_frame: dict[int, tuple[float, float]] = {}
    for tidx, track in enumerate(tracks):
        inner = track.get("track") or track
        raw_frames = inner.get("frame", [])
        frames = [int(f) for f in _as_float_array(raw_frames if raw_frames is not None else [])]
        if not frames:
            continue
        track_scores = _as_float_array(scores[tidx] if tidx < len(scores) else [])
        for idx, frame in enumerate(frames):
            score = _score_at(track_scores, idx, len(frames))
            if score < min_score:
                continue
            cx = _bbox_center_x(track, idx, frame_width)
            if cx is None:
                continue
            cx = max(0.0, min(1.0, cx))
            prev = by_frame.get(frame)
            if prev is None or score > prev[0]:
                by_frame[frame] = (score, cx)

    if not by_frame:
        return None
    return [(round(frame / fps, 3), cx) for frame, (_score, cx) in sorted(by_frame.items())]


def attribute_speakers(src_path: str, words: list[Word], *,
                       start: float = 0.0, end: float | None = None) -> list[Word]:
    """Keep ASR speaker labels unchanged.

    WhisperX diarization owns stable transcript speaker IDs. LR-ASD tracks
    visible face clips, which are excellent for crop decisions but not yet a
    stable identity map for caption speaker toggles.
    """
    return words
