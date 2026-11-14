"""Tests for importing a local video without copying it, and for finding the
user's recordings (``app.pipeline.ingest.place_local_source``, ``app.footage``).

    cd backend && python -m pytest tests/test_import_local.py -q
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

from app import footage  # noqa: E402
from app.api import routes_recordings  # noqa: E402
from app.pipeline import ingest  # noqa: E402


def _video(path: Path, *, size: int = footage.MIN_BYTES + 10, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# --- placing the source -----------------------------------------------------------

def test_a_hard_link_shares_the_file_and_deleting_it_leaves_the_original(tmp_path):
    src = tmp_path / "Videos" / "match.mp4"
    src.parent.mkdir()
    src.write_bytes(b"original recording")
    dest = tmp_path / "project" / "source.mp4"
    dest.parent.mkdir()
    assert ingest.place_local_source(src, dest, mode="link") == "link"
    assert dest.read_bytes() == b"original recording"
    assert os.stat(src).st_nlink == 2                   # one file, two names: no copy
    dest.unlink()                                       # what deleting the project does
    assert src.read_bytes() == b"original recording" and os.stat(src).st_nlink == 1


def test_copy_when_asked_or_when_no_link_is_possible(tmp_path, monkeypatch):
    src = tmp_path / "match.mp4"
    src.write_bytes(b"x" * 100)
    copied = tmp_path / "copied.mp4"
    assert ingest.place_local_source(src, copied, mode="copy") == "copy"
    assert copied.read_bytes() == src.read_bytes() and os.stat(src).st_nlink == 1

    def other_drive(*_a, **_kw):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(ingest.os, "link", other_drive)
    fallback = tmp_path / "fallback.mp4"
    assert ingest.place_local_source(src, fallback, mode="link") == "copy"
    assert fallback.read_bytes() == src.read_bytes()


# --- finding recordings ----------------------------------------------------------

def test_recordings_newest_first_two_levels_deep_and_each_once(tmp_path):
    now = time.time()
    videos = tmp_path / "Videos"
    _video(videos / "obs recording" / "valorant.mp4", mtime=now - 3600)
    _video(videos / "Source record" / "normal" / "eafc.mkv", mtime=now - 60)
    _video(videos / "NVIDIA" / "Game" / "deeper" / "too_deep.mp4", mtime=now)
    _video(videos / "notes.txt", mtime=now)                       # not a video
    _video(videos / "tiny.mp4", size=1000, mtime=now)             # too small to be a recording
    _video(videos / "live.mp4", mtime=now - 5)                    # OBS still writing it
    recs = footage.list_recordings([videos, videos / "obs recording"], now=now)
    assert [r.name for r in recs] == ["live.mp4", "eafc.mkv", "valorant.mp4"]
    assert [r.in_progress for r in recs] == [True, False, False]
    assert recs[1].folder == str(videos / "Source record" / "normal")
    assert footage.list_recordings([tmp_path / "missing"]) == []
    assert len(footage.list_recordings([videos], limit=2, now=now)) == 2


def test_obs_recording_folders_from_both_output_modes(tmp_path):
    profiles = tmp_path / "obs-studio" / "basic" / "profiles"
    (profiles / "Streaming").mkdir(parents=True)
    (profiles / "Streaming" / "basic.ini").write_text(
        "[SimpleOutput]\nFilePath=C:\\\\Users\\\\me\\\\Videos\n\n"
        "[AdvOut]\nRecFilePath=C:/Users/me/Videos/obs recording\n", encoding="utf-8")
    (profiles / "Broken").mkdir()
    (profiles / "Broken" / "basic.ini").write_bytes(b"\xff\xfe no sections here")
    assert footage.obs_recording_dirs(tmp_path) == [
        Path("C:\\Users\\me\\Videos"), Path("C:/Users/me/Videos/obs recording")]
    assert footage.obs_recording_dirs(tmp_path / "no-obs") == []


def test_footage_dirs_existing_ones_each_once(tmp_path):
    home = tmp_path / "home"
    (home / "Videos").mkdir(parents=True)
    extra = tmp_path / "Captures"
    extra.mkdir()
    listed = os.pathsep.join([str(extra), str(tmp_path / "gone"), str(home / "Videos")])
    assert footage.footage_dirs(listed, home=home, appdata=tmp_path / "no-obs") == [
        extra, home / "Videos"]


def test_recordings_route(tmp_path, monkeypatch):
    _video(tmp_path / "match.mp4")
    monkeypatch.setattr(routes_recordings.footage, "footage_dirs", lambda extra: [tmp_path])
    rows = asyncio.run(routes_recordings.list_recordings())
    assert [r["name"] for r in rows] == ["match.mp4"]
    assert set(rows[0]) == {"path", "name", "folder", "size_bytes", "modified", "in_progress"}
