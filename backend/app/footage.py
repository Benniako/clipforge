"""Find the user's recordings so they can be imported by path, without an upload.

Searched folders: the ones in ``CLIPFORGE_FOOTAGE_DIRS``, the recording folders
set in OBS Studio (read from its profile files) and the Videos folder in the
user's home, each up to two folder levels deep (Videos/NVIDIA/<game>/...,
Videos/Source record/normal/...). Newest first.
"""
from __future__ import annotations

import configparser
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

MAX_DEPTH = 2
MAX_FILES = 60
MIN_BYTES = 1 << 20            # smaller files are not recordings worth clipping
WRITING_S = 20.0               # modified this recently: OBS may still be writing it


@dataclass
class Recording:
    path: str
    name: str
    folder: str
    size_bytes: int
    modified: float            # epoch seconds
    in_progress: bool          # changed within the last WRITING_S seconds

    def to_dict(self) -> dict:
        return asdict(self)


def obs_recording_dirs(appdata: Path | None = None) -> list[Path]:
    """Recording folders set in OBS Studio's profiles (simple and advanced output)."""
    base = appdata if appdata is not None else Path(os.environ.get("APPDATA", ""))
    root = base / "obs-studio" / "basic" / "profiles"
    if not root.is_dir():
        return []
    out: list[Path] = []
    for ini in sorted(root.glob("*/basic.ini")):
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            parser.read(ini, encoding="utf-8-sig")
        except (configparser.Error, OSError, UnicodeDecodeError):
            continue
        for section, key in (("SimpleOutput", "FilePath"), ("AdvOut", "RecFilePath")):
            value = parser.get(section, key, fallback="").strip()
            if value:
                # OBS writes Windows paths with escaped backslashes.
                out.append(Path(value.replace("\\\\", "\\")))
    return out


def footage_dirs(extra: str = "", *, home: Path | None = None,
                 appdata: Path | None = None) -> list[Path]:
    """Existing folders to search, each once: ``extra``, OBS's, then Videos."""
    candidates = [Path(p.strip()) for p in extra.split(os.pathsep) if p.strip()]
    candidates += obs_recording_dirs(appdata)
    candidates.append((home if home is not None else Path.home()) / "Videos")
    out: list[Path] = []
    seen: set[str] = set()
    for d in candidates:
        try:
            key = os.path.normcase(str(d.resolve()))
        except OSError:
            continue
        if key in seen or not d.is_dir():
            continue
        seen.add(key)
        out.append(d)
    return out


def _walk(root: Path, max_depth: int) -> Iterator[Path]:
    stack = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    yield Path(entry.path)
                elif (entry.is_dir(follow_symlinks=False) and depth < max_depth
                      and not entry.name.startswith(".")):
                    stack.append((Path(entry.path), depth + 1))
            except OSError:
                continue


def list_recordings(dirs: list[Path], *, max_depth: int = MAX_DEPTH, limit: int = MAX_FILES,
                    now: float | None = None) -> list[Recording]:
    """Video files in ``dirs``, newest first; a file found twice is listed once."""
    from .pipeline.ingest import VIDEO_EXTS

    now = time.time() if now is None else now
    found: dict[str, Recording] = {}
    for root in dirs:
        for path in _walk(root, max_depth):
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            key = os.path.normcase(str(path))
            if key in found:
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size < MIN_BYTES:
                continue
            found[key] = Recording(path=str(path), name=path.name, folder=str(path.parent),
                                   size_bytes=int(st.st_size), modified=float(st.st_mtime),
                                   in_progress=now - st.st_mtime < WRITING_S)
    return sorted(found.values(), key=lambda r: -r.modified)[:limit]
