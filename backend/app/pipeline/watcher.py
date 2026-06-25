"""Polling-based directory watcher for unattended video processing.

Watches a configurable directory (``CLIPFORGE_WATCH_DIR``) every 10 seconds for
new video files. When one appears (not currently being written), it creates a
ClipForge project, attaches the source, and enqueues it in the pipeline.

This is a minimal, zero-dependency watcher — no inotify, no watchdog. It keeps
the process alive as a daemon thread and logs all activity through the standard
clipforge logger.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from .. import store
from ..config import get_settings
from ..models import ContentType, ImportSettings, Platform, PowerMode, Project, ProjectStatus
from ..pipeline import ingest
from ..pipeline.orchestrator import engine

log = logging.getLogger("clipforge.watcher")


class WatchDirectoryPoller:
    """Poll a directory for new video files and auto-import them.

    Typical usage (started from ``main.py`` lifespan)::

        poller = WatchDirectoryPoller("/path/to/watch")
        poller.start()

    The poller runs as a daemon thread so it exits when the main process does.
    Files that are still being written (detected by size changing between polls)
    are skipped until they stabilise.
    """

    def __init__(self, directory: str | Path, interval: float = 10.0) -> None:
        self._directory = Path(directory)
        self._interval = interval
        self._seen: set[str] = set()
        self._sizes: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Begin polling in a daemon background thread."""
        if self._thread is not None:
            return
        if not self._directory.is_dir():
            log.warning("watch dir %s does not exist — watcher not started",
                        self._directory)
            return
        log.info("watching %s for new videos (poll every %.1fs)",
                 self._directory, self._interval)
        self._thread = threading.Thread(
            target=self._poll_loop, name="clipforge-watcher", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the poller to stop on the next cycle."""
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                log.exception("watcher poll cycle failed")
            self._stop.wait(self._interval)

    def _poll_once(self) -> None:
        if not self._directory.is_dir():
            return
        for entry in self._directory.iterdir():
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in ingest.VIDEO_EXTS:
                continue
            key = entry.name
            size = entry.stat().st_size

            # Skip files that are still growing (being copied / streamed).
            prev = self._sizes.get(key)
            if prev is not None and prev != size:
                self._sizes[key] = size
                continue

            if key in self._seen:
                continue

            if prev is None:
                # First sighting — record size and wait for next cycle.
                self._sizes[key] = size
                continue

            # File is stable and unseen — import it.
            self._seen.add(key)
            self._import_video(entry)

    def _import_video(self, path: Path) -> None:
        """Create a project from a video file and enqueue it."""
        try:
            project = Project(
                name=path.stem[:60],
                status=ProjectStatus.created,
                settings=ImportSettings(
                    platform=Platform.generic,
                    power_mode=PowerMode.balanced,
                    content_type=ContentType.auto,
                ),
            )
            store.save(project)
            src = ingest.attach_source_file(project, path, path.name)
            with store.mutate(project.id) as p:
                p.source = src
                p.name = path.stem[:60]
            engine.enqueue(project.id)
            log.info("watcher imported %s -> project %s (%s)",
                     path.name, project.id, src.filename)
        except Exception:
            log.exception("watcher failed to import %s", path)


def create_watcher() -> WatchDirectoryPoller | None:
    """Factory: return a :class:`WatchDirectoryPoller` if configured, else None.

    Reads ``CLIPFORGE_WATCH_DIR`` from the environment.  Returns ``None`` when
    the env var is unset, empty, or points to a non-existent directory so the
    caller can safely skip watcher startup.
    """
    raw = os.environ.get("CLIPFORGE_WATCH_DIR", "").strip()
    if not raw:
        return None
    watch_path = Path(raw)
    if not watch_path.is_dir():
        log.warning("CLIPFORGE_WATCH_DIR=%s is not a directory — watcher disabled", raw)
        return None
    return WatchDirectoryPoller(watch_path)
