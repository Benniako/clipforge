"""Example ClipForge plugin: a custom hook detector.

This plugin demonstrates the plugin API by logging every pipeline stage and
collecting simple statistics about processed projects.  It is **not** loaded
by default — users copy or symlink it (and any other plugins) into their
``plugins/`` directory.

To enable::

    cp plugins/example_hook_detector.py plugins/hook_detector.py

The plugin is then discovered automatically on the next application start.
"""
from __future__ import annotations

import time
from typing import Any

from backend.app.plugin_base import ClipForgePlugin


class HookDetector(ClipForgePlugin):
    """Tracks pipeline stage timing and logs a summary on completion."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._stage_times: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def name(self) -> str:
        return "hook-detector"

    def version(self) -> str:
        return "0.1.0"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def on_startup(self) -> None:
        self.log("hook-detector plugin started")

    def on_shutdown(self) -> None:
        self.log("hook-detector plugin shut down")

    # ------------------------------------------------------------------ #
    # Pipeline hooks
    # ------------------------------------------------------------------ #
    def before_stage(self, stage: str, project: Any) -> None:
        self._stage_times[stage] = time.time()
        self.log("before stage '%s' for project %s", stage, project.id)

    def after_stage(self, stage: str, project: Any) -> None:
        elapsed = time.time() - self._stage_times.get(stage, time.time())
        self.log("after stage '%s' for project %s (%.2fs)", stage, project.id, elapsed)

    def on_error(self, stage: str, project: Any, error: Exception) -> None:
        self.log("ERROR in stage '%s' for project %s: %s", stage, project.id, error)

    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.log("event '%s': %s", event_type, data)
