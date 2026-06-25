"""ClipForge example plugin: timestamp logger.

Logs every pipeline stage with elapsed timing. Demonstrates the full
ClipForgePlugin API: before_stage, after_stage, on_error, on_event hooks.

Drop this file into the plugins/ directory and restart ClipForge. The plugin
is auto-discovered and logs timing info for every pipeline stage.
"""
from __future__ import annotations

import time
from typing import Any

from backend.app.plugin_base import ClipForgePlugin


class TimestampLogger(ClipForgePlugin):
    """Logs timing info for each pipeline stage."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._stage_times: dict[str, float] = {}

    def name(self) -> str:
        return "timestamp-logger"

    def version(self) -> str:
        return "0.1.0"

    def on_startup(self) -> None:
        self.log("TimestampLogger plugin started")

    def on_shutdown(self) -> None:
        self.log("TimestampLogger plugin shut down")

    def before_stage(self, stage: str, project: Any) -> None:
        self._stage_times[stage] = time.time()
        self.log("BEFORE stage '%s' — project %s", stage, project.id)

    def after_stage(self, stage: str, project: Any) -> None:
        elapsed = time.time() - self._stage_times.get(stage, time.time())
        self.log("AFTER  stage '%s' — project %s (%.2fs)", stage, project.id, elapsed)

    def on_error(self, stage: str, project: Any, error: Exception) -> None:
        self.log("ERROR in stage '%s' — project %s: %s", stage, project.id, error)

    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        self.log("EVENT '%s': %s", event_type, data)
