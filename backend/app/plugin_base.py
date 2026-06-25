"""Plugin base class for ClipForge extensibility.

Defines the ``ClipForgePlugin`` abstract base that all plugins must subclass.
Plugins can hook into the pipeline lifecycle (before/after each stage) and
react to events.  The plugin system is intentionally minimal — no dependency
injection, no registry — just a clean interface so the loader can discover and
invoke plugins.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger("clipforge.plugin")


class ClipForgePlugin(ABC):
    """Base class for a ClipForge plugin.

    Subclass this and implement the hooks you need.  Every method has a default
    no-op implementation so you only override what is relevant.

    Usage::

        class MyHookDetector(ClipForgePlugin):
            def name(self) -> str:
                return "my-hook-detector"

            def after_stage(self, stage: str, project: Any) -> None:
                if stage == "detect":
                    log.info("detect complete for %s", project.id)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._config_snapshot = dict(self._config)  # immutable view at init

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name (used in logs and error messages)."""
        ...

    def version(self) -> str:
        """Semver string.  Defaults to ``"0.1.0"``."""
        return "0.1.0"

    # ------------------------------------------------------------------ #
    # Lifecycle hooks
    # ------------------------------------------------------------------ #
    def on_startup(self) -> None:
        """Called once when the application starts (after plugin load).
        Override to set up model files, open connections, etc.
        """

    def on_shutdown(self) -> None:
        """Called once during graceful shutdown.
        Override to release resources, flush buffers, etc.
        """

    # ------------------------------------------------------------------ #
    # Pipeline hooks
    # ------------------------------------------------------------------ #
    def before_stage(self, stage: str, project: Any) -> None:
        """Called **before** a pipeline stage runs on a project.

        :param stage: Stage name, e.g. ``"transcribe"``, ``"detect"``,
            ``"score"``, ``"reframe"``, ``"caption"``, ``"render"``.
        :param project: The :class:`~backend.app.models.Project` instance
            being processed (mutate it in place if needed).
        """

    def after_stage(self, stage: str, project: Any) -> None:
        """Called **after** a pipeline stage completes successfully.

        ``project.status`` is still ``"processing"`` unless the stage was
        the final one.

        :param stage: Stage name.
        :param project: The project instance.
        """

    def on_error(self, stage: str, project: Any, error: Exception) -> None:
        """Called when a pipeline stage raises an unhandled exception.

        :param stage: Stage name that failed.
        :param project: The project instance.
        :param error: The exception that was raised.
        """

    # ------------------------------------------------------------------ #
    # Generic event hook
    # ------------------------------------------------------------------ #
    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Generic event bus hook for arbitrary events.

        :param event_type: Dot-separated event name, e.g.
            ``"clip.rated"``, ``"project.created"``.
        :param data: Event-specific payload dict.
        """

    # ------------------------------------------------------------------ #
    # Utilities for subclasses
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> dict[str, Any]:
        """Read-only view of the configuration dict passed at init."""
        return dict(self._config_snapshot)

    def log(self, message: str, level: str = "info") -> None:
        """Convenience logger that prefixes with the plugin name."""
        getattr(log, level, log.info)("[%s] %s", self.name(), message)
