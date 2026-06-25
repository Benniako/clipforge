"""Plugin discovery and loading.

Scans one or more ``plugins/`` directories for modules that contain subclasses
of :class:`ClipForgePlugin`, instantiates them, and returns the list so the
application can invoke their lifecycle hooks.

The loader is intentionally simple:

* It walks a list of directory paths (defaults to a ``plugins/`` folder next
  to the backend package).
* It imports every ``.py`` file that isn't a private helper (name starts with
  ``_``).
* It locates all non-abstract subclasses of ``ClipForgePlugin``, skipping the
  base class itself, and instantiates each with an optional config dict.

Usage::

    from backend.app.plugin_loader import discover_plugins

    plugins = discover_plugins()
    for plugin in plugins:
        plugin.on_startup()
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .plugin_base import ClipForgePlugin

log = logging.getLogger("clipforge.plugin_loader")

# Default search paths (relative to the backend package root).
_DEFAULT_PLUGIN_DIRS = [
    Path(__file__).resolve().parent.parent.parent / "plugins",
]


def discover_plugins(
    directories: list[Path] | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[ClipForgePlugin]:
    """Scan *directories* for plugin modules and return loaded instances.

    :param directories: List of directory paths to scan.  Defaults to
        ``[<project-root>/plugins]``.
    :param configs: Optional mapping of plugin name → config dict.  The
        loader passes the matching dict (or ``{}``) to each plugin's
        constructor.
    :returns: A list of instantiated plugin objects, one per discovered
        subclass.
    """
    dirs = directories or list(_DEFAULT_PLUGIN_DIRS)
    configs = configs or {}
    plugins: list[ClipForgePlugin] = []

    for plugin_dir in dirs:
        resolved = Path(plugin_dir).resolve()
        if not resolved.is_dir():
            log.debug("plugin directory %s not found, skipping", resolved)
            continue
        for entry in sorted(resolved.iterdir()):
            if entry.suffix != ".py":
                continue
            if entry.name.startswith("_"):
                continue
            try:
                mod = _import_module(entry)
            except Exception:
                log.exception("failed to import plugin %s", entry.name)
                continue
            for plugin_cls in _find_plugin_classes(mod):
                try:
                    name = plugin_cls.name  # unbound
                    cfg = configs.get(name, {})
                    inst = plugin_cls(config=cfg)
                    plugins.append(inst)
                    log.info("loaded plugin: %s v%s", inst.name(), inst.version())
                except Exception:
                    log.exception("failed to instantiate plugin %s", plugin_cls.__name__)

    log.info("discovered %d plugin(s)", len(plugins))
    return plugins


def _import_module(path: Path) -> ModuleType:
    """Import a single ``.py`` file as a module.

    Uses ``importlib`` so there is no reliance on ``sys.path`` hacks.
    """
    stem = path.stem
    spec = importlib.util.spec_from_file_location(
        f"clipforge_plugins.{stem}", path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_plugin_classes(module: ModuleType) -> list[type[ClipForgePlugin]]:
    """Return all non-abstract ``ClipForgePlugin`` subclasses in *module*."""
    results: list[type[ClipForgePlugin]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is ClipForgePlugin:
            continue
        if issubclass(obj, ClipForgePlugin) and not inspect.isabstract(obj):
            results.append(obj)
    return results
