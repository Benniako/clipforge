"""ClipForge example plugin: custom CSV exporter.

Adds an on_event hook that exports clip metadata as a CSV manifest instead
of video files. Demonstrates the export extension pattern.

Drop this file into the plugins/ directory and restart ClipForge. When a
project is exported, a CSV manifest is written alongside the video clips.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.app.plugin_base import ClipForgePlugin


class CustomExporter(ClipForgePlugin):
    """Exports clip metadata to a CSV manifest."""

    def name(self) -> str:
        return "custom-exporter"

    def version(self) -> str:
        return "0.1.0"

    def on_startup(self) -> None:
        self.log("CustomExporter plugin started — CSV export enabled")

    def on_shutdown(self) -> None:
        self.log("CustomExporter plugin shut down")

    def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Intercept export events to generate a CSV manifest."""
        if event_type == "project.export":
            project = data.get("project")
            output_dir = Path(data.get("output_dir", "."))
            if project is None:
                return
            self._write_csv(project, output_dir)

    def _write_csv(self, project: Any, output_dir: Path) -> None:
        """Write a CSV manifest of all ready clips."""
        rows = []
        for clip in project.clips:
            if clip.status != "ready":
                continue
            rows.append({
                "clip_id": clip.id,
                "title": clip.title or "",
                "score": clip.score or 0,
                "start_sec": round(clip.start, 3),
                "end_sec": round(clip.end, 3),
                "duration_sec": round(clip.duration, 3),
            })

        if not rows:
            self.log("No ready clips to export")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{project.name}_manifest.csv"
        fieldnames = ["clip_id", "title", "score", "start_sec", "end_sec", "duration_sec"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self.log("Exported %d clips → %s", len(rows), csv_path)
