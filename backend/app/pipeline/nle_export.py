"""NLE interchange helpers (Premiere/Resolve-friendly EDL)."""
from __future__ import annotations

from pathlib import Path

from ..models import Clip, Project


def _fps(project: Project) -> int:
    fps = float(project.source.fps if project.source else 0.0)
    if fps <= 0:
        return 30
    return max(1, min(int(round(fps)), 120))


def timecode(seconds: float, fps: int) -> str:
    """Non-drop-frame HH:MM:SS:FF timecode."""
    total_frames = max(0, int(round(float(seconds) * fps)))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{frames:02d}"


def ready_clips_for_edl(project: Project) -> list[Clip]:
    return sorted(
        [c for c in project.clips if c.export_url],
        key=lambda c: (-int(c.score), float(c.start), c.id),
    )


def build_cmx3600(project: Project, *, source_file: Path | str | None = None,
                  clips: list[Clip] | None = None) -> str:
    """Build a simple CMX 3600 EDL from source ranges, not rendered files."""
    fps = _fps(project)
    selected = clips if clips is not None else ready_clips_for_edl(project)
    source_name = Path(source_file).name if source_file else (
        project.source.filename if project.source else "source.mp4")
    source_path = str(source_file or (project.source.path if project.source else source_name))
    lines = [
        f"TITLE: {project.name[:48] or 'ClipForge'}",
        "FCM: NON-DROP FRAME",
        "",
    ]
    record_t = 0.0
    for idx, clip in enumerate(selected, 1):
        dur = max(0.0, float(clip.end) - float(clip.start))
        src_in = timecode(clip.start, fps)
        src_out = timecode(clip.end, fps)
        rec_in = timecode(record_t, fps)
        rec_out = timecode(record_t + dur, fps)
        lines.append(
            f"{idx:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append(f"* SOURCE FILE: {source_path}")
        lines.append(f"* COMMENT: {clip.score:02d} - {clip.title[:120]}")
        lines.append("")
        record_t += dur
    return "\n".join(lines).rstrip() + "\n"
