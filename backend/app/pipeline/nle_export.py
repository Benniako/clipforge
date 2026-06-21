"""NLE interchange helpers (Premiere/Resolve-friendly EDL)."""
from __future__ import annotations

from pathlib import Path

from ..models import Clip, Project


def _fps(project: Project) -> int:
    fps = float(project.source.fps if project.source else 0.0)
    if fps <= 0:
        return 30
    return max(1, min(int(round(fps)), 120))


def _frames_to_tc(total_frames: int, fps: int) -> str:
    """Non-drop-frame HH:MM:SS:FF timecode from a whole frame count."""
    total_frames = max(0, int(total_frames))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{frames:02d}"


def timecode(seconds: float, fps: int) -> str:
    """Non-drop-frame HH:MM:SS:FF timecode from seconds."""
    return _frames_to_tc(int(round(float(seconds) * fps)), fps)


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
    # Work in whole frames so each event's record duration equals its source
    # duration exactly — NLEs reject an EDL where the two drift by a frame.
    record_f = 0
    for idx, clip in enumerate(selected, 1):
        src_in_f = max(0, int(round(float(clip.start) * fps)))
        src_out_f = max(src_in_f, int(round(float(clip.end) * fps)))
        dur_f = src_out_f - src_in_f
        src_in = _frames_to_tc(src_in_f, fps)
        src_out = _frames_to_tc(src_out_f, fps)
        rec_in = _frames_to_tc(record_f, fps)
        rec_out = _frames_to_tc(record_f + dur_f, fps)
        lines.append(
            f"{idx:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append(f"* SOURCE FILE: {source_path}")
        lines.append(f"* COMMENT: {clip.score:02d} - {clip.title[:120]}")
        lines.append("")
        record_f += dur_f
    return "\n".join(lines).rstrip() + "\n"
