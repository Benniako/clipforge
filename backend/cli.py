#!/usr/bin/env python3
"""ClipForge CLI — headless batch clipping from the command line.

Processes a video file and outputs ranked, captioned short clips without
starting the web server. The full pipeline runs locally (transcribe → detect →
score → reframe → caption → render).

Usage:
    python -m backend.cli batch --input video.mp4 --output ./clips
    python -m backend.cli batch --input video.mp4 --format tiktok --min-len 15 --max-len 45
    python -m backend.cli batch --input video.mp4 --output ./clips --json  # report as JSON
    python -m backend.cli batch --help

Environment variables (same as the web app):
    CLIPFORGE_WHISPER_MODEL, CLIPFORGE_DEVICE, CLIPFORGE_RENDER_WORKERS, etc.
    See backend/app/config.py for the full list.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure the backend package is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import store
from app.config import get_settings
from app.models import (
    ContentType,
    ImportSettings,
    Platform,
    PowerMode,
    Project,
    ProjectStatus,
    SourceMedia,
    now,
)
from app.pipeline.orchestrator import engine

log = logging.getLogger("clipforge.cli")

# Per-platform short-hand presets.
PLATFORM_PRESETS: dict[str, dict] = {
    "tiktok":  {"platform": "tiktok", "min_len": 15, "max_len": 45, "target_clips": 12, "aspect": "9:16"},
    "reels":   {"platform": "reels",  "min_len": 15, "max_len": 45, "target_clips": 12, "aspect": "9:16"},
    "shorts":  {"platform": "shorts", "min_len": 20, "max_len": 50, "target_clips": 10, "aspect": "9:16"},
    "generic": {"platform": "generic","min_len": 15, "max_len": 60, "target_clips": 10, "aspect": "9:16"},
}


def _resolve_input(path: str) -> str:
    """Resolve a video file path; supports URLs via yt-dlp when available."""
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    # Not a local file — try yt-dlp for URL import.
    try:
        import yt_dlp
        log.info("Downloading %s via yt-dlp…", path)
        out_dir = Path(get_settings().media_dir) / "imports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ydl_opts = {
            "outtmpl": str(out_dir / "%(title).100s_%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(path, download=True)
            fp = ydl.prepare_filename(info)
            # yt-dlp may append a different extension than the template
            actual = next(out_dir.glob(f"{Path(fp).stem}.*"), None)
            if actual and actual.exists():
                return str(actual)
            if Path(fp).exists():
                return str(fp)
    except Exception as e:
        log.error("yt-dlp import failed: %s", e)
    raise FileNotFoundError(f"Input not found: {path}")


def _human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _copy_clips(project: Project, output_dir: Path, *, rename: bool = True) -> list[dict]:
    """Copy rendered clips from the media directory to ``output_dir``.

    Returns a list of {path, title, score, duration, factors} for each clip.
    """
    settings = get_settings()
    media_dir = settings.media_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i, clip in enumerate(project.clips):
        if clip.status != "ready" or not clip.export_url:
            continue
        # Resolve the export URL relative to the media directory.
        # export_url looks like "/media/clips/abc123.mp4"
        rel = clip.export_url.lstrip("/media/")
        src = media_dir / rel
        if not src.exists():
            log.warning("Clip %s export missing: %s", clip.id, src)
            continue

        score_str = f"[{clip.score:02d}]" if clip.score else ""
        if rename and clip.title:
            safe_title = "".join(c if c.isalnum() or c in " _-.,()" else "_" for c in clip.title)
            dst_name = f"{i+1:02d}_{score_str}_{safe_title[:60]}{src.suffix}"
        else:
            dst_name = f"{i+1:02d}_{score_str}_{clip.id[:8]}{src.suffix}"
        dst = output_dir / dst_name

        shutil.copy2(src, dst)
        results.append({
            "file": str(dst),
            "title": clip.title,
            "score": clip.score,
            "duration": round(clip.duration, 1),
            "factors": [{"label": f.label, "weight": f.weight} for f in clip.factors[:3]],
        })
        log.info("  ✔ %s  (%s, score %d)", dst.name, _human_size(dst), clip.score)

    return results


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_batch(args: argparse.Namespace) -> int:
    """Process a video file and export clips to a directory."""
    # Resolve input
    try:
        src_path = _resolve_input(args.input)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1

    # Build settings from presets + overrides
    preset = PLATFORM_PRESETS.get(args.format, PLATFORM_PRESETS["generic"])
    settings = ImportSettings(
        platform=Platform(preset["platform"]),
        power_mode=PowerMode(args.power_mode),
        min_len=args.min_len or preset["min_len"],
        max_len=args.max_len or preset["max_len"],
        target_clips=args.target_clips or preset["target_clips"],
        aspect=args.aspect or preset["aspect"],
        language=args.language or "auto",
        content_type=ContentType(args.content_type),
        burn_captions=not args.no_captions,
        tighten=args.tighten,
        denoise=args.denoise,
    )
    output_dir = Path(args.output) if args.output else Path.cwd() / "clips"

    # Initialise the store and pipeline engine if not already running.
    store.init_db()
    if not engine._started:
        engine.start()

    # Create a project from the source file.
    src_filename = Path(src_path).name
    src_size = Path(src_path).stat().st_size
    src_rel = f"cli/{src_filename}"

    # Copy the source into the media directory so the pipeline can find it.
    media_dir = get_settings().media_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    cli_dir = media_dir / "cli"
    cli_dir.mkdir(exist_ok=True)
    dst_path = cli_dir / src_filename
    shutil.copy2(src_path, dst_path)

    from app.media.ffmpeg import probe
    info = probe(str(dst_path))

    project = Project(
        name=Path(src_path).stem,
        status=ProjectStatus.created,
        settings=settings,
        source=SourceMedia(
            filename=src_filename,
            path=str(dst_path.relative_to(media_dir)),
            duration=info.duration,
            width=info.width,
            height=info.height,
            fps=info.fps,
            size_bytes=src_size,
        ),
        created_at=now(),
        updated_at=now(),
    )
    project = store.save(project)
    log.info("Project %s created from %s", project.id, src_filename)

    # Enqueue and wait.
    engine.enqueue(project.id)
    log.info("Processing…")

    # Poll for completion.
    last_status = ""
    while True:
        p = store.get(project.id)
        if p is None:
            log.error("Project vanished from store")
            return 1
        msg = p.progress.message or p.status.value
        if msg != last_status:
            log.info("  %s", msg)
            last_status = msg
        if p.status == ProjectStatus.ready:
            break
        if p.status == ProjectStatus.failed:
            log.error("Pipeline failed: %s", p.error or "unknown error")
            return 1
        time.sleep(1.0)

    # Copy clips to output directory.
    project = store.get(project.id)  # reload final state
    results = _copy_clips(project, output_dir, rename=not args.no_rename)

    log.info("")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  %d clips exported → %s", len(results), output_dir)
    for r in results:
        log.info("  %s  (score %d, %.1fs)  — %s",
                 Path(r["file"]).name, r["score"], r["duration"], r["title"] or "(no title)")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.json:
        print(json.dumps(results, indent=2))

    # Cleanup source copy (optional, leave for now).
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show environment capabilities and exit."""
    settings = get_settings()
    caps = settings.capability_report()
    print(f"ClipForge v{__import__('app').__version__}")
    print(f"  Data dir:     {settings.data_dir}")
    print(f"  Media dir:    {settings.media_dir}")
    print(f"  DB path:      {settings.db_path}")
    print(f"  ffmpeg:       {caps.get('ffmpeg', 'N/A')}")
    print(f"  ffprobe:      {caps.get('ffprobe', 'N/A')}")
    print(f"  Transcription:{caps.get('transcription', 'N/A')}")
    print(f"  Device:       {caps.get('device', 'N/A')}")
    print(f"  Whisper:      {caps.get('whisper_model', 'N/A')}")
    print(f"  Face tracking:{caps.get('face_tracking', False)}")
    print(f"  OCR:          {caps.get('ocr', 'N/A')}")
    print(f"  GPU encode:   {caps.get('gpu_encode', False)}")
    print(f"  LLM:          {caps.get('llm', False)} ({caps.get('llm_model', '-')})")
    print(f"  VLM:          {caps.get('vlm', False)} ({caps.get('vlm_model', '-')})")
    return 0


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="ClipForge — one long video in, a batch of ranked short clips out.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__import__('app').__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # info
    sub.add_parser("info", help="Show environment capabilities and exit")

    # batch
    bp = sub.add_parser("batch", help="Process a video and export clips")
    bp.add_argument("--input", "-i", required=True,
                    help="Video file path or URL (YouTube/Twitch/etc.)")
    bp.add_argument("--output", "-o", default=None,
                    help="Output directory for clips (default: ./clips/)")
    bp.add_argument("--format", "-f", default="tiktok",
                    choices=list(PLATFORM_PRESETS),
                    help="Platform preset for clip length and aspect (default: tiktok)")
    bp.add_argument("--power-mode", default="balanced",
                    choices=["balanced", "max_gpu", "quality"],
                    help="Compute power mode (default: balanced)")
    bp.add_argument("--min-len", type=float, default=None,
                    help="Minimum clip length in seconds")
    bp.add_argument("--max-len", type=float, default=None,
                    help="Maximum clip length in seconds")
    bp.add_argument("--target-clips", type=int, default=None,
                    help="Target number of clips to generate")
    bp.add_argument("--aspect", default=None,
                    choices=["9:16", "4:5", "1:1", "16:9"],
                    help="Output aspect ratio (default: from platform preset)")
    bp.add_argument("--language", default=None,
                    help="Spoken language hint (de, en, auto)")
    bp.add_argument("--content-type", default="auto",
                    choices=["auto", "talking", "gameplay"],
                    help="Content type detection (default: auto)")
    bp.add_argument("--no-captions", action="store_true",
                    help="Skip caption burn-in (clean clips for NLE editing)")
    bp.add_argument("--tighten", action="store_true",
                    help="Remove silence/dead air (jump cuts)")
    bp.add_argument("--denoise", action="store_true",
                    help="Isolate voice from background music/game audio")
    bp.add_argument("--no-rename", action="store_true",
                    help="Keep original clip IDs as filenames")
    bp.add_argument("--json", action="store_true",
                    help="Output clip metadata as JSON at the end")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return cmd_info(args)
    elif args.command == "batch":
        return cmd_batch(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
