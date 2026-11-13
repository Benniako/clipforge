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

Measuring detection (no web server, no database):
    python cli.py hud --input video.mp4 --labels-out video.labels.json   # goals read off the scoreboard
    python cli.py eval --labels video.labels.json --min-recall 0.9       # detector vs hand-checked labels

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


def _copy_clips(project: Project, output_dir: Path, *, rename: bool = True,
                name_pattern: str | None = None) -> list[dict]:
    """Copy rendered clips from the media directory to ``output_dir``.

    ``name_pattern`` supports tokens: {rank}, {title}, {score}, {id}, {duration}.
    Default (None) uses format "01_[72]_Title.mp4".

    Returns a list of {path, title, score, duration, factors} for each clip.
    """
    settings = get_settings()
    media_dir = settings.media_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for i, clip in enumerate(project.clips):
        if clip.status != "ready" or not clip.export_url:
            continue
        rel = clip.export_url.lstrip("/media/")
        src = media_dir / rel
        if not src.exists():
            log.warning("Clip %s export missing: %s", clip.id, src)
            continue

        if rename:
            safe_title = "".join(c if c.isalnum() or c in " _-.,()" else "_" for c in (clip.title or ""))
            if name_pattern:
                tokens = {
                    "rank": str(i + 1),
                    "title": safe_title[:60],
                    "score": str(clip.score),
                    "id": clip.id[:8],
                    "duration": f"{clip.duration:.1f}",
                }
                dst_name = name_pattern.format(**tokens) + src.suffix
            else:
                score_str = f"[{clip.score:02d}]" if clip.score else ""
                dst_name = f"{i+1:02d}_{score_str}_{safe_title[:60]}{src.suffix}"
        else:
            dst_name = f"{i+1:02d}_{clip.id[:8]}{src.suffix}"
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
    results = _copy_clips(project, output_dir, rename=not args.no_rename,
                          name_pattern=args.name_pattern)

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


# The label both evaluation detectors are compared on.
EVAL_LABEL = "goal"


def _refine_arg(value: str) -> int | None:
    """``--refine``: ``all`` (no cap) or how many goals get the 5 fps timing pass."""
    if value.strip().lower() == "all":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'all' or a whole number, got {value!r}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected 'all' or a whole number >= 0, got {value!r}")
    return n


def _score_text(score) -> str:
    return f"{score[0]}-{score[1]}"


def _print_text(text: str) -> None:
    """``print``, except that a character the console cannot show is replaced.

    A label note or a file name outside the console's code page (cp1252 when
    PowerShell redirects to a file) would otherwise end the command with a
    traceback and exit code 1 - the code that means "gate not met".
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding))


def _labels_guard(path: Path) -> str | None:
    """Why the label file at ``path`` must not be overwritten (``None``: it may be).

    Verified labels are hours of somebody's checking. A file that cannot be
    parsed is protected as well: it may be a verified one with a typo in it.
    """
    from app import evaluation

    try:
        existing = evaluation.load_labels(path)
    except (OSError, ValueError) as e:
        return f"it exists but cannot be read as a label file ({e})"
    if existing.verified:
        return "it holds verified labels"
    return None


def _hud_table(result, video: Path, duration: float) -> list[str]:
    """The scoreboard tracker's result as text: summary, matches, goals."""
    from app.evaluation import format_hms

    lines = [f"video:     {video}"]
    if result is None:
        lines.append("result:    none (no scoreboard layout for this profile, or the video could not be read)")
        return lines
    side = {"a": "a (first team on the scoreboard)", "b": "b (second team on the scoreboard)"}
    lines += [
        f"layout:    {result.layout}",
        f"healthy:   {'yes' if result.healthy else 'NO - the scoreboard was not found or not read reliably'}",
        f"coverage:  {result.coverage * 100:.1f}% of {format_hms(duration)} with the scoreboard on screen",
        f"streamer:  {side.get(result.streamer_side, 'unknown')}",
        f"samples:   {result.samples} (one every {result.sample_interval:.1f} s)",
        "",
        f"matches ({len(result.matches)}):",
    ]
    if result.matches:
        lines.append(f"  {'#':>3}  {'start':>8}  {'end':>8}  {'final':>5}  {'goals':>5}")
    for k, m in enumerate(result.matches, 1):
        lines.append(f"  {k:>3}  {format_hms(m.t_start):>8}  {format_hms(m.t_end):>8}"
                     f"  {_score_text(m.final_score):>5}  {m.goals:>5}")
    lines += ["", f"goals ({len(result.events)}):"]
    if result.events:
        lines.append(f"  {'time':>8}  {'score':<12}  {'min':>4}  {'team':<4}  {'streamer':<8}"
                     f"  {'importance':>10}  {'confidence':>10}  notes")
    for ev in result.events:
        change = f"{_score_text(ev.score_before)} -> {_score_text(ev.score_after)}"
        minute = f"{ev.minute}'" if ev.minute is not None else "-"
        streamer = {True: "yes", False: "no"}.get(ev.streamer, "unknown")
        notes = ", ".join(n for n, on in (("inferred", ev.inferred), ("scoreboard hidden", ev.hidden),
                                          ("coarse timing", not ev.refined)) if on)
        lines.append(f"  {format_hms(ev.t):>8}  {change:<12}  {minute:>4}  {ev.team:<4}  {streamer:<8}"
                     f"  {ev.importance:>+10.1f}  {ev.confidence:>10.2f}  {notes}".rstrip())
    return lines


def _cmd_killfeed(args: argparse.Namespace, src: Path, info) -> int:
    """`hud` for games with a kill feed (Valorant): list kills, deaths and multi-kills."""
    import dataclasses

    from app.providers.hud import killfeed

    if args.labels_out:
        log.warning("--labels-out is only available for scoreboard games; ignored.")
    from app.providers.hud import segments, valorant_rounds

    # Read only the stretches that show Valorant (a recording can hold several
    # games); without a top bar anywhere, read the whole file.
    spans = segments.valorant_spans(str(src), info)
    if spans:
        result = segments.track_killfeed_spans(str(src), info, spans)
    else:
        result = killfeed.track_killfeed(str(src), info)
    healthy = result is not None and bool(result.healthy)
    rounds, plants = [], []
    if result is not None:
        rounds, plants = valorant_rounds.enrich(str(src), info, result, spans or None)
    if args.json:
        payload = None
        if result is not None:
            payload = {
                "events": [dataclasses.asdict(e) for e in result.events],
                "rounds": [{"index": r.index, "label": r.label, "kills": r.count,
                            "headshots": r.headshots,
                            "t_first": r.t_first, "t_last": r.t_last, "won": r.won,
                            "after_plant": r.after_plant,
                            "clutch": r.clutch.enemies if r.clutch is not None else None}
                           for r in result.rounds],
                "samples": result.samples, "active_s": result.active_s,
                "diagnostics": result.diagnostics,
            }
        print(json.dumps({"video": str(src), "game_profile": args.profile,
                          "duration": float(info.duration), "healthy": healthy,
                          "result": payload}, indent=2, default=str))
        return 0 if healthy else 1

    def clock(t: float) -> str:
        return f"{int(t // 60)}:{int(t % 60):02d}"

    lines = [f"{src.name} - {clock(float(info.duration))} - profile {args.profile}"]
    if result is None:
        lines.append("no result: the video could not be scanned")
    else:
        headshots = sum(1 for k in result.kills if k.headshot)
        lines.append(f"{len(result.kills)} kill(s), {headshots} headshot(s), "
                     f"{len(result.deaths)} death(s), "
                     f"kill feed on screen for {result.active_s:.0f} s"
                     + ("" if healthy else "  [NOT read reliably]"))
        for r in result.rounds:
            times = ", ".join(clock(k.t) + (" (headshot)" if k.headshot else "") for k in r.kills)
            outcome = {True: "round won", False: "round lost", None: ""}[r.won]
            clutch = f"clutch 1v{r.clutch.enemies}" if r.clutch is not None else ""
            notes = ", ".join(n for n in (clutch, outcome, "spike planted" if r.after_plant else "")
                              if n)
            lines.append(f"  round {r.index:2d}  {r.label:<12s} {r.count} kill(s) at {times}"
                         + (f"  ({notes})" if notes else ""))
        if result.deaths:
            lines.append("  deaths: " + ", ".join(clock(d.t) for d in result.deaths))
        if spans:
            lines.append("  Valorant on screen: " + ", ".join(f"{clock(a)}-{clock(b)}" for a, b in spans))
        if rounds:
            lines.append(f"  round results ({len(rounds)}), own team first: " + " ".join(
                f"{clock(r.t)}={r.score[0]}-{r.score[1]}" for r in rounds))
        if plants:
            lines.append(f"  spike planted ({len(plants)}): " + ", ".join(clock(a) for a, _b in plants))
        clutches = result.diagnostics.get("clutches", [])
        if clutches:
            lines.append(f"  last one alive ({len(clutches)}): " + ", ".join(
                f"{clock(c['t'])} 1v{c['enemies']} {'won' if c['won'] else 'lost'}" for c in clutches))
    _print_text("\n".join(lines))
    return 0 if healthy else 1


def cmd_hud(args: argparse.Namespace) -> int:
    """Track a video's scoreboard and print its matches and goals.

    Exit code 0 for a healthy read, 1 when there is no result or the
    scoreboard was not read reliably, 2 when the input cannot be used or a
    label file would be overwritten that must not be.
    """
    import dataclasses

    from app import evaluation
    from app.media.ffmpeg import probe
    from app.providers.hud import track_scoreboard

    src = Path(args.input)
    if not src.is_file():
        log.error("Input not found: %s", args.input)
        return 2
    src = src.resolve()
    labels_out = Path(args.labels_out) if args.labels_out else None
    # Checked before the scan, which can take a while on a long recording.
    if labels_out is not None and labels_out.exists() and not args.force:
        reason = _labels_guard(labels_out)
        if reason:
            log.error("Not overwriting %s: %s. Pass --force to replace it.", labels_out, reason)
            return 2
    try:
        info = probe(str(src))
    except Exception as e:
        log.error("Could not read %s: %s", src, e)
        return 2

    from app.providers.hud import killfeed

    if killfeed.supports(args.profile):
        return _cmd_killfeed(args, src, info)

    result = track_scoreboard(str(src), info, ImportSettings(game_profile=args.profile),
                              refine_limit=args.refine)
    healthy = result is not None and bool(result.healthy)

    written: Path | None = None
    if labels_out is not None and result is not None:
        if not healthy:
            log.warning("The scoreboard was not read reliably; the draft labels will need extra care.")
        draft = evaluation.draft_from_hud(
            result, video=evaluation.video_reference(src, labels_out),
            game_profile=args.profile, duration=float(info.duration))
        try:
            written = evaluation.save_labels(draft, labels_out)
        except (OSError, ValueError) as e:
            log.error("Could not write %s: %s", labels_out, e)
            return 2

    if args.json:
        print(json.dumps({
            "video": str(src),
            "game_profile": args.profile,
            "duration": float(info.duration),
            "healthy": healthy,
            "labels_out": str(written) if written else None,
            "result": dataclasses.asdict(result) if result is not None else None,
        }, indent=2, default=str))
    else:
        _print_text("\n".join(_hud_table(result, src, float(info.duration))))
        if written is not None:
            _print_text(f"\ndraft labels: {written} ({len(draft.events)} events) - check them by eye, "
                        f"then set \"status\" to \"verified\"")
    return 0 if healthy else 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Run a detector on a labelled video and report how it did.

    Exit code 0, 1 when ``--min-precision`` / ``--min-recall`` is given and
    not met, 2 when the label file or its video cannot be used (or the
    detector itself fails, so no numbers exist).
    """
    from app import evaluation
    from app.media.ffmpeg import probe

    if not args.tolerance >= 0:
        log.error("--tolerance must be >= 0, got %s", args.tolerance)
        return 2
    try:
        labels = evaluation.load_labels(args.labels)
    except (OSError, ValueError) as e:
        log.error("Cannot use label file %s: %s", args.labels, e)
        return 2
    video = evaluation.resolve_video(labels, args.labels)
    if not video.is_file():
        log.error("Video not found: %s (named in %s)", video, args.labels)
        return 2
    try:
        info = probe(str(video))
    except Exception as e:
        log.error("Could not read %s: %s", video, e)
        return 2

    warnings: list[str] = []
    if not labels.verified:
        warnings.append("these labels are a DRAFT - the numbers are not trustworthy until a human "
                        "has checked the labels and set \"status\" to \"verified\"")
    clip_metrics = None
    try:
        if args.detector == "hud":
            from app.providers.hud import track_scoreboard

            result = track_scoreboard(str(video), info,
                                      ImportSettings(game_profile=labels.game_profile),
                                      refine_limit=args.refine)
            if result is None:
                warnings.append("the scoreboard tracker returned no result for this video")
            elif not result.healthy:
                warnings.append("the scoreboard was not read reliably (tracker result not healthy)")
            found = result.events if result is not None else []
            predicted = [(float(ev.t), ev.label) for ev in found if ev.label == EVAL_LABEL]
        else:
            from app.providers.detect_gameplay import detect_gameplay

            recorded: list = []
            settings = ImportSettings(game_profile=labels.game_profile,
                                      content_type=ContentType.gameplay)
            clips = detect_gameplay(str(video), info, settings, events_out=recorded)
            predicted = [(float(e.t), e.label) for e in recorded if e.label == EVAL_LABEL]
            clip_metrics = evaluation.score_clips(
                labels.events, [(float(c.start), float(c.end)) for c in clips],
                labelled_ranges=labels.labelled_ranges)
    except Exception:
        log.exception("The %s detector failed on %s", args.detector, video)
        return 2

    metrics = evaluation.score_events(labels.events, predicted, tolerance=args.tolerance,
                                      labelled_ranges=labels.labelled_ranges, labels=(EVAL_LABEL,))
    overall = metrics.overall
    failed: list[str] = []
    if args.min_precision is not None and overall.precision < args.min_precision:
        failed.append(f"precision {overall.precision:.3f} is below the required {args.min_precision:g}")
    if args.min_recall is not None and overall.recall < args.min_recall:
        failed.append(f"recall {overall.recall:.3f} is below the required {args.min_recall:g}")

    title = f"{args.detector} detector vs {labels.status} labels: {video.name}"
    if args.json:
        payload = evaluation.to_json(metrics, clip_metrics, title=title, warnings=warnings)
        payload.update({
            "labels_file": str(Path(args.labels).resolve()),
            "labels_status": labels.status,
            "video": str(video),
            "game_profile": labels.game_profile,
            "detector": args.detector,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            "failed": failed,
            "passed": not failed,
        })
        print(json.dumps(payload, indent=2))
        for w in warnings:
            log.warning("%s", w)
    else:
        _print_text(evaluation.format_report(metrics, clip_metrics, title=title, warnings=warnings))
        for f in failed:
            print(f"FAILED: {f}")
    return 1 if failed else 0


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
    bp.add_argument("--name-pattern", default=None,
                    help="Output filename pattern with {rank}, {title}, {score}, "
                         "{id}, {duration} tokens. "
                         "Example: --name-pattern '{rank:02d}_{score:02d}_{title}'")
    bp.add_argument("--json", action="store_true",
                    help="Output clip metadata as JSON at the end")

    # hud
    hp = sub.add_parser("hud", help="Read the on-screen scoreboard and list every match and goal")
    hp.add_argument("--input", "-i", required=True, help="Video file path")
    hp.add_argument("--profile", default="eafc",
                    help="Game profile that selects the scoreboard layout (default: eafc)")
    hp.add_argument("--refine", type=_refine_arg, default="all", metavar="all|N",
                    help="How many goals get the 5 fps timing pass (default: all)")
    hp.add_argument("--labels-out", default=None, metavar="FILE",
                    help="Write the goals as draft labels for 'eval' (to be checked by a human)")
    hp.add_argument("--force", action="store_true",
                    help="Let --labels-out replace a verified or unreadable label file")
    hp.add_argument("--json", action="store_true",
                    help="Print the result as JSON instead of the table")

    # eval
    ep = sub.add_parser("eval", help="Score a detector against a hand-checked label file")
    ep.add_argument("--labels", required=True, metavar="FILE",
                    help="Label file (JSON); it names the video to run on")
    ep.add_argument("--tolerance", type=float, default=6.0,
                    help="Seconds a detection may be off and still count (default: 6)")
    ep.add_argument("--detector", default="hud", choices=["hud", "gameplay"],
                    help="hud: the scoreboard tracker alone; gameplay: the full gameplay "
                         "detector, which also scores its clips (default: hud)")
    ep.add_argument("--refine", type=_refine_arg, default="all", metavar="all|N",
                    help="hud detector only: how many goals get the 5 fps timing pass (default: all)")
    ep.add_argument("--min-precision", type=float, default=None, metavar="X",
                    help="Exit with code 1 when goal precision is below X (0..1)")
    ep.add_argument("--min-recall", type=float, default=None, metavar="Y",
                    help="Exit with code 1 when goal recall is below Y (0..1)")
    ep.add_argument("--json", action="store_true",
                    help="Print the report as JSON instead of text")

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
    elif args.command == "hud":
        return cmd_hud(args)
    elif args.command == "eval":
        return cmd_eval(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
