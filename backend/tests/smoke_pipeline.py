"""End-to-end smoke test: generate a video, run the whole pipeline, verify MP4s.

Runs the real stages (no HTTP) against a synthetic 16:9 source so we can confirm
the pipeline produces valid 9:16 captioned clips. Uses an isolated temp data dir.

    python -m tests.smoke_pipeline
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Isolate storage before importing the app.
_DATA = Path(tempfile.mkdtemp(prefix="clipforge-smoke-"))
os.environ["CLIPFORGE_DATA_DIR"] = str(_DATA)

from app import store  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.media import ffmpeg  # noqa: E402
from app.models import ImportSettings, Platform, Project  # noqa: E402
from app.pipeline import ingest  # noqa: E402
from app.pipeline.orchestrator import engine  # noqa: E402


def make_test_video(path: Path, *, seconds: int = 64, audio: bool = False) -> None:
    ff = get_settings().ffmpeg
    args = [ff, "-hide_banner", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}"]
    vf = ("drawbox=x='mod(t*120\\,1100)':y=260:w=180:h=180:color=red@0.8:t=fill,"
          "drawtext=text='%{pts\\:hms}':x=40:y=40:fontsize=48:fontcolor=white")
    args += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    subprocess.run(args, check=True, capture_output=True)


def main() -> int:
    store.init_db()
    src = _DATA / "test_source.mp4"
    print(f"generating test video -> {src}")
    make_test_video(src, seconds=64, audio=False)
    info = ffmpeg.probe(src)
    print(f"source: {info.width}x{info.height} {info.duration:.1f}s "
          f"audio={info.has_audio} fps={info.fps:.1f}")

    project = Project(name="Smoke Test",
                      settings=ImportSettings(platform=Platform.tiktok,
                                              min_len=15, max_len=45, target_clips=6))
    store.save(project)
    project.source = ingest.attach_source_file(project, src, "test_source.mp4")
    store.save(project)

    print("running pipeline…")
    engine._process(project.id)  # synchronous, in-process

    final = store.get(project.id)
    print(f"\nstatus: {final.status}  clips: {len(final.clips)}  "
          f"transcript: {final.transcript.provider} "
          f"({len(final.transcript.words)} words)")

    ok = True
    media_dir = get_settings().media_dir
    for c in final.clips:
        path = media_dir / c.export_url.removeprefix("/media/") if c.export_url else None
        exists = path.exists() if path else False
        cinfo = ffmpeg.probe(path) if exists else None
        dims = f"{cinfo.width}x{cinfo.height}" if cinfo else "—"
        good = exists and cinfo and cinfo.width == 1080 and cinfo.height == 1920
        ok = ok and good
        reasons = ", ".join(f.label for f in c.factors[:2])
        print(f"  [{ 'OK ' if good else 'BAD'}] score={c.score:3d} {c.duration:5.1f}s "
              f"{dims} reframe={'tracked' if c.reframe.tracked else 'center'} "
              f"| {c.title[:40]!r} | {reasons}")

    assert final.status.value == "ready", f"status={final.status}"
    assert final.clips, "no clips produced"
    assert ok, "one or more clips invalid"
    print("\n✅ SMOKE TEST PASSED")
    print(f"   data dir: {_DATA}")
    return 0


if __name__ == "__main__":
    # Windows consoles default to a legacy code page that can't print "✅".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
