"""Full pipeline on REAL speech: piper TTS -> video -> whisper -> clips.

Verifies the production path (real transcription, real captions) and writes a
sample rendered frame we can eyeball.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
import wave
from pathlib import Path

_DATA = Path(tempfile.mkdtemp(prefix="clipforge-real-"))
os.environ["CLIPFORGE_DATA_DIR"] = str(_DATA)

from app import store  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.media import ffmpeg  # noqa: E402
from app.models import ImportSettings, Platform, Project  # noqa: E402
from app.pipeline import ingest  # noqa: E402
from app.pipeline.orchestrator import engine  # noqa: E402

SCRIPT = (
    "So here is the biggest mistake that almost nobody tells you about when you start. "
    "I learned this the hard way, and honestly it completely changed how I work. "
    "The secret is simple. Start small, stay consistent, and let the results compound. "
    "Most people quit way too early, right before the breakthrough actually happens. "
    "Here is a question for you. What would change if you showed up every single day? "
    "The truth is that motivation fades, but a system you trust will carry you. "
    "I remember the exact moment it clicked for me, and it was honestly incredible. "
    "So do not chase perfect. Chase one percent better, over and over again. "
    "That is the whole game, and the best part is that anyone can do it."
)


def synth_speech(dst: Path) -> None:
    from piper import PiperVoice

    onnx = glob.glob("/tmp/**/en_US-lessac-low.onnx", recursive=True)[0]
    voice = PiperVoice.load(onnx)
    with wave.open(str(dst), "wb") as wf:
        voice.synthesize_wav(SCRIPT, wf)


def main() -> int:
    store.init_db()
    audio = _DATA / "voice.wav"
    print("synthesizing speech…")
    synth_speech(audio)
    adur = ffmpeg.probe(audio).duration
    print(f"speech: {adur:.1f}s")

    src = _DATA / "talk.mp4"
    ff = get_settings().ffmpeg
    subprocess.run([
        ff, "-hide_banner", "-y",
        "-f", "lavfi", "-i", f"color=c=0x1b2a4a:size=1280x720:rate=30:duration={adur:.2f}",
        "-i", str(audio),
        "-vf", "drawtext=text='ClipForge demo':x=(w-text_w)/2:y=(h-text_h)/2:"
               "fontsize=64:fontcolor=white,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-shortest", str(src),
    ], check=True, capture_output=True)

    project = Project(name="Real Demo",
                      settings=ImportSettings(platform=Platform.tiktok,
                                              min_len=12, max_len=40, target_clips=6, language="en"))
    store.save(project)
    project.source = ingest.attach_source_file(project, src, "talk.mp4")
    store.save(project)

    print("running pipeline…")
    engine._process(project.id)
    final = store.get(project.id)
    print(f"\nstatus={final.status} transcript={final.transcript.provider} "
          f"words={len(final.transcript.words)} clips={len(final.clips)}\n")
    # The whole point is the *real* ASR path — a silent fallback to the
    # synthetic transcript must fail loudly, not "pass".
    assert final.transcript.provider != "synthetic", \
        "ASR fell back to the synthetic transcript — is a Whisper model installed?"

    for c in final.clips:
        cap = " ".join(w.text for w in c.captions.words[:10])
        print(f"  score={c.score:3d} {c.duration:5.1f}s  {c.title!r}")
        print(f"       factors: {', '.join(f'{f.label} (+{f.weight})' for f in c.factors)}")
        print(f"       caption: {cap!r}…\n")

    # Export a frame from the top clip for visual inspection.
    top = max(final.clips, key=lambda c: c.score)
    clip_path = get_settings().media_dir / top.export_url.removeprefix("/media/")
    out_frame = Path(__file__).resolve().parents[2] / "assets" / "sample_frame.png"
    out_frame.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.make_thumbnail(clip_path, out_frame, at=top.duration * 0.5, width=1080)
    print(f"sample frame -> {out_frame}")
    print(f"top clip mp4 -> {clip_path}  ({clip_path.stat().st_size} bytes)")
    print("\n✅ REAL PIPELINE PASSED")
    return 0


if __name__ == "__main__":
    import sys
    # Windows consoles default to a legacy code page that can't print "✅".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
