"""Pre-seeded interactive streamer studio demo project.

Provides instant out-of-the-box interactive testing for creators without
requiring video uploads.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .config import get_settings
from .models import (
    CaptionSet,
    CaptionWord,
    Clip,
    ClipStatus,
    ImportSettings,
    LayoutType,
    Montage,
    Platform,
    Project,
    ProjectStatus,
    Rect,
    Reframe,
    ScoreFactor,
    SourceMedia,
)
from . import store

log = logging.getLogger("clipforge.demo")

DEMO_PROJECT_ID = "proj_demo"


def ensure_demo_media() -> Path:
    """Ensure media/demo_clutch.mp4 exists for previewing interactive controls."""
    settings = get_settings()
    media_dir = settings.media_dir
    media_dir.mkdir(parents=True, exist_ok=True)
    demo_file = media_dir / "demo_clutch.mp4"
    if demo_file.exists() and demo_file.stat().st_size > 1000:
        return demo_file

    if settings.ffmpeg:
        cmd = [
            settings.ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=20:size=1080x1920:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", "20",
            "-c:a", "aac", "-shortest", str(demo_file),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if r.returncode == 0:
                log.info("generated demo media at %s", demo_file)
                return demo_file
        except Exception as exc:
            log.warning("failed to generate demo video with ffmpeg: %s", exc)

    # Fallback dummy file
    if not demo_file.exists():
        demo_file.write_bytes(b"\x00" * 1024)
    return demo_file


def ensure_demo_project() -> Project:
    """Ensure proj_demo is populated in the SQLite store."""
    ensure_demo_media()
    existing = store.get(DEMO_PROJECT_ID)
    if existing and existing.status == ProjectStatus.ready and len(existing.clips) >= 3:
        return existing

    words_ace = [
        CaptionWord(t=0.5, d=0.6, text="WATCH", speaker=0),
        CaptionWord(t=1.1, d=0.5, text="THE", speaker=0),
        CaptionWord(t=1.6, d=0.7, text="FLANK!", speaker=0),
        CaptionWord(t=2.6, d=0.5, text="HE'S", speaker=0),
        CaptionWord(t=3.1, d=0.7, text="DEFUSING", speaker=0),
        CaptionWord(t=4.2, d=0.6, text="ONE", speaker=0),
        CaptionWord(t=4.8, d=0.6, text="TAP!", speaker=0),
        CaptionWord(t=6.0, d=0.8, text="ACE!!!", speaker=0),
        CaptionWord(t=7.2, d=1.2, text="LET'S", speaker=0),
        CaptionWord(t=8.4, d=1.0, text="GOOOOO!", speaker=0),
    ]

    words_fc = [
        CaptionWord(t=0.4, d=0.6, text="CROSS", speaker=0),
        CaptionWord(t=1.0, d=0.5, text="INTO", speaker=0),
        CaptionWord(t=1.5, d=0.6, text="THE", speaker=0),
        CaptionWord(t=2.1, d=0.8, text="BOX...", speaker=0),
        CaptionWord(t=3.2, d=0.7, text="OH", speaker=0),
        CaptionWord(t=3.9, d=0.6, text="MY", speaker=0),
        CaptionWord(t=4.5, d=1.0, text="BICYCLE", speaker=0),
        CaptionWord(t=5.5, d=1.2, text="KICK!", speaker=0),
    ]

    words_talk = [
        CaptionWord(t=0.3, d=0.5, text="IF", speaker=0),
        CaptionWord(t=0.8, d=0.5, text="YOU", speaker=0),
        CaptionWord(t=1.3, d=0.6, text="PLAY", speaker=0),
        CaptionWord(t=1.9, d=0.6, text="RANKED", speaker=0),
        CaptionWord(t=2.8, d=0.7, text="YOU", speaker=0),
        CaptionWord(t=3.5, d=0.8, text="NEED", speaker=0),
        CaptionWord(t=4.3, d=0.9, text="THIS", speaker=0),
        CaptionWord(t=5.2, d=1.1, text="LOADOUT!", speaker=0),
    ]

    p = Project(
        id=DEMO_PROJECT_ID,
        name="🎮 Streamer Studio Demo — Valorant Clutch",
        status=ProjectStatus.ready,
        settings=ImportSettings(
            game_profile="valorant",
            platform=Platform.tiktok,
            aspect="9:16",
            default_style_id="bold-pop",
            tighten=True,
            denoise=False,
            facecam_layout="split",
        ),
        source=SourceMedia(
            filename="demo_clutch.mp4",
            path="demo_clutch.mp4",
            duration=20.0,
            width=1080,
            height=1920,
            fps=30.0,
        ),
        clips=[
            Clip(
                id="c_demo_ace",
                title="🔥 INSANE 1v5 CLUTCH & ACE!",
                description="Unreal round-win clutch with clean crosshair placement and stinger sync.",
                kind="gameplay",
                score=98,
                factors=[
                    ScoreFactor(label="ACE Multi-Kill", weight=40.0, detail="5 kills in 8 seconds"),
                    ScoreFactor(label="Loud Climax", weight=35.0, detail="Scream burst at 6.0s"),
                    ScoreFactor(label="Strong Hook", weight=23.0, detail="Action begins in first 1.5s"),
                ],
                start=0.0,
                end=15.0,
                status=ClipStatus.ready,
                export_url="/media/demo_clutch.mp4",
                captions=CaptionSet(id="cap_demo_ace", words=words_ace, style_id="bold-pop", max_words_per_line=3, lang="en"),
                reframe=Reframe(layout=LayoutType.split, facecam=Rect(x=0.04, y=0.04, w=0.32, h=0.28)),
                features={"climax_t": 6.0, "scream_peak": 6.2, "kill_streak": 5},
                feedback="up",
            ),
            Clip(
                id="c_demo_fc",
                title="⚽ 90+3 BICYCLE KICK WINNER",
                description="Stoppage-time overhead kick winner with stadium crowd roar.",
                kind="gameplay",
                score=95,
                factors=[
                    ScoreFactor(label="Goal Stinger", weight=50.0, detail="Match winning goal at 90+3"),
                    ScoreFactor(label="Crowd Roar", weight=30.0, detail="+18 dB volume spike"),
                    ScoreFactor(label="Delay Snapped", weight=15.0, detail="Lead time adjusted -4.5s"),
                ],
                start=0.0,
                end=12.0,
                status=ClipStatus.ready,
                export_url="/media/demo_clutch.mp4",
                captions=CaptionSet(id="cap_demo_fc", words=words_fc, style_id="kinetic-bounce", max_words_per_line=3, lang="en"),
                reframe=Reframe(layout=LayoutType.framed, facecam=Rect(x=0.62, y=0.04, w=0.34, h=0.26)),
                features={"climax_t": 5.0, "crowd_roar": 5.2},
                feedback="up",
            ),
            Clip(
                id="c_demo_talk",
                title="🎙️ STREAMER REACTION & TIER LIST",
                description="Fast-paced meta analysis with kinetic captions and auto-zoom.",
                kind="talking",
                score=92,
                factors=[
                    ScoreFactor(label="High Engagement", weight=45.0, detail="Punchy speech pacing"),
                    ScoreFactor(label="Viral Hook", weight=30.0, detail="Curiosity gap question in first 3s"),
                    ScoreFactor(label="Zero Dead Air", weight=17.0, detail="Jump cuts tightened"),
                ],
                start=0.0,
                end=14.0,
                status=ClipStatus.ready,
                export_url="/media/demo_clutch.mp4",
                captions=CaptionSet(id="cap_demo_talk", words=words_talk, style_id="hormozi-yellow", max_words_per_line=3, lang="en"),
                reframe=Reframe(layout=LayoutType.center),
                features={"climax_t": 4.5},
                feedback="up",
            ),
        ],
        montages=[
            Montage(
                id="mtg_demo",
                title="🔥 Ultimate Stream Highlights Reel",
                clip_ids=["c_demo_ace", "c_demo_fc", "c_demo_talk"],
                score=99,
                factors=[
                    ScoreFactor(label="Strong opening hook", weight=45.0, detail="Opens on 98/100 ACE"),
                    ScoreFactor(label="High energy pacing", weight=35.0, detail="3 climax peaks across 41s"),
                    ScoreFactor(label="Smooth transitions", weight=19.0, detail="Audio crossfades aligned"),
                ],
                duration=41.0,
                status=ClipStatus.ready,
                export_url="/media/demo_clutch.mp4",
            )
        ],
    )
    store.save(p)
    log.info("seeded demo project: %s", p.id)
    return p
