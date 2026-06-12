#!/usr/bin/env python3
"""Add a reference game cue to ClipForge (from a file or a direct URL).

Usage:
    python scripts/add_cue.py <game> <event> <file-or-url>

Examples:
    python scripts/add_cue.py valorant kill        C:/sounds/valo_kill.mp3
    python scripts/add_cue.py eafc     goal        https://www.myinstants.com/media/sounds/goal.mp3
    python scripts/add_cue.py valorant kill_banner C:/shots/kill_banner.png

Audio cues come from MyInstants/Voicy, an SFX pack, or FModel; visual cues are
images cropped from a screenshot of the on-screen graphic (see docs/GAME_CUES.md).
A sound is normalised to 16 kHz mono at <data>/game_cues/<game>/<event>.wav, an
image to <data>/game_cues/<game>/visual/<event>.png — ClipForge then finds every
occurrence.
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import game_packs                   # noqa: E402
from app.game_packs import PACKS             # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        print("Known games:", ", ".join(PACKS))
        return 1
    game, event, src = sys.argv[1].lower(), sys.argv[2].lower(), sys.argv[3]

    if game not in PACKS:
        print(f"[!] '{game}' isn't a known pack ({', '.join(PACKS)}). Adding anyway.")
    else:
        names = [e[0] for e in PACKS[game]["events"]] + \
                [e[0] for e in PACKS[game].get("visual_events", [])]
        if event not in names:
            print(f"[!] '{event}' isn't a standard {game} event ({', '.join(names)}). Adding anyway.")

    with tempfile.TemporaryDirectory() as tmp:
        if src.lower().startswith(("http://", "https://")):
            tmp_in = Path(tmp) / "in"
            print(f"downloading {src} …")
            req = urllib.request.Request(src, headers={"User-Agent": "ClipForge/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp_in, "wb") as f:
                f.write(r.read())
            src_path = str(tmp_in)
        else:
            src_path = src
            if not Path(src_path).exists():
                print(f"[X] file not found: {src_path}")
                return 1
        # Sound or image is auto-detected and normalised by the pack installer.
        game_packs.install_cue(game, event, src_path)

    print(f"✓ cue added for {game}/{event}")
    print(f"  ClipForge will now match '{event}' in {game} footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
