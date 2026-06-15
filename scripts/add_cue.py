#!/usr/bin/env python3
"""Add a reference game-sound cue to ClipForge (from a file or a direct URL).

Usage:
    python scripts/add_cue.py <game> <event> <file-or-url>

Examples:
    python scripts/add_cue.py valorant kill  C:/sounds/valo_kill.mp3
    python scripts/add_cue.py eafc     goal  https://www.myinstants.com/media/sounds/goal.mp3

Grab isolated sounds from MyInstants/Voicy, an SFX pack, or FModel (see
docs/GAME_CUES.md). The cue is normalised to 16 kHz mono and saved as
<data>/game_cues/<game>/<event>.wav — ClipForge then finds every occurrence.
"""
from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings          # noqa: E402
from app.game_packs import PACKS             # noqa: E402
from app.media import ffmpeg                 # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        print("Known games:", ", ".join(PACKS))
        return 1
    game, event, src = sys.argv[1].lower(), sys.argv[2].lower(), sys.argv[3]

    if game not in PACKS:
        print(f"[!] '{game}' isn't a known pack ({', '.join(PACKS)}). Adding anyway.")
    else:
        names = [e[0] for e in PACKS[game]["events"]]
        if event not in names:
            print(f"[!] '{event}' isn't a standard {game} event ({', '.join(names)}). Adding anyway.")

    dest = get_settings().data_dir / "game_cues" / game / f"{event}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)

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
        # Normalise to a clean 16 kHz mono cue.
        ffmpeg.run(["-i", src_path, "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(dest)])

    print(f"✓ cue added -> {dest}")
    print(f"  ClipForge will now match '{event}' in {game} footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
