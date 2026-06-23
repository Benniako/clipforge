#!/usr/bin/env python3
"""Add a reference game-sound cue to ClipForge from a file or URL.

Usage:
    python scripts/add_cue.py <game> <event> <file-or-url>

Examples:
    python scripts/add_cue.py valorant kill C:/sounds/valo_kill.mp3
    python scripts/add_cue.py valorant kill https://www.myinstants.com/de/instant/valorant-1-kill-44893/

The cue is normalised to 16 kHz mono and saved as
<data>/game_cues/<game>/<event>.wav. ClipForge then matches that event in future
runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.game_packs import PACKS, install_cue, install_cue_from_url  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        print("Known games:", ", ".join(PACKS))
        return 1

    game, event, src = sys.argv[1].lower(), sys.argv[2].lower(), sys.argv[3]

    if game not in PACKS:
        print(f"[!] '{game}' is not a known pack ({', '.join(PACKS)}). Adding anyway.")
    else:
        names = [e[0] for e in PACKS[game]["events"]]
        if event not in names:
            print(f"[!] '{event}' is not a standard {game} event ({', '.join(names)}). Adding anyway.")

    dest = get_settings().data_dir / "game_cues" / game / f"{event}.wav"
    if src.lower().startswith(("http://", "https://")):
        print(f"downloading {src} ...")
        install_cue_from_url(game, event, src)
    else:
        if not Path(src).exists():
            print(f"[X] file not found: {src}")
            return 1
        install_cue(game, event, src)

    print(f"cue added -> {dest}")
    print(f"ClipForge will now match '{event}' in {game} footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
