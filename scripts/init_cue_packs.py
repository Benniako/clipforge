#!/usr/bin/env python3
"""Scaffold the cue-pack folders with a per-game guide of what sounds to add.

    python scripts/init_cue_packs.py

Creates <data>/game_cues/<game>/ for every supported game and drops a README
listing the events, a MyInstants search link, and the exact add_cue command.
"""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings   # noqa: E402
from app.game_packs import PACKS      # noqa: E402


def main() -> int:
    base = get_settings().data_dir / "game_cues"
    for game, pack in PACKS.items():
        d = base / game
        d.mkdir(parents=True, exist_ok=True)
        lines = [f"# {pack['label']} cues", "",
                 "Add a sound for each event, then ClipForge pinpoints it in your footage.",
                 "Grab isolated sounds from MyInstants / an SFX pack / FModel.", ""]
        for name, desc, hint in pack["events"]:
            q = urllib.parse.quote(hint)
            lines += [
                f"## {name} — {desc}",
                f"- Find: https://www.myinstants.com/en/search/?name={q}",
                f"- Add:  `python scripts/add_cue.py {game} {name} <file-or-url>`",
                "",
            ]
        (d / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"  {pack['label']:<16} -> {d}")
    print(f"\nScaffolded {len(PACKS)} cue packs under {base}")
    print("Add sounds with scripts/add_cue.py, then re-run a gameplay project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
