#!/usr/bin/env python3
"""Install the user-supplied Valorant cue URLs into ClipForge.

The audio files are stored only in the local ignored runtime data folder:
backend/data/game_cues/valorant/*.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.game_packs import install_cue_from_url, pack_status  # noqa: E402


VALORANT_CUES: dict[str, str] = {
    "spike_plant": "https://www.myinstants.com/de/instant/valorant-spike-plant-60796/",
    "spike_defuse": "https://www.myinstants.com/de/instant/valorant-defuse-83075/",
    "kill": "https://www.myinstants.com/de/instant/valorant-1-kill-44893/",
    "double_kill": "https://www.myinstants.com/de/instant/valorant-2-kills-31547/",
    "triple_kill": "https://www.myinstants.com/de/instant/valorant-3-kills-88885/",
    "quad_kill": "https://www.myinstants.com/de/instant/valorant-4-kills-58625/",
    "ace": "https://www.myinstants.com/de/instant/valorant-5-kills-75541/",
}


def main() -> int:
    failures: list[str] = []
    for event, url in VALORANT_CUES.items():
        try:
            print(f"Installing Valorant cue: {event}")
            install_cue_from_url("valorant", event, url)
        except Exception as exc:
            failures.append(f"{event}: {exc}")

    dest = get_settings().data_dir / "game_cues" / "valorant"
    status = pack_status().get("valorant", {})
    configured = status.get("configured", 0)
    total = status.get("total", 0)
    print(f"\nInstalled {len(VALORANT_CUES) - len(failures)}/{len(VALORANT_CUES)} provided Valorant cues")
    print(f"ClipForge Valorant pack status: {configured}/{total}")
    if total and configured < total:
        print("The remaining standard Valorant slot is optional unless you add that sound too.")
    print(f"Saved in: {dest}")

    if failures:
        print("\nSome cues failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
