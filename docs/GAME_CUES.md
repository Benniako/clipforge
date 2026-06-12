# Game cues — audio & visual (local, no API)

ClipForge can pinpoint **exact** game events (a Valorant kill, an EA FC goal, a
Rocket League goal explosion, a horror jump-scare sting) by matching a short
**reference sound** against your video's audio, and/or a **reference image** of
an on-screen graphic (kill banner, GOAL overlay, death screen) against the video
frames — fully locally, no footage of yours and no API needed. When no cue is
present it falls back to the generic audio-energy detector, so this is purely
additive.

## Easiest: add cues in the app (no command line)
On the import screen, pick a **Game profile** and a **"Pinpoint cues"** panel
appears listing that game's events. For each one, **paste a sound URL**
(e.g. a MyInstants link) or **upload a file** and click *Add* — it's installed
and matched immediately. Remove with the ✕. Events marked 🖼 are **visual cues**:
give them an image instead (crop just the on-screen graphic — the kill banner,
the GOAL overlay — out of a screenshot). Sound vs image is auto-detected from
the file, so the same Add button handles both.

## Quick start with the scripts (alternative)
```bash
python scripts/init_cue_packs.py          # scaffolds folders + a guide per game
python scripts/add_cue.py valorant kill <file-or-url>   # add a sound (file or direct URL)
```
`init_cue_packs` writes a README in each game folder listing the events with a
MyInstants search link. Check progress any time at **GET /api/cues** (also shown
on the import screen next to the Game profile). Supported packs: Valorant, CS2,
EA FC, Rocket League, Horror.

## How it works
You drop reference clips into a folder per game; ClipForge spectrally matches
each one against the audio and anchors a highlight on every match (these take
priority over loudness-only clips and are labelled by the cue, e.g. *"Kill — 3:21"*).
Reference **images** in the `visual/` subfolder are template-matched against the
video frames the same way (multi-scale, so the reference screenshot's resolution
doesn't need to match your footage; needs OpenCV, which face tracking already uses).

```
backend/data/game_cues/
  valorant/      kill.wav   ace.wav   spike_plant.wav
    visual/        kill_banner.png   ace_banner.png
  cs2/           headshot.wav   bomb_plant.wav
    visual/        kill_icon.png
  eafc/          goal.wav   whistle.wav
    visual/        goal_banner.png
  rocketleague/  goal.wav   demolition.wav
  horror/        stinger.wav   scream.wav
  common/        <cues matched for every project>
```
- Folder name = the **Game profile** you pick at import (`auto` → `generic`).
- Any audio format works (`.wav/.mp3/.m4a/.ogg/...`); ~0.5–2 s isolated clips work best.
- Visual cues: `.png/.jpg/.webp/.bmp`, **cropped tight** around the stable UI
  element (no gameplay background) — a banner, an overlay, an icon.
- Put a few variants per event for better recall (e.g. several kill banners).
- When the same moment fires several templates (Valorant's escalating kill-1…ace
  banner tones, or a kill's sound *and* banner), only the **best match wins** —
  so a single kill is never miscounted as a multi-kill.

## Where to get the reference sounds
(Per your own research — all give isolated cues you can drop straight in:)
- **Soundboards** — [MyInstants](https://www.myinstants.com), Voicy: search e.g.
  "Valorant kill", "FC 26 referee whistle", "Rust headshot".
- **SFX packs** — YouTube "Valorant All Kill Sounds" / "CS2 Headshot SFX Pack"
  (creators post free Drive/Mega folders of isolated `.wav`s).
- **Game files** — extract pristine cues from the game with **FModel / UModel**
  (Unreal titles) — best signal-to-noise for matching.

## Or bootstrap from one moment in your own video
If you have a clip where you know a kill/goal happens at, say, 72.5 s:
```bash
python scripts/extract_cue.py myclip.mp4 72.5 backend/data/game_cues/valorant/kill.wav
```
That saves a clean 1.2 s reference you can reuse forever.

## Tuning
- Default match thresholds are conservative (audio 0.62, visual 0.72). If you get
  misses, add more cue variants; if false positives, use a longer/cleaner reference
  (for images: a tighter crop with less background).
- Cues are matched **in addition** to per-game audio profiles
  (Valorant/CS2/EA FC/Rocket League/Horror), so even with zero cue files the
  gameplay detector already finds the loud, high-energy moments.
