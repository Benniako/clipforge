# ClipForge: where it stands and what comes next

Status as of 2026-11-13. Measured on an RTX 5060 Ti 16 GB / Ryzen 5 7600 / 31 GB
RAM with real 1440p60 OBS recordings (EA FC 27 and Valorant, German game client).
This file replaces the older roadmap notes in this folder as the current plan.

## 1. What ClipForge can do today

Everything runs on the local machine. No cloud API is called.

| Area | Capability | Engine |
|---|---|---|
| Import | Local files, URLs (yt-dlp), batch CLI, resumable projects; your recordings (Videos, OBS's folders) picked in the app and hard-linked instead of uploaded | ffmpeg 8, yt-dlp |
| Speech | Transcription with word timing and speakers, German and English | whisperX, faster-whisper, pyannote |
| EA FC goals | Reads the scoreboard: matches, every goal, the minute, who scored | `providers/hud` scoreboard tracker |
| Valorant kills | Reads the kill banner and kill feed: own kills, deaths, double / triple / quadra kills and aces per round; round won or lost and spike plants from the top bar | `providers/hud/killfeed.py`, `providers/hud/valorant_rounds.py` |
| Other game events | On-screen keywords (German and English), learned audio cues per game | RapidOCR (EasyOCR, Tesseract as fallbacks), cue matcher |
| Audio | Crowd, cheering, laughter, explosions; free-text audio prompts | PANNs, LAION-CLAP |
| Ranking | Explainable score per clip with its factors; local LLM re-rank and titles; vision second opinion on keyframes | Ollama (`qwen2.5vl:7b`) |
| Framing | 9:16 / 1:1 / 16:9, face and object tracking, facecam layouts, scene-cut snapping; in EA FC the vertical crop follows the ball | YOLO, SCRFD, YuNet, PySceneDetect |
| Captions | Word-level animated captions, game-announcer lines filtered out | ASS via ffmpeg |
| Render | GPU decode and encode, loudness normalisation, platform presets | `h264_cuvid`, `h264_nvenc` |
| Editor | Timeline trim, caption edit, re-render, feedback that re-weights scoring | React + FastAPI |
| Evaluation | Hand-checked label files and `cli.py eval` to score a detector | `app/evaluation.py` |

### Measured results

| Test | Result |
|---|---|
| EA FC 27, 16.6 min | 1 match, 9 of 9 goals, scoreboard coverage 83 %, tracker 13 s, whole pipeline 3 min 58 s for 6 clips |
| Valorant, 40.7 min | 8 of 8 own kills found, 0 false kills, 1 double kill, 20 deaths; checked frame by frame |
| Mixed session, 96.6 min (Valorant, EA FC, Fortnite in one file) | Split into Valorant 2:15-17:45 and EA FC 17:29-38:08; EA FC read healthy (1 match, 6 goals); Valorant part: 0 kills, 8 deaths (no kill banner appeared; the team score stayed at 0) |
| Tests | 247 unit tests, 179 pytest tests, 274 frontend tests, lint clean |

### How the Valorant tracker decides

1. **Kill banner** (white skull in a ring, bottom centre, about 2 s per kill) is the proof of a kill. It has no text, so the game language does not matter.
2. **Kill feed** (top right) confirms it: the streamer's own row carries a yellow tab left of the killer name; their death carries a yellow outline at the victim end.
3. **Buy-phase plate** ("KAUFPHASE" in the German client; read by shape, not by text) marks round starts, which groups kills into rounds and labels multi-kills.
4. The banner also shows while spectating a teammate after dying. A banner after the streamer's death in the same round is dropped unless the feed confirms it.

Run it alone with `python cli.py hud --profile valorant --input <video>`.

## 2. Is the current build at its limit?

No. The measurements point to these gaps, largest first:

| Gap | Evidence | Fix |
|---|---|---|
| Each detector decodes the video again | Scoreboard, kill feed, OCR, scene detection and the facecam search each start their own ffmpeg. The kill-feed scan itself is now parallel (48 s for 40 min, was 182 s) | Put the remaining detectors on the same pass (rest of R1) |
| Recordings with several games | HUD trackers now run per game stretch (R2 done); audio cues and keyword OCR still use one profile per file | Per-stretch profiles for the remaining detectors |
| GPU is mostly idle | Average GPU load 20 % over a full run | Follows from R1; batch OCR and VLM frames |
| Source is copied into the data folder | Recordings picked in the app are now hard-linked (R5); a file dropped into the browser is still uploaded, and the CLI still copies | Hard-link in the CLI batch import too |
| Football framing | The vertical crop now follows the ball (R4); clips with a facecam layout still show a fixed game pane | Ball-follow inside the facecam layouts |
| PaddleOCR does not load | It needs `opencv-contrib-python`, which conflicts with OpenCV 5. No longer needed: RapidOCR (R6) runs the same PP-OCR models on onnxruntime | Remove PaddleOCR from the extras once nothing else uses it |
| LLM cold start | Fixed (R7): models load before a batch's time budget and the vision model loads in the background a stage early; from a cold start the VLM batch now scores 6 of 6 clips instead of 0 | — |

## 3. Roadmap

### Next (highest value, each about a day or less)

- **R1 Fast dense scan: done (2026-11-13).** The Valorant scan of the 40.7-min recording went from 182 s to 48 s with identical kills. `hud/scan.map_frames` cuts the video into chunks read by one GPU decoder plus CPU decoders at once (the GPU alone tops out at 26x realtime; with two CPU workers it is 43x), frames stay on the GPU until after frame dropping, and CPU decoders skip non-reference frames. Still open under R1: OCR, scene detection and the facecam search each decode on their own; moving them onto the same pass is the remaining part.
- **R2 Game segmentation: done (2026-11-13).** `hud/segments.py` finds the Valorant stretches of a recording by the top bar (green team band left, red right; a 5 s keyframe pass for 96 min), and the scoreboard tracker drops stretches whose score never reads. Both trackers now run on every gameplay recording whatever the profile, each only on its own stretches, and the EA FC "only during a match" gate no longer mutes other games. On the 96.6-min session: Valorant 2:15-17:45, EA FC 17:29-38:08, scoreboard healthy with 6 of 6 goals (was flagged unhealthy). Still open: audio cues, keyword OCR and scoring weights use one profile for the whole file, and Fortnite has no tracker of its own.
- **R3 More Valorant events: round results and spike plants done (2026-11-13).** `hud/valorant_rounds.py` reads both team scores off the top bar on every keyframe and decodes them by the rule that a score never falls in a match (the longest series of reads that never falls is kept; 40-60 % of reads on the translucent plates come back empty and a few gain or lose a digit). On the 40.7-min match all 20 score changes up to 8-12 decode in order and 10 spike plants are found, in about 8 s. Kill rounds are marked won or lost (+2 on the clip score) and kills with the spike planted get +1; round wins, losses and plants are stored as events. **Clutches** (done the same day): the top bar drops a dead player's portrait and packs the rest towards the centre, so the alive count per team is the number of filled slots (edge density 80-200 against 5-30 for the empty band). The streamer is the last one alive when one own portrait is left on two samples in a row and they have not died since the whole team was last alive (the buy-phase plate cannot mark rounds here: the round-end banner makes it fire 8-26 s early). A won clutch scores 88-98 by the number of enemies (1v1 to 1v5), its clip runs from being left alone to the round's end, and a clutch won without a kill gets a clip of its own. On the 40.7-min match: one 1v5 (lost, with a double kill); every other one-left moment was a teammate after the streamer's death, checked round by round. Still open: headshot icon in the feed row, spike defuse.
- **R4 Ball-follow framing for EA FC: done (2026-11-13).** `pipeline/ballcam.py` finds the ball with YOLO's "sports ball" class at a 1920-px input (the ball is about 8 px wide at 1440p with the zoomed-out camera), masks the HUD (radar, name bars, scoreboard), and picks the most confident path through the detections that a ball can actually travel, with camera cuts at a fixed cost. A dead zone, One-Euro smoothing and a pan speed limit turn that path into crop keyframes for the existing renderer. On three goal sequences of the 16.6-min recording the ball was found in 81-96 % of samples and was inside the 9:16 crop on every one of them; about 5.5 s per 26-s clip at 4 samples a second. Clips where the ball is seen too rarely keep the old motion-centre crop. 2026-11-14: the crop no longer pans over to something else for a moment and straight back. Flat boxes are skipped (the penalty spot read as a ball at up to 0.79, but seen from the raised camera it is twice as wide as tall), and a target the track holds for 0.5 s or less beside the ball is dropped unless the detector is sure of it (the shot into the net just before the cut). On the nine goals of the recording: 5 such pans instead of 61, and sure ball detections inside the crop 99.8 % of the time (99.4 % before). Not done: following the ball inside the facecam layouts, and a blurred-background 16:9 layout for wide plays.

### Soon

- **R5 No-copy import: done (2026-11-14).** The import screen lists "Deine Aufnahmen" from the Videos folder (two levels deep: OBS, Source Record, NVIDIA), the recording folders set in OBS's profiles and `CLIPFORGE_FOOTAGE_DIRS`, newest first; picked recordings are imported by path (`/api/recordings`, `local_path`). `ingest.place_local_source` hard-links the file into the project: no upload, no copy, no extra disk space, and deleting the project removes only the link (checked: same file, 2 links; after deleting the test project the recording was intact with 1 link). Another drive falls back to a copy; `CLIPFORGE_IMPORT_MODE=copy` restores the old behaviour. Not yet: the CLI batch import still copies, and a file dragged into the browser is still uploaded (a browser cannot reveal its path).
- **R6 RapidOCR: done (2026-11-14).** `rapidocr` 3.10 is the first OCR engine: its PP-OCRv6 small models ship inside the wheel (no download; modelscope.cn refuses downloads from here) and run on the CPU through the `onnxruntime` already installed, so no `onnxruntime-gpu` was needed. Keyword OCR: on 2 minutes of the Valorant match it found 21 events in 26 s against EasyOCR's 22 in 118 s, with clean German text at 0.93-1.00 confidence where EasyOCR garbled it ("VERURSACKT KAI GEDRUCI"); the detector only ever shrinks images, which made HUD crops 3-4x faster. HUD readers run it without the detector (9.5 ms a read): the EA FC scoreboard gave the same 9 goals and minutes as EasyOCR + Tesseract (13 s against 16 s), and the Valorant top bar read 829 of 1174 score plates against EasyOCR's 596. EasyOCR, PaddleOCR and Tesseract stay as fallbacks; a RapidOCR start failure shows in the diagnostics panel. The clean reads exposed a lexicon bug, fixed the same day: the death recap ("GETÖTET VON", "KAMPFBERICHT") counted as a kill and "RUNDE LÄUFT" as a clutch, at 89-99 against 76+ for a real kill. The recap is now a "death" event that never makes a clip and overrules kill words on the same screen, and while the kill feed is read, keyword, cue and audio clips stay below its kills (at most 75). Map callouts such as "C Lobby" next to the minimap no longer count as the lobby menu, which had muted the loud moments within 10 s of them. The buy phase ("KAUFPHASE", "BUY PHASE") now counts like a menu instead: no clip of its own, and sounds and loud moments around it are muted. While the streamer is dead and watches a teammate (from 3 s after their death to the round's end on the top bar plus 5 s), game sounds, audio events, keyword hits and loud moments make no clips; only if that would leave no clip at all do the loud moments stay. The buy-phase plate is no end marker for this: it also lights up before a round is over (13 s early on the scoreboard a dead player opened), so it only caps the stretch at the end of that buy phase.
- **R7 Engine reliability: done (2026-11-14).** emotion2vec loads from the modelscope cache (`~/.cache/modelscope/iic/emotion2vec_plus_base`, 2.6 s) instead of by name, which modelscope.cn refuses (403) and FunASR then reports as "not registered"; `CLIPFORGE_EMOTION_MODEL` overrides. The VLM and LLM batches load their model before their time budget starts: measured from a cold start, the VLM batch scored 0 of 6 clips (a 70 s load against a 45 s budget) and now scores 6 of 6; the vision model also loads in the background a stage early (gameplay: when detection starts; talking: after the text-LLM reads), which took the batch from 34 s to 25 s here. A batch whose model cannot load returns at once instead of spending its budget. Engines that are installed but fail when they run (emotion2vec, PANNs, CLAP, Ollama) report why, and the diagnostics panel shows it ("läuft nicht" plus the reason) instead of "verfügbar".
- **R8 Kill-feed names by OCR.** Read killer and victim names once per row; needed for "who killed whom" titles and as a second check on own kills ([montage_maker](https://github.com/rlmsinclair/montage_maker) works this way).
- **R9 More games on the same framework:** CS2 (feed and round timer), Fortnite (elimination banner, placement), Rocket League (goal banner and scoreboard), Apex (kill banner, squad wipe).
- **R10 OBS integration.** Read chapter markers from Hybrid MP4 recordings and use them as high-priority moments; optional obs-websocket listener that writes a marker on a hotkey. Note: OBS replay-buffer files do not carry chapters yet (OBS issue 13567).

### Later

- **R11 One model for titles and vision** (`qwen3.5:9b`, 6.6 GB) instead of two, and Parakeet TDT v3 for speech (German accuracy on par with Whisper large-v3, several times faster).
- **R12 Chat as a signal** for stream VODs: message rate and emote bursts from the Twitch chat replay ([stream-clipper](https://github.com/nirvagold/stream-clipper) uses this).
- **R13 Learned ranking.** The app already stores feedback on clips; train a small model on it once there are a few hundred rated clips.
- **R14 Montage export:** one video from the best kills or goals of a session, cut to the beat of a chosen track.
- **R15 Direct upload** to TikTok, YouTube Shorts and Instagram with per-platform titles (needs the user's own accounts; stays optional).

## 4. What others do, and what that means here

| Tool | Approach | Takeaway |
|---|---|---|
| Eklipse, Insights Capture, Powder, Medal | Per-game event models or game APIs, mostly cloud or closed | Per-game HUD readers are the right direction; ClipForge does it locally |
| [montage_maker](https://github.com/rlmsinclair/montage_maker) | OCR of the kill feed, keeps rows with the player's name | Name OCR as a second check (R8) |
| [EZClips](https://github.com/corp14x3/EZClips) | Template matching on the CS2 kill feed | Same idea as the banner test; confirms CS2 is feasible (R9) |
| [stream-clipper](https://github.com/nirvagold/stream-clipper) | Audio peaks plus chat activity | Chat signal (R12) |
| openshorts, supoclip, FunClip | Transcript first, for talking-head video | Not suited to gameplay without speech; nothing to adopt for detection |
| Roboflow Valorant kill-feed dataset | Labelled kill-feed images (CC BY 4.0) | Training data if the colour tests ever stop being enough |
