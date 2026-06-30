# OCR visual cues

ClipForge can turn on-screen game text into exact highlight anchors: kill-feed
lines, round-win banners, score changes, `VICTORY`, `ELIMINATED`, `GOAL`, and
your own custom phrases. OCR is local and optional; when no OCR engine is
installed the audio and cue detectors still run.

## Install

```bash
pip install -r backend/requirements-ocr.txt
```

Recommended order:

- `paddleocr` for clean HUD and banner text.
- `easyocr` as a robust fallback for noisy stream/video-compressed overlays.
- `rapidocr` as a fast ONNX/CPU fallback when the heavier OCR stacks are too
  slow or hard to install.
- `surya-ocr` for heavier visual OCR experiments; benchmark before making it
  the default for gameplay crops.
- `pytesseract` plus the system Tesseract binary as a lightweight fallback.
- `rapidfuzz` for typo-tolerant matching when OCR reads `V1CT0RY` or `HEADSH0T`.

The active OCR engine appears in `/api/capabilities` and the app capability
panel. If OCR is enabled but no engine is installed, the project now records a
warning instead of silently finding zero visual cues.

To benchmark or debug one backend, force it before starting the backend:

```powershell
$env:CLIPFORGE_OCR_ENGINE = "rapidocr"
python -m app.main
```

Accepted values are `paddleocr`, `easyocr`, `rapidocr`, `surya`, `tesseract`,
and `off`.

## How Detection Works

1. Gameplay detection finds likely event times from audio peaks, reference cues,
   CLAP audio events, and scene cuts.
2. OCR samples frames around those likely moments plus a light safety sweep.
3. Each sampled frame is split into useful regions:
   - full frame every few samples
   - Valorant/CS2 kill-feed and banner regions
   - any Cue Lab regions saved for the selected game profile
   - any per-project manual regions
4. Unchanged crops are skipped by perceptual hash, and uncached crops from the
   same frame are batched through the OCR backend.
5. Text is normalized, OCR confusions are repaired, fuzzy matching is applied,
   false positives are filtered, and repeated banners are deduped.
6. Matched visual events become high-priority clips and can bootstrap reusable
   audio cues for future runs.

## Cue Lab Workflow

Use Cue Lab when built-in OCR misses a game-specific marker.

1. Import with **Content type: Gameplay** and the right **Game profile**.
2. Open Cue Lab from the cues area.
3. Pick a frame where the marker is visible.
4. Draw a tight box around the stable text area, not the whole screen.
5. Run OCR on the crop.
6. Save the phrase as a visual cue label such as `round_win`, `kill`, or `goal`.
7. Reprocess the project.

Tight boxes are faster and more accurate. Good regions are scoreboards,
kill-feed columns, round banners, and goal/finish overlays. Avoid chat boxes,
menus, subtitles, and noisy motion areas unless the cue text only appears there.

## OCR Scan Report

After a gameplay run, open **Scan-Kontrolle -> OCR-Bericht** on the project.
The report shows:

- the OCR engine and whether the scan ran, was skipped, failed, or had no
  installed backend
- sampled frames, OCR crops sent to the engine, repeated crop cache hits, raw
  texts found, and matched visual events
- detector warnings for missing OCR installs, crashed scans, or configured Cue
  Lab phrases/regions that did not match
- raw OCR reads with timestamp, region, confidence, text, and matched labels

Use the raw reads to tune missed questions or game-specific overlays. If text is
visible in the report but has `kein Match`, add that phrase in Cue Lab or extend
the game profile keyword list. If no raw reads appear, check OCR install status,
the selected game profile, and whether the saved ROI is too tight or too broad.

## Why A Cue May Not Fire

- OCR is not installed or disabled for the project.
- The project was classified as talking-head footage instead of gameplay.
- The game profile is wrong, so saved regions/phrases are read from another
  profile.
- The phrase is not in the built-in lexicon or Cue Lab visual cue list.
- The region is too large, too small, or includes moving background clutter.
- The marker appears between sampled frames. Add an audio cue for the same event,
  or save a stable visual region so focused sampling has a better chance.
- OCR found text, but it supported a lower-ranked clip outside the final target
  count. Increase target clips for debugging.

When custom visual cues or regions are configured and OCR still finds nothing,
ClipForge now records a project warning that points back to Cue Lab tuning.

## Performance Tips

- Prefer a saved ROI over full-frame OCR. It cuts model work and reduces false
  text from the rest of the screen.
- Keep `rapidfuzz` installed; it is small and helps stylized fonts.
- Use `paddleocr` first, with EasyOCR installed as fallback. Try `rapidocr` when
  CPU speed and simple deployment matter more than maximum accuracy.
- For long VODs, use audio/reference cues too. OCR then focuses around likely
  moments instead of scanning blindly.
- Add one clean audio cue after OCR detects a visual event. Future runs can find
  the same moment without relying on OCR every time.
