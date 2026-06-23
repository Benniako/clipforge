# ClipForge — Capabilities Overview

A local-first tool that turns long-form video (podcasts, interviews, gameplay
VODs, streams) into ranked, captioned, vertically-framed short clips — ready for
Reels/Shorts/TikTok. Everything runs on the user's machine; no video, audio, or
transcript leaves the machine unless the user explicitly exports it.

This document is an engineer-facing map of what the system actually does, which
subsystem owns each behaviour, and where the optional/hard dependencies are.

---

## End-to-end pipeline

A project moves through six stages. Each is resilient: a missing optional
dependency degrades that stage gracefully rather than failing the run.

```
ingest → transcribe → detect → score → caption → render
```

| Stage | What it does | Owner | Required deps | Optional deps (upgrade) |
|-------|--------------|-------|---------------|-------------------------|
| **ingest** | Accept an upload or a pasted URL; probe duration/dimensions/audio | `pipeline/ingest.py` | ffmpeg | `yt-dlp` (URL import) |
| **transcribe** | Word-timed speech-to-text + speaker labels | `providers/transcribe.py` | `faster-whisper`, `silero-vad` | `whisperx` (alignment + diarization), `torchaudio` (extra alignment) |
| **detect** | Find highlight moments: salience, audio cues, OCR, facecam | `providers/detect.py`, `detect_gameplay.py`, `detect_ocr.py`, `audio_events.py` | — | `opencv`, CLAP, PaddleOCR/EasyOCR/Tesseract, LR-ASD |
| **score** | Rank moments by virality; LLM re-rank; per-user feedback learning | `providers/score.py`, `feedback.py` | numpy | Ollama (LLM re-rank) |
| **caption** | Build word-timed ASS subtitles; tight segment cuts | `pipeline/captions.py`, `captionize.py` | ffmpeg | — |
| **render** | Vertical reframe + facecam layout + burn-in captions → mp4 | `pipeline/reframe.py`, `pipeline/render.py` | ffmpeg | `opencv` (tracked reframe) |

---

## Capability detail

### 1. Import (ingest)
- **File upload** — drag-and-drop or file picker; capped by an upload size limit.
- **URL import** — `yt-dlp` handles YouTube and ~1000 other hosts. The download
  path is hardened against YouTube throttling (`player_client: android/web`),
  transient 429s (5 retries + fragment retries), playlists (`noplaylist`), and
  missing separate-audio streams (progressive format fallback). Real errors are
  surfaced to the UI, not swallowed.
- **Direct HTTP download** — fallback when `yt-dlp` isn't installed.

### 2. Transcription & alignment
- **faster-whisper** is the baseline engine (word timestamps + VAD filter).
- **whisperX** upgrades the path with wav2vec2 forced alignment (sub-100 ms word
  accuracy) and optional pyannote speaker diarization (needs an HF token).
- **Silero VAD** is a hard dependency: it clamps every caption word to its
  speech region and drops Whisper's silence hallucinations ("thank you for
  watching"). Without it captions drift and the UI warns.
- **Standalone wav2vec2 aligner** (`providers/align.py`) — a self-contained CTC
  forced-alignment pass that tightens faster-whisper timings when whisperX isn't
  available. Pure-fail-safe no-op when torchaudio is absent.
- **Anti-hallucination params** — `condition_on_previous_text=False` (stops
  hallucinations propagating across segments) and `beam_size≥3` (not the
  hallucination-prone `beam_size=1`).

### 3. Moment detection
- **Salience scoring** — transcript-driven hook/emotion/clarity/quote/pace/
  length/list signals with weighted fusion and non-max suppression (IoU-based).
- **Gameplay audio cues** — CLAP zero-shot audio classification with rich,
  attribute-style prompt sets per game profile. Short user prompts are
  auto-enriched (`audio_events.enrich_prompts`) per the finding that prompt
  phrasing swings CLAP accuracy ~20%.
- **OCR** — PaddleOCR → EasyOCR → Tesseract cascade for in-game HUD text (kill
  banners, scorelines, "VICTORY" splashes). Low-confidence PaddleOCR frames are
  retried with EasyOCR (noisy VODs). Learned cues persist per game profile.
- **Active-speaker attribution** — optional LR-ASD ties transcript words to the
  on-screen speaker for multi-person content.
- **Facecam detection** — YuNet/OpenCV; stable-face clustering finds a static
  streamer-cam overlay once for the whole source.

### 4. Scoring & personalization
- **Virality score (0–100)** — transparent weighted sum of named factors, each
  with a human-readable detail string shown in the UI.
- **LLM re-rank** — an Ollama VLM reads the top candidates and applies a
  rubric-based virality assessment, with project visual cues injected.
- **Feedback learning** — explicit good/bad ratings + download signal train a
  per-user logistic weight vector (cold-start blended, evidence-capped). The
  learner personalizes factor weights without ever sending content off-machine.

### 5. Reframing (the "camera")
- **Speaker-aware vertical reframe** — samples frames at 3 fps, detects faces,
  and tracks the speaking face. The crop centre is smoothed with a **One-Euro
  filter** (adaptive: jitter-free when still, responsive during pans) and
  velocity-clamped + edge-safety-clamped.
- **Speaker-switch stability** — incumbent/challenger logic with dwell
  hysteresis: the camera only switches when a new speaker's mouth motion beats
  the incumbent by a clear margin for ≥3 consecutive frames, or on speech onset.
  This stops the jittery "camera breathes" and mis-switches.
- **Manual override** — users can pin the crop, choose aspect (9:16/4:5/1:1/16:9),
  and lay out a facecam (stacked / picture-in-picture / gameplay-only).
- **Fallbacks** — YOLO/MediaPipe subject tracking follows a non-face subject
  (turned-away player, pet); pure smart-center when no tracking is possible.

### 6. Captioning
- **Word-timed ASS subtitles** with per-word highlight, configurable style
  (font, size, colour, outline, case, emphasis), and integer-centisecond
  timing math (avoids the `:60.00` float rollover bug).
- **Caption integrity** — end-time is unconditionally floored to `start + 0.08s`
  so words sharing a timestamp never produce a zero-duration (libass-dropped)
  span. Game-sound windows mute captions so an announcer call doesn't fight the
  burned-in transcript.
- **Tight cuts** — `captionize` snaps clip boundaries to sentence breaks and
  speech onsets; optional silence trimming tightens pacing.

### 7. Rendering
- **ffmpeg encode** with burn-in captions, reframe crop, facecam composite, and
  per-clip aspect override. Per-clip locking prevents two ffmpeg processes
  hitting the same output file (Windows file-lock safe).
- **Live preview during processing** — rendered clips appear as they finish.
- **Export** — mp4 download + standalone `.srt` for Premiere/Resolve.

---

## Power modes

Three power modes trade speed for quality and are surfaced in the UI:
`balanced`, `quality`, `max_gpu`. They control Whisper batch size, compute type
(float16/int8), and reframe sampling density.

---

## Observability in the UI (render window)

The processing view shows:
- **ETA** with elapsed/source-length/throughput (× realtime) pills
- **Per-stage** status with individual % and elapsed time (active + completed)
- **System** pills: CPU %, GPU %, VRAM usage (when available)
- **Live clip grid** of clips rendered so far
- **Pause/resume** with safe drain of in-flight encodes
- **Warnings** (e.g. VAD absent → captions may drift) at error/warning severity

---

## What ClipForge deliberately does NOT do

- **No cloud calls for content** — transcription, scoring, rendering are local.
  The only network egress is the initial URL download (user-initiated) and
  optional Ollama (runs locally by default).
- **No upload-and-forget** — projects are synchronous and inspectable; the user
  can edit any clip's title, cut, reframe, captions, and speakers before render.
- **No opaque scoring** — every score factor has a name and a detail string.

---

## Extension points

- **Game packs** (`game_packs.py`) — per-game CLAP prompts, OCR keywords, and
  profile aliases. Add a game by registering its prompt set.
- **Visual cue lab** (`VisualCueCalibration`) — users calibrate OCR regions and
  mark false recognitions; learned cues persist and improve future scans.
- **Style templates** — caption styles are JSON-defined and selectable at import.
