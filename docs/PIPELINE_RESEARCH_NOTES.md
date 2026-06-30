# Pipeline research notes

Last checked: 2026-06-30.

These notes capture current implementation candidates from upstream docs,
GitHub, and Hugging Face. The bias is practical: ship low-risk adapters and
diagnostics now, benchmark heavier model swaps before changing defaults.

## OCR and visual cue detection

### Implemented now: RapidOCR optional backend

- Source: https://github.com/RapidAI/RapidOCR
- Tradeoff: less proven on stylized gameplay HUDs than the current
  PaddleOCR/EasyOCR pair, but much easier to run on CPU via ONNX.
- Complexity: low; it fits the existing OCR provider interface.
- Risk: result shapes changed across RapidOCR packages, so the adapter accepts
  both legacy row tuples and newer object-style outputs.
- Expected benefit: faster OCR experiments on machines where PaddleOCR or
  EasyOCR are too heavy, plus a clean benchmark knob through
  `CLIPFORGE_OCR_ENGINE=rapidocr`.

### Keep as default: PaddleOCR with EasyOCR fallback

- Source: https://github.com/PaddlePaddle/PaddleOCR
- Tradeoff: best accuracy/capability surface, but the dependency stack is
  heavier and version churn is real.
- Complexity: already integrated.
- Risk: repeated backend/language cache reuse can silently route fallbacks to
  the wrong engine; this pass fixes the cache key.
- Expected benefit: strongest current OCR path for scoreboards, banners, and
  kill-feed text.

### Candidate: Surya OCR

- Source: https://github.com/datalab-to/surya
- Tradeoff: stronger document/layout OCR capability, but heavier than needed
  for many short HUD crops.
- Complexity: medium; keep as an optional backend, then benchmark crop latency
  and recall before making it prominent.
- Risk: model size and install friction may hurt the local-first setup.
- Expected benefit: possible improvement for hard layouts, non-Latin text, and
  dense menus once benchmarked.

## ASR, alignment, and diarization

### Keep: faster-whisper plus whisperX/pyannote upgrades

- Sources: https://github.com/m-bain/whisperX and
  https://huggingface.co/pyannote/speaker-diarization-community-1
- Tradeoff: whisperX and pyannote are heavier and may need HF token acceptance,
  but they give word alignment and speaker assignment the current pipeline can
  consume directly.
- Complexity: already integrated.
- Risk: dependency friction and GPU memory.
- Expected benefit: best current path for tight captions and speaker-aware
  clips without changing the project model.

### Candidate: NVIDIA Parakeet TDT 0.6B v3

- Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Tradeoff: strong modern ASR candidate, but it should be proven on German,
  streamer speech, noisy gameplay, and word timing before replacing Whisper.
- Complexity: medium to high via NeMo dependencies and timestamp adaptation.
- Risk: larger install and timestamp semantics may not map cleanly to existing
  caption timing.
- Expected benefit: possible faster or more accurate transcripts on capable
  GPUs if benchmark results justify it.

## VAD and caption timing

### Keep: Silero VAD as speech gate

- Source: https://github.com/snakers4/silero-vad
- Tradeoff: small model dependency, but it materially reduces caption hangover
  after speech stops.
- Complexity: already integrated.
- Risk: aggressive gating can clip very quiet trailing syllables if thresholds
  are tuned too hard.
- Expected benefit: words disappear when speech ends, and Whisper silence
  hallucinations are easier to suppress.

### Candidate: WebRTC VAD fallback

- Source idea: tiny non-neural fallback when torch/onnx paths are unavailable.
- Tradeoff: fast and dependency-light, but less robust on music/game audio.
- Complexity: low.
- Risk: worse recall on noisy streamer/gameplay mixtures.
- Expected benefit: useful warning-free fallback for CPU-only installs.

## Gameplay event detection

### Keep: PANNs plus CLAP cue scoring

- Sources: https://github.com/qiuqiangkong/audioset_tagging_cnn and
  https://github.com/LAION-AI/CLAP
- Tradeoff: audio event models catch hype, explosions, laughter, and applause
  without OCR, but prompt wording and game audio mixing affect reliability.
- Complexity: already integrated.
- Risk: false positives from background music or announcer calls.
- Expected benefit: gives long VODs likely moments so OCR can sample around
  evidence instead of scanning blindly.

### Candidate: GroundingDINO or OWL-style open-vocabulary visual events

- Source: https://github.com/IDEA-Research/GroundingDINO
- Tradeoff: visual object/event prompts could detect "scoreboard", "kill feed",
  or "victory banner", but frame-level inference is expensive.
- Complexity: high; requires batching, prompt calibration, and UI/debug output.
- Risk: slow long-video scans and hard-to-explain false positives.
- Expected benefit: better cue-region discovery and automatic ROI suggestions.

## Tracking, reframe, and subject detection

### Keep: Ultralytics tracker mode behind existing setting

- Source: https://docs.ultralytics.com/modes/track/
- Tradeoff: ByteTrack/BoT-SORT improves temporal continuity, but adds model
  runtime and tracker-state tuning.
- Complexity: already partially exposed through `CLIPFORGE_YOLO_TRACKER`.
- Risk: tracking the wrong gameplay subject can be worse than a stable crop.
- Expected benefit: smoother non-face reframes when no talking-head face is
  available.

## Upload and large-file processing

### Implemented now: raw octet-stream upload route

- Sources: https://www.starlette.io/requests/ and
  https://fastapi.tiangolo.com/tutorial/request-files/
- Tradeoff: bypasses multipart conveniences, so metadata must ride in a prefix
  or header.
- Complexity: already integrated and documented.
- Risk: client/server prefix parsing must stay tested.
- Expected benefit: avoids Starlette multipart body parsing failures before
  ClipForge handler logic sees large videos.

### Candidate: tus/Uppy resumable uploads

- Sources: https://tus.io/protocols/resumable-upload and
  https://uppy.io/docs/tus/
- Tradeoff: much better for interrupted browser uploads, but it adds protocol
  state, partial-file cleanup, and UI retry semantics.
- Complexity: medium to high.
- Risk: abandoned partial uploads can waste disk unless cleanup is strict.
- Expected benefit: long files survive tab/network hiccups and can resume
  instead of restarting from byte zero.

