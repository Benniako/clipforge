"""Runtime configuration and capability resolution for ClipForge.

This module centralises two concerns:

1. Where things live (data dir, database, media storage).
2. Which optional capabilities are available in this environment — a static
   ffmpeg/ffprobe, Whisper for transcription, OpenCV for face tracking. The
   pipeline reads these flags and degrades gracefully when something is
   missing, so the core loop always runs.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


log = logging.getLogger("clipforge.config")


def _resolve_ffmpeg() -> tuple[str | None, str | None]:
    """Find an ffmpeg + ffprobe pair, preferring a fully static build.

    Order of preference:
      1. ``static_ffmpeg`` (ships both ffmpeg and ffprobe)
      2. system ffmpeg/ffprobe on PATH
      3. ``imageio_ffmpeg`` (ffmpeg only; ffprobe may still be missing)
    Explicit ``FFMPEG_BIN`` / ``FFPROBE_BIN`` env vars win over everything.
    """
    env_ff = os.environ.get("FFMPEG_BIN")
    env_fp = os.environ.get("FFPROBE_BIN")
    if env_ff and env_fp:
        return env_ff, env_fp

    ffmpeg = env_ff
    ffprobe = env_fp

    # 1. static_ffmpeg — bundles a matched ffmpeg + ffprobe.
    try:
        import static_ffmpeg.run as _sfr

        sff, sfp = _sfr.get_or_fetch_platform_executables_else_raise()
        ffmpeg = ffmpeg or sff
        ffprobe = ffprobe or sfp
    except Exception as exc:
        log.debug("static_ffmpeg unavailable: %s", exc)

    # 2. system binaries on PATH.
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")

    # 3. imageio_ffmpeg — ffmpeg only.
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            log.debug("imageio_ffmpeg unavailable: %s", exc)

    return ffmpeg, ffprobe


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _detect_ocr() -> str:
    """Best available OCR backend for on-screen game text, or "" if none.

    Preference follows accuracy on noisy game UI text (2026 benchmarks):
    PaddleOCR (PP-OCRv6 -> PP-OCRv5, most accurate) → EasyOCR (great on screenshots /
    overlays) → Tesseract (lightweight fallback, needs the system binary too).
    Every backend is optional — with none installed, OCR detection is skipped
    and the audio-energy / cue path still finds highlights.
    """
    if _has_module("paddleocr"):
        return "paddleocr"
    if _has_module("easyocr"):
        return "easyocr"
    if _has_module("surya"):
        return "surya"
    if _has_module("pytesseract") and shutil.which("tesseract"):
        return "tesseract"
    return ""


def _ollama_tags_url(url: str, host: str, port: int) -> str:
    """Build the /api/tags URL from the OLLAMA_URL env var or host/port fallback."""
    return url or f"http://{host}:{port}/api/tags"


def _detect_ollama() -> tuple[bool, str]:
    """Return (available, model_names_string).

    True when the local Ollama server is reachable via CLI or port probe.
    Also returns a comma-separated list of installed models to show in the
    diagnostics panel so users can see exactly what AI models are ready.
    """
    models = ""
    if shutil.which("ollama"):
        try:
            from ._util import run_subprocess
            out = run_subprocess(["ollama", "list"], timeout=5, check=False, log_label="ollama")
            lines = out.stdout.splitlines()
            if len(lines) > 1:
                names = [l.split()[0] for l in lines[1:] if l.strip()]
                if names:
                    models = ", ".join(names)
        except Exception as exc:
            log.debug("ollama list unavailable: %s", exc)
        if models:
            return True, models
    import socket
    try:
        # CLIPFORGE_OLLAMA_URL is the canonical env var (used by llm.py and vlm.py).
        # Parse host/port from it if set; fall back to individual vars for compat.
        url = os.environ.get("CLIPFORGE_OLLAMA_URL", "")
        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
        else:
            host = os.environ.get("CLIPFORGE_OLLAMA_HOST", "127.0.0.1")
            port = int(os.environ.get("CLIPFORGE_OLLAMA_PORT", "11434"))
        with socket.create_connection((host, port), timeout=0.5):
            if not models:
                # Socket responded and CLI didn't; try HTTP API for model names.
                try:
                    import json, urllib.request
                    with urllib.request.urlopen(
                            _ollama_tags_url(url, host, port), timeout=1.5) as r:
                        tags = json.loads(r.read()).get("models", [])
                        names = [m.get("name", "") for m in tags if m.get("name")]
                        if names:
                            models = ", ".join(names)
                except Exception:
                    pass
            return True, models or "server running"
    except OSError:
        return False, ""


def _detect_reframe_engine() -> str:
    """Best installed subject-tracking backend for content-aware 9:16 reframe.

    ultralytics (YOLO) tracks people/objects through cuts > mediapipe pose/face
    > the built-in OpenCV Haar/YuNet face crop (always available). Optional.
    """
    if _has_module("ultralytics"):
        return "yolo"
    if _has_module("mediapipe"):
        return "mediapipe"
    return "haar"


def _detect_asd_adapter() -> bool:
    """True only when active-speaker detection can actually relabel words.

    ``CLIPFORGE_ASD_DIR`` alone is not enough: the LR-ASD checkout must include
    its demo entrypoint, model code, weights, and the small MFCC dependency.
    """
    candidates: list[Path] = []
    env = os.environ.get("CLIPFORGE_ASD_DIR")
    if env:
        candidates.append(Path(env))
    data_dir = Path(os.environ.get("CLIPFORGE_DATA_DIR",
                                   Path(__file__).resolve().parents[1] / "data"))
    candidates.append(data_dir / "models" / "LR-ASD")

    required = ("ASD.py", "Columbia_test.py", "model/Model.py")
    weights = ("weight/pretrain_AVA.model", "model/faceDetector/s3fd/sfd_face.pth")
    deps = ("cv2", "numpy", "python_speech_features", "scipy", "sklearn", "torch")
    for asd_dir in candidates:
        if not all((asd_dir / rel).exists() for rel in required):
            continue
        missing_weight = False
        for rel in weights:
            weight_path = asd_dir / rel
            if not weight_path.exists() or weight_path.stat().st_size < 100_000:
                missing_weight = True
                break
        if missing_weight:
            continue
        if not _lr_asd_script_compatible(asd_dir):
            continue
        if not all(_has_module(dep) for dep in deps):
            continue
        if not _torch_cuda_available():
            continue
        return True
    return False


def _lr_asd_script_compatible(asd_dir: Path) -> bool:
    """Guard against an old LR-ASD demo script + new PySceneDetect install."""
    script = asd_dir / "Columbia_test.py"
    try:
        text = script.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if "VideoManager = None" in text and "open_video" in text:
        return True
    if "from scenedetect.video_manager import VideoManager" not in text:
        return True
    return _has_module("scenedetect.video_manager")


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_cuda() -> bool:
    """True if an NVIDIA GPU is usable for the neural models."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception as exc:
        log.debug("ctranslate2 GPU check: %s", exc)
    return _torch_cuda_available()


def _detect_nvenc(ffmpeg: str | None) -> tuple[bool, bool]:
    """(h264_nvenc, av1_nvenc) compiled into this ffmpeg build.

    Note: this only means the encoder is *compiled in*, not that a GPU is present
    — that's why ``use_nvenc`` additionally requires :func:`_detect_nvidia_gpu`.
    """
    if not ffmpeg:
        return False, False
    try:
        from ._util import run_subprocess
        out = run_subprocess([ffmpeg, "-hide_banner", "-encoders"],
                             timeout=20, check=False, log_label="ffmpeg-encoders")
        return "h264_nvenc" in out.stdout, "av1_nvenc" in out.stdout
    except Exception as exc:
        log.debug("nvenc encoder probe: %s", exc)
        return False, False


def _detect_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU + driver is actually present (via nvidia-smi)."""
    import shutil

    if not shutil.which("nvidia-smi"):
        return False
    try:
        from ._util import run_subprocess
        return run_subprocess(["nvidia-smi"], timeout=10, check=False,
                              log_label="nvidia-smi").returncode == 0
    except Exception as exc:
        log.debug("nvidia-smi unavailable: %s", exc)
        return False


def _detect_vram_mb() -> int:
    """Total VRAM of the first NVIDIA GPU in MB (0 if none)."""
    import shutil

    if not shutil.which("nvidia-smi"):
        return 0
    try:
        from ._util import run_subprocess
        out = run_subprocess(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            timeout=10, check=False, log_label="nvidia-smi-vram")
        return int(out.stdout.strip().splitlines()[0])
    except Exception as exc:
        log.debug("nvidia-smi VRAM query: %s", exc)
        return 0


def _auto_whisper_model(has_cuda: bool, vram_mb: int, cpu: int) -> str:
    """Pick the best Whisper size for the actual compute device.

    GPU (CUDA usable by ctranslate2) → large-v3 (or medium on a small card).
    CPU only → scale by core count (large-v3 on CPU is impractically slow).

    Accounts for concurrent GPU load: CLAP (~1GB) and Ollama (~1GB) run
    alongside the transcriber, so the VRAM budget is 2GB less than total.
    Without this reserve, a whisper-large model loads on a 6GB card, then
    CLAP OOMs the pipeline 30 seconds later.
    """
    if has_cuda:
        # Reserve 2 GB for other GPU models (CLAP, Ollama) that load during
        # or after transcription. Without this guard, a card with 6 GB picks
        # large-v3-turbo, and CLAP OOMs before scoring a single clip.
        budget = max(vram_mb - 2000, 0)
        return "large-v3-turbo" if budget >= 4500 else "medium"
    if cpu >= 12:
        return "small"
    if cpu >= 6:
        return "base"
    return "tiny"


def _auto_workers(cpu: int) -> int:
    return max(2, min(cpu // 4, 4))


@dataclass(frozen=True)
class Settings:
    # --- locations -------------------------------------------------------
    data_dir: Path
    db_path: Path
    media_dir: Path

    # --- media engine ----------------------------------------------------
    ffmpeg: str | None
    ffprobe: str | None

    # --- optional capabilities ------------------------------------------
    has_whisper: bool       # faster-whisper
    has_whisperx: bool      # whisperX (word alignment + diarization)
    has_opencv: bool
    has_ytdlp: bool
    has_cuda: bool          # CUDA available for ML (ctranslate2/torch)
    has_nvenc: bool         # ffmpeg has the h264_nvenc encoder compiled in
    has_nvidia: bool        # an NVIDIA GPU + driver is actually present
    has_av1_nvenc: bool = False  # ffmpeg has av1_nvenc (RTX 40/50 series)
    ocr_engine: str = ""    # on-screen text OCR: "paddleocr"|"easyocr"|"tesseract"|""
    # --- optional power-ups (graceful: no-op when absent) ---------------
    has_vad: bool = False        # Silero VAD — snap captions to exact speech
    has_scenedetect: bool = False  # PySceneDetect — better scene-cut snapping
    has_emotion: bool = False    # emotion2vec/FunASR — excitement virality signal
    has_demucs: bool = False     # Demucs — isolate voice from music/game audio
    has_audio_events: bool = False  # PANNs — cheering/laughter/explosion detection
    has_clap: bool = False       # CLAP zero-shot audio cue detection
    reframe_engine: str = "haar"  # "yolo" | "mediapipe" | "haar"
    has_asd: bool = False        # LR-ASD active-speaker detection wired in
    vram_mb: int = 0        # total VRAM of the first GPU (MB)
    auto_model: bool = True  # whisper model was auto-selected for this hardware
    # --- individual optional tools surfaced in the capability report ----------
    has_deno: bool = False       # deno JS runtime — yt-dlp needs it for 1080p YouTube
    has_ollama: bool = False     # local LLM server — virality re-ranking (optional)
    ollama_models: str = ""      # installed model names (comma-separated)
    has_openmodel: bool = False  # OpenModel.ai API key — cloud LLM replacement
    has_torchaudio: bool = False # wav2vec2 forced alignment for tighter captions
    has_paddleocr: bool = False  # OCR engine (best accuracy overall)
    has_easyocr: bool = False    # OCR engine (best on noisy frames)
    has_tesseract: bool = False  # OCR engine (fallback)
    has_scrfd: bool = False      # SCRFD face detection (upgrade from YuNet)
    has_surya: bool = False      # Surya OCR (vision-language based OCR)

    # --- pipeline tunables ----------------------------------------------
    whisper_model: str = os.environ.get("CLIPFORGE_WHISPER_MODEL", "tiny")
    render_workers: int = int(os.environ.get("CLIPFORGE_RENDER_WORKERS", "2"))
    # Concurrent *projects* in the pipeline. Default 1: transcription and the GPU
    # encoder are the bottlenecks, so a second project mostly contends; raise it
    # if you batch many small videos (transcription is internally serialized).
    # Default 2: safe when GPU encoding is active (NVENC offloads encode to
    # hardware, freeing the CPU for a second project). Bump higher for batches
    # of short videos where I/O dominates. Set CLIPFORGE_PIPELINE_WORKERS=1 to
    # revert to the sequential behaviour.
    pipeline_workers: int = int(os.environ.get("CLIPFORGE_PIPELINE_WORKERS", "2"))
    # Upload / URL-import size cap in MB; 0 (default) = unlimited. A local
    # single-user tool processing your own VODs shouldn't reject them — set
    # this only to guard a small disk.
    max_upload_mb: int = int(os.environ.get("CLIPFORGE_MAX_UPLOAD_MB", "0"))
    # Compute device for the neural models ("cpu" or "cuda").
    device: str = os.environ.get("CLIPFORGE_DEVICE", "cpu")
    # Batched-inference batch size for faster-whisper on GPU (BatchedInference
    # Pipeline) — bigger keeps the GPU saturated; 0 disables batching.
    whisper_batch_size: int = int(os.environ.get("CLIPFORGE_WHISPER_BATCH", "8"))
    # Which transcriber to prefer: "auto" (whisperX if present, else faster-whisper),
    # or force one of "whisperx" / "faster" / "synthetic".
    transcriber: str = os.environ.get("CLIPFORGE_TRANSCRIBER", "auto")
    # Hugging Face token — required only for whisperX speaker diarization
    # (the pyannote model is gated). Without it, whisperX still aligns words.
    hf_token: str | None = (os.environ.get("HF_TOKEN")
                            or os.environ.get("CLIPFORGE_HF_TOKEN") or None)
    german_gaming_prompt: str = os.environ.get(
        "CLIPFORGE_GERMAN_GAMING_PROMPT",
        "Dies ist ein deutsches Gameplay-Video. Begriffe wie Ace, Clutch, "
        "Enemy down, Bombe geplant, Bombe gelegt, Runde gewonnen, Kopfschuss, "
        "krass und insane werden gesprochen.",
    )
    # pyannote's current free local pipeline. Override only when you have a
    # different gated model/API-key arrangement.
    diarization_model: str = os.environ.get(
        "CLIPFORGE_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    # Output codec: "h264" (default — universal playback) or "av1" (av1_nvenc,
    # better quality per bitrate; needs an RTX 40/50-series GPU encoder).
    codec: str = os.environ.get("CLIPFORGE_CODEC", "h264")
    # Output canvas (9:16). 1080x1920 is the platform-native short-form size.
    out_width: int = 1080
    out_height: int = 1920

    allowed_origins: list[str] = field(
        default_factory=lambda: os.environ.get(
            "CLIPFORGE_CORS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
    )

    @property
    def can_render(self) -> bool:
        return bool(self.ffmpeg)

    @property
    def has_ocr(self) -> bool:
        return bool(self.ocr_engine)

    # ---------------------------------------------------------------- #
    # Capability detail — a structured, human-readable inventory of what
    # ClipForge found installed. Used by /api/capabilities and the UI's
    # diagnostics panel so users can see exactly what's available and what
    # each piece unlocks, rather than guessing from behaviour.
    # ---------------------------------------------------------------- #
    def capability_detail(self) -> dict:
        """Return a grouped inventory of detected capabilities.

        Shape: ``{"categories": [{name, items: [{key, available, label, impact}]}]}``.
        ``impact`` explains what each capability unlocks (or what degrades when
        absent) so the panel is actionable, not just a checklist.
        """
        def item(key: str, available: bool, label: str, impact: str) -> dict:
            return {"key": key, "available": available,
                    "label": label, "impact": impact}

        return {"categories": [
            {"name": "core", "items": [
                item("ffmpeg", bool(self.ffmpeg),
                     "ffmpeg", "Required for video decode/encode. Without it nothing renders."),
                item("ffprobe", bool(self.ffprobe),
                     "ffprobe", "Media probing (duration, dimensions, audio)."),
                item("deno", self.has_deno,
                     "deno (JS runtime)",
                     "yt-dlp needs this to read YouTube's player. Without it, YouTube "
                     "imports may be capped at 360p."),
                item("yt_dlp", self.has_ytdlp,
                     "yt-dlp", "Enables importing from a pasted URL (YouTube + ~1000 sites)."),
            ]},
            {"name": "transcription", "items": [
                item("faster_whisper", self.has_whisper,
                     "faster-whisper", "Baseline word-timed transcription (Whisper). Required for captions."),
                item("whisperx", self.has_whisperx,
                     "whisperX", "Upgrades captions with sub-100ms word alignment + speaker diarization."),
                item("silero_vad", self.has_vad,
                     "Silero VAD",
                     "Pins captions to exact speech and drops silence hallucinations. "
                     "Without it captions may drift."),
                item("torchaudio", self.has_torchaudio,
                     "torchaudio",
                     "Optional wav2vec2 forced alignment to tighten faster-whisper timestamps."),
                item("ollama", self.has_ollama,
                     "Ollama",
                     f"Local LLM server for virality re-ranking. Models: {self.ollama_models or 'none installed'}"),
            ]},
            {"name": "vision", "items": [
                item("opencv", self.has_opencv,
                     "OpenCV", "Face tracking for speaker-aware 9:16 reframing."),
                item("reframe_engine", True,
                     f"Reframe backend: {self.reframe_engine}",
                     "yolo (best) > mediapipe > haar/YuNet (always available)."),
                item("asd", self.has_asd,
                     "LR-ASD active speaker",
                     "Ties transcript words to the on-screen speaker for multi-person content."),
                item("scrfd", self.has_scrfd,
                     "SCRFD face detection",
                     "Improved face detection. Replaces YuNet. ONNX GPU-accelerated."),
            ]},
            {"name": "ocr", "items": [
                item("paddleocr", self.has_paddleocr,
                     "PaddleOCR", "Best overall OCR accuracy for in-game HUD text."),
                item("easyocr", self.has_easyocr,
                     "EasyOCR", "Better than PaddleOCR on noisy/bitrate-starved frames."),
                item("tesseract", self.has_tesseract,
                     "Tesseract", "Fallback OCR engine."),
                item("surya", self.has_surya,
                     "Surya OCR", "Vision-language based OCR (90+ langs, GPU). Alternative to PaddleOCR."),
                item("ocr_selected", bool(self.ocr_engine),
                     f"Active OCR: {self.ocr_engine or 'none'}",
                     "Selected automatically from the engines above. None = OCR detection skipped."),
            ]},
            {"name": "audio", "items": [
                item("clap", self.has_clap,
                     "CLAP", "Zero-shot audio cue detection (cheers, explosions, custom prompts)."),
                item("panns", self.has_audio_events,
                     "PANNs", "Cheering/laughter/explosion detection as a virality signal."),
                item("emotion", self.has_emotion,
                     "emotion2vec/FunASR", "Excitement/intensity virality signal from voice."),
                item("demucs", self.has_demucs,
                     "Demucs", "Isolates voice from music/game audio for cleaner transcription."),
            ]},
            {"name": "gpu", "items": [
                item("nvidia", self.has_nvidia,
                     "NVIDIA GPU", "Present and detected by nvidia-smi."),
                item("cuda", self.has_cuda,
                     "CUDA", "Available for ML acceleration (ctranslate2/torch)."),
                item("nvenc", self.has_nvenc,
                     "NVENC (h264)",
                     f"ffmpeg h264_nvenc encoder. {'Used for GPU rendering.' if self.use_nvenc else 'Compile of ffmpeg includes it but no GPU is active.'}"),
                item("av1_nvenc", self.has_av1_nvenc,
                     "NVENC (av1)", "RTX 40/50-series AV1 hardware encoding."),
                item("vram", self.vram_mb > 0,
                     f"VRAM: {self.vram_mb} MB",
                     "Total VRAM on the first GPU. Drives the auto-selected Whisper model size."),
            ]},
            {"name": "scenework", "items": [
                item("scenedetect", self.has_scenedetect,
                     "PySceneDetect", "Snaps clip boundaries to real scene cuts."),
            ]},
            {"name": "extras", "items": [
                item("image_gen", False,
                     "AI Cover Image Generation",
                     "Generate stylised thumbnails from clip titles using a local "
                     "diffusion model (Krea-2-Raw or ideogram). Install deps to enable."),
            ]},
        ]}

    @property
    def upload_cap_bytes(self) -> int | None:
        """Byte cap for uploads/URL imports; None = unlimited."""
        return self.max_upload_mb * 1024 * 1024 if self.max_upload_mb > 0 else None

    @property
    def transcription_engine(self) -> str:
        """The transcriber that will actually be used, given prefs + availability."""
        pref = self.transcriber
        if pref == "whisperx" and self.has_whisperx:
            return "whisperx"
        if pref == "faster" and self.has_whisper:
            return "whisper"
        if pref == "synthetic":
            return "synthetic"
        # auto
        if self.has_whisperx:
            return "whisperx"
        if self.has_whisper:
            return "whisper"
        return "synthetic"

    @property
    def use_nvenc(self) -> bool:
        flag = os.environ.get("CLIPFORGE_NVENC")  # "0"/"1" to force off/on
        if flag is not None:
            return flag == "1" and self.has_nvenc
        # Only when the encoder exists AND a real GPU is present, or CUDA is up.
        return self.has_nvenc and (self.has_nvidia or self.has_cuda)

    def video_encoder_args(self) -> list[str]:
        """ffmpeg video-encode args — GPU (NVENC) when available, else x264.

        ``codec="av1"`` opts into av1_nvenc when the encoder exists; anything
        else (or a missing AV1 encoder) degrades to the H.264 path so a bad
        setting can never break rendering.
        """
        if self.use_nvenc:
            if self.codec == "av1" and self.has_av1_nvenc:
                return ["-c:v", "av1_nvenc", "-preset", "p5", "-rc", "vbr",
                        "-cq", "30", "-b:v", "0", "-pix_fmt", "yuv420p"]
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                    "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p", "-profile:v", "high"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-profile:v", "high"]

    def render_workers_for(self, power_mode: str | None) -> int:
        """Per-project render fan-out."""
        mode = (power_mode or "balanced").lower()
        cpu = os.cpu_count() or 4
        if mode == "max_gpu":
            return max(1, min(max(self.render_workers, cpu // 2), 8))
        if mode == "quality":
            return max(1, min(self.render_workers, 2))
        return max(1, self.render_workers)

    def whisper_batch_for(self, power_mode: str | None) -> int:
        """Batch size for faster-whisper/WhisperX on this project."""
        base = max(0, self.whisper_batch_size)
        if self.device != "cuda":
            return 0
        mode = (power_mode or "balanced").lower()
        if mode == "max_gpu":
            return max(base, 16 if self.vram_mb >= 12000 else 8)
        if mode == "quality":
            return max(base, 8)
        return base

    def vlm_options_for(self, power_mode: str | None) -> dict[str, float | int]:
        """Budget/parallelism for local vision-model scoring."""
        mode = (power_mode or "balanced").lower()
        if mode == "max_gpu":
            return {"budget": 90.0, "max_workers": 2, "n_frames": 2,
                    "timeout": 35.0}
        if mode == "quality":
            return {"budget": 120.0, "max_workers": 2, "n_frames": 3,
                    "timeout": 45.0}
        return {"budget": 45.0, "max_workers": 1, "n_frames": 2,
                "timeout": 30.0}

    def capability_report(self) -> dict:
        return {
            "ffmpeg": bool(self.ffmpeg),
            "ffprobe": bool(self.ffprobe),
            "transcription": self.transcription_engine,
            "diarization": self.has_whisperx and bool(self.hf_token),
            "ocr": self.ocr_engine or False,
            "vad": self.has_vad,
            "scene_detect": self.has_scenedetect,
            "emotion": self.has_emotion,
            "denoise": self.has_demucs,
            "audio_events": self.has_audio_events or self.has_clap,
            "panns_audio": self.has_audio_events,
            "clap_audio": self.has_clap,
            "reframe_engine": self.reframe_engine,
            "active_speaker": self.has_asd,
            "face_tracking": self.has_opencv,
            "url_import": self.has_ytdlp,
            "gpu": self.has_cuda,
            "gpu_encode": self.use_nvenc,
            "codec": ("av1" if self.use_nvenc and self.codec == "av1"
                      and self.has_av1_nvenc else "h264"),
            "device": self.device,
            "whisper_model": self.whisper_model,
            "diarization_model": self.diarization_model if self.hf_token else None,
            "auto_model": self.auto_model,
            "vram_gb": round(self.vram_mb / 1024, 1) if self.vram_mb else 0,
            "cpu": os.cpu_count() or 0,
            "recommended_power_mode": (
                "max_gpu" if self.has_cuda and self.vram_mb >= 12000 else "balanced"
            ),
            # New, surfaced for the diagnostics panel.
            "deno": self.has_deno,
            "ollama": self.has_ollama,
            "ollama_models": self.ollama_models,
            "torchaudio": self.has_torchaudio,
            "paddleocr": self.has_paddleocr,
            "easyocr": self.has_easyocr,
            "tesseract": self.has_tesseract,
            "scrfd": self.has_scrfd,
            "surya": self.has_surya,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("CLIPFORGE_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    media_dir = data_dir / "media"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg, ffprobe = _resolve_ffmpeg()
    has_cuda = _detect_cuda()
    has_nvenc, has_av1_nvenc = _detect_nvenc(ffmpeg)
    has_nvidia = _detect_nvidia_gpu()
    vram_mb = _detect_vram_mb()
    device = os.environ.get("CLIPFORGE_DEVICE") or ("cuda" if has_cuda else "cpu")

    cpu = os.cpu_count() or 4
    _ollama_result = _detect_ollama()  # call once, use for both availability + models string
    model_env = os.environ.get("CLIPFORGE_WHISPER_MODEL")
    whisper_model = model_env or _auto_whisper_model(has_cuda, vram_mb, cpu)
    workers_env = os.environ.get("CLIPFORGE_RENDER_WORKERS")
    render_workers = int(workers_env) if workers_env else _auto_workers(cpu)

    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "clipforge.db",
        media_dir=media_dir,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        has_whisper=_has_module("faster_whisper"),
        has_whisperx=_has_module("whisperx"),
        has_opencv=_has_module("cv2"),
        has_ytdlp=_has_module("yt_dlp"),
        ocr_engine=_detect_ocr(),
        has_vad=_has_module("silero_vad"),
        has_scenedetect=_has_module("scenedetect"),
        has_emotion=_has_module("funasr"),
        has_demucs=_has_module("demucs"),
        has_audio_events=_has_module("panns_inference"),
        has_clap=_has_module("laion_clap"),
        reframe_engine=_detect_reframe_engine(),
        has_asd=_detect_asd_adapter(),
        has_cuda=has_cuda,
        has_nvenc=has_nvenc,
        has_nvidia=has_nvidia,
        has_av1_nvenc=has_av1_nvenc,
        vram_mb=vram_mb,
	        auto_model=model_env is None,
        has_deno=bool(shutil.which("deno")),
	        has_ollama=_ollama_result[0],
	        ollama_models=_ollama_result[1],
        has_torchaudio=_has_module("torchaudio"),
        has_paddleocr=_has_module("paddleocr"),
        has_easyocr=_has_module("easyocr"),
        has_tesseract=bool(_has_module("pytesseract") and shutil.which("tesseract")),
        has_scrfd=_has_module("scrfd"),
        has_surya=_has_module("surya"),
        device=device,
        whisper_model=whisper_model,
        render_workers=render_workers,
        diarization_model=os.environ.get(
            "CLIPFORGE_DIARIZATION_MODEL",
            "pyannote/speaker-diarization-community-1",
        ),
        german_gaming_prompt=os.environ.get(
            "CLIPFORGE_GERMAN_GAMING_PROMPT",
            Settings.__dataclass_fields__["german_gaming_prompt"].default,
        ),
    )
