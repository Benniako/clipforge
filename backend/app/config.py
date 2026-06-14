"""Runtime configuration and capability resolution for ClipForge.

This module centralises two concerns:

1. Where things live (data dir, database, media storage).
2. Which optional capabilities are available in this environment — a static
   ffmpeg/ffprobe, Whisper for transcription, OpenCV for face tracking. The
   pipeline reads these flags and degrades gracefully when something is
   missing, so the core loop always runs.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


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
    except Exception:
        pass

    # 2. system binaries on PATH.
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")

    # 3. imageio_ffmpeg — ffmpeg only.
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    return ffmpeg, ffprobe


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _detect_ocr() -> str:
    """Best available OCR backend for on-screen game text, or "" if none.

    Preference follows accuracy on noisy game UI text (2026 benchmarks):
    PaddleOCR (PP-OCRv5, most accurate) → EasyOCR (great on screenshots /
    overlays) → Tesseract (lightweight fallback, needs the system binary too).
    Every backend is optional — with none installed, OCR detection is skipped
    and the audio-energy / cue path still finds highlights.
    """
    if _has_module("paddleocr"):
        return "paddleocr"
    if _has_module("easyocr"):
        return "easyocr"
    if _has_module("pytesseract") and shutil.which("tesseract"):
        return "tesseract"
    return ""


def _detect_cuda() -> bool:
    """True if an NVIDIA GPU is usable for the neural models."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_nvenc(ffmpeg: str | None) -> tuple[bool, bool]:
    """(h264_nvenc, av1_nvenc) compiled into this ffmpeg build.

    Note: this only means the encoder is *compiled in*, not that a GPU is present
    — that's why ``use_nvenc`` additionally requires :func:`_detect_nvidia_gpu`.
    """
    if not ffmpeg:
        return False, False
    try:
        import subprocess

        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20)
        return "h264_nvenc" in out.stdout, "av1_nvenc" in out.stdout
    except Exception:
        return False, False


def _detect_nvidia_gpu() -> bool:
    """True if an NVIDIA GPU + driver is actually present (via nvidia-smi)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _detect_vram_mb() -> int:
    """Total VRAM of the first NVIDIA GPU in MB (0 if none)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def _auto_whisper_model(has_cuda: bool, vram_mb: int, cpu: int) -> str:
    """Pick the best Whisper size for the actual compute device.

    GPU (CUDA usable by ctranslate2) → large-v3 (or medium on a small card).
    CPU only → scale by core count (large-v3 on CPU is impractically slow).
    """
    if has_cuda:
        return "large-v3" if vram_mb >= 4500 else "medium"
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
    vram_mb: int = 0        # total VRAM of the first GPU (MB)
    auto_model: bool = True  # whisper model was auto-selected for this hardware

    # --- pipeline tunables ----------------------------------------------
    whisper_model: str = os.environ.get("CLIPFORGE_WHISPER_MODEL", "tiny")
    render_workers: int = int(os.environ.get("CLIPFORGE_RENDER_WORKERS", "2"))
    # Concurrent *projects* in the pipeline. Default 1: transcription and the GPU
    # encoder are the bottlenecks, so a second project mostly contends; raise it
    # if you batch many small videos (transcription is internally serialized).
    pipeline_workers: int = int(os.environ.get("CLIPFORGE_PIPELINE_WORKERS", "1"))
    # Upload / URL-import size cap in MB; 0 (default) = unlimited. A local
    # single-user tool processing your own VODs shouldn't reject them — set
    # this only to guard a small disk.
    max_upload_mb: int = int(os.environ.get("CLIPFORGE_MAX_UPLOAD_MB", "0"))
    # Compute device for the neural models ("cpu" or "cuda").
    device: str = os.environ.get("CLIPFORGE_DEVICE", "cpu")
    # Which transcriber to prefer: "auto" (whisperX if present, else faster-whisper),
    # or force one of "whisperx" / "faster" / "synthetic".
    transcriber: str = os.environ.get("CLIPFORGE_TRANSCRIBER", "auto")
    # Hugging Face token — required only for whisperX speaker diarization
    # (the pyannote model is gated). Without it, whisperX still aligns words.
    hf_token: str | None = (os.environ.get("HF_TOKEN")
                            or os.environ.get("CLIPFORGE_HF_TOKEN") or None)
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

    def capability_report(self) -> dict:
        return {
            "ffmpeg": bool(self.ffmpeg),
            "ffprobe": bool(self.ffprobe),
            "transcription": self.transcription_engine,
            "diarization": self.has_whisperx and bool(self.hf_token),
            "ocr": self.ocr_engine or False,
            "face_tracking": self.has_opencv,
            "url_import": self.has_ytdlp,
            "gpu": self.has_cuda,
            "gpu_encode": self.use_nvenc,
            "codec": ("av1" if self.use_nvenc and self.codec == "av1"
                      and self.has_av1_nvenc else "h264"),
            "device": self.device,
            "whisper_model": self.whisper_model,
            "auto_model": self.auto_model,
            "vram_gb": round(self.vram_mb / 1024, 1) if self.vram_mb else 0,
            "cpu": os.cpu_count() or 0,
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
        has_cuda=has_cuda,
        has_nvenc=has_nvenc,
        has_nvidia=has_nvidia,
        has_av1_nvenc=has_av1_nvenc,
        vram_mb=vram_mb,
        auto_model=model_env is None,
        device=device,
        whisper_model=whisper_model,
        render_workers=render_workers,
    )
