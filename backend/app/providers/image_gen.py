"""Optional text-to-image thumbnail generation via local diffusion models.

Currently ClipForge thumbnails are purely frame-extraction + PIL text overlay
(see ``pipeline/render.py:_make_thumbnail``). When a local diffusion model is
available (Krea-2-Raw on Hugging Face) we can generate a stylised cover image
keyed off the clip's title instead.

Fully optional: no model installed ⇒ ``detected()`` is False and the pipeline
keeps using the existing ffmpeg-frame + text-stamping path without any change.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("clipforge.image_gen")

# Known compatible model repos on Hugging Face that support txt2img.
_COMPATIBLE_MODELS = ("krea/Krea-2-Raw",)

# Cached pipeline instance — loaded once, reused across calls.
_pipeline = None


def _hf_cache() -> Path:
    """Resolve the Hugging Face hub cache directory."""
    root = (
        os.environ.get("HF_HOME")
        or os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    )
    return Path(root) / "huggingface" / "hub"


def detected() -> bool:
    """True when a compatible diffusion model is reachable in Hugging Face cache.

    Checks:
    1. That ``diffusers`` and ``torch`` are importable.
    2. That at least one compatible model has a snapshot in HF hub cache.
    """
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    hub = _hf_cache()
    if not hub.is_dir():
        return False
    for model in _COMPATIBLE_MODELS:
        safe = "models--" + model.replace("/", "--")
        if (hub / safe).is_dir():
            return True
    return False


def generate(prompt: str, *, width: int = 1024, height: int = 1024,
             dst: str | Path | None = None) -> Path | None:
    """Generate an image from ``prompt`` using a cached local diffusion model.

    Loads the first compatible model found in Hugging Face cache via
    ``diffusers.DiffusionPipeline`` and runs inference on GPU (or CPU with
    offload). Saves to ``dst`` and returns the path, or None on failure.

    The pipeline is cached globally after the first load so subsequent calls
    skip model loading entirely.

    Gracefully degrades: returns None when no model is cached, when CUDA OOMs,
    or on any other error — the caller falls back to the frame-extraction path.
    """
    global _pipeline
    if not detected():
        return None
    if not dst:
        return None
    dst = Path(dst)
    try:
        import torch
        from diffusers import DiffusionPipeline

        if _pipeline is None:
            model = _COMPATIBLE_MODELS[0]  # first cached model wins
            log.info("loading %s for thumbnail generation…", model)
            _pipeline = DiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16, variant="fp16"
            )
            if torch.cuda.is_available():
                _pipeline = _pipeline.to("cuda")
            else:
                _pipeline.enable_model_cpu_offload()

        image = _pipeline(
            prompt,
            width=width, height=height,
            num_inference_steps=30,
            guidance_scale=6.0,
        ).images[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, quality=92)
        log.info("generated thumbnail from '%s' at %s", prompt[:50], dst)
        return dst
    except Exception as e:
        log.warning("image generation failed (%s); falling back to frame grab", e)
        return None
