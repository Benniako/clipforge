"""Centralized Two-Tier GPU VRAM Memory Supervisor for ClipForge.

Coordinates sequential loading, model offloading, garbage collection, and CUDA
cache flushes across pipeline stages. Enforces peak VRAM <= 7.5 GB on 16 GB GPUs
(RTX 5060 Ti) with >= 8.5 GB safety headroom, preventing Windows DWM paging
and CUDA out-of-memory errors.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

from .torch_guard import TORCH_LOAD_LOCK

log = logging.getLogger("clipforge.memory_supervisor")


@dataclass
class VRAMInfo:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    total_mb: float = 0.0
    free_mb: float = 0.0


def get_vram_usage() -> VRAMInfo:
    """Query current GPU VRAM utilization in MB via PyTorch CUDA runtime."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = 0
            allocated = torch.cuda.memory_allocated(dev) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(dev) / (1024 * 1024)
            props = torch.cuda.get_device_properties(dev)
            total = props.total_memory / (1024 * 1024)
            free = max(0.0, total - reserved)
            return VRAMInfo(
                allocated_mb=round(allocated, 2),
                reserved_mb=round(reserved, 2),
                total_mb=round(total, 2),
                free_mb=round(free, 2),
            )
    except Exception as e:
        log.debug("could not query torch.cuda memory: %s", e)
    return VRAMInfo(allocated_mb=0.0, reserved_mb=0.0, total_mb=0.0, free_mb=0.0)


def vram_cleanup(stage_name: str | None = None) -> VRAMInfo:
    """Enforce sequential GPU model unloading and memory reclamation.

    Acquires TORCH_LOAD_LOCK to synchronize with any background ASR tasks,
    selectively frees models depending on the pipeline stage transition,
    and runs deterministic garbage collection and CUDA cache empty sweeps.
    """
    with TORCH_LOAD_LOCK:
        if stage_name in ("stage_transcribe_done", "global_reset"):
            try:
                from . import transcribe as transcribe_mod
                if hasattr(transcribe_mod, "unload_model"):
                    transcribe_mod.unload_model()
            except Exception as e:
                log.debug("transcribe unload error: %s", e)

        if stage_name in ("pre_vlm", "stage_detect_done", "global_reset"):
            try:
                from . import emotion as emo_mod
                if hasattr(emo_mod, "unload_model"):
                    emo_mod.unload_model()
            except Exception as e:
                log.debug("emotion unload error: %s", e)

            try:
                from . import audio_events as ae_mod
                if hasattr(ae_mod, "unload_model"):
                    ae_mod.unload_model()
            except Exception as e:
                log.debug("audio_events unload error: %s", e)

        if stage_name in ("stage_detect_done", "global_reset"):
            try:
                from . import vlm as vlm_mod
                if hasattr(vlm_mod, "unload_model"):
                    vlm_mod.unload_model()
            except Exception as e:
                log.debug("vlm unload error: %s", e)

            try:
                from . import llm as llm_mod
                if hasattr(llm_mod, "unload_model"):
                    llm_mod.unload_model()
            except Exception as e:
                log.debug("llm unload error: %s", e)

        if stage_name in ("stage_reframe_done", "global_reset"):
            try:
                from ..media import faces as faces_mod
                if hasattr(faces_mod, "unload_model"):
                    faces_mod.unload_model()
            except Exception as e:
                log.debug("faces unload error: %s", e)

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    info = get_vram_usage()
    if stage_name:
        log.info("VRAM supervisor cleanup [%s]: allocated=%.1fMB reserved=%.1fMB free=%.1fMB",
                 stage_name, info.allocated_mb, info.reserved_mb, info.free_mb)
    return info


def ensure_vram_budget(stage_name: str, required_mb: float = 0.0,
                       max_budget_mb: float = 14000.0) -> bool:
    """Verify that the required VRAM budget is available; trigger cleanup if needed."""
    info = get_vram_usage()
    if info.total_mb > 0:
        used_mb = info.reserved_mb or info.allocated_mb
        if used_mb + required_mb > max_budget_mb:
            log.warning("VRAM budget warning before %s: %.1fMB + %.1fMB > %.1fMB ceiling. Triggering cascade unload.",
                        stage_name, used_mb, required_mb, max_budget_mb)
            info = vram_cleanup(stage_name)
    return True
