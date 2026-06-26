# GPU Pipeline Optimization — Design Spec

**Date:** 2026-06-26
**Branch:** `fix/stuck-transcribe-and-hardening`
**Target:** 1-hour video processed in ≤30 minutes total (import → rendered clips) on RTX 5060 Ti 16 GB

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture & Bottlenecks](#2-current-architecture--bottlenecks)
3. [Section A: Dedicated ASR Thread](#3-section-a-dedicated-asr-thread)
4. [Section B: Async Ollama Calls](#4-section-b-async-ollama-calls)
5. [Section C: Shared Frame Cache](#5-section-c-shared-frame-cache)
6. [Section D: Parallel LLM + VLM](#6-section-d-parallel-llm--vlm)
7. [Section E: Source-Based Thumbnails](#7-section-e-source-based-thumbnails)
8. [Section F: Scaling Knobs & Concurrency](#8-section-f-scaling-knobs--concurrency)
9. [Implementation Order](#9-implementation-order)
10. [Quality Guarantees](#10-quality-guarantees)

---

## 1. Executive Summary

The existing `fix/stuck-transcribe-and-hardening` branch fixes pipeline stability and GPU detection. This spec stacks seven structural optimizations on top that eliminate redundant work, parallelize independent GPU resources, and remove blocking I/O — targeting **≤30 minutes** for a 1-hour VOD on an RTX 5060 Ti 16 GB.

**Current estimated time for 1-hour VOD (10 clips):** ~25-40 min
**Target after optimizations:** ~18-25 min (headroom for the 30-min goal)

### Resource Map

| Resource | Used By | Can Run Concurrently With |
|----------|---------|---------------------------|
| CUDA cores (ASR model) | faster-whisper/whisperX | Ollama (separate process), NVENC (separate hardware) |
| NVENC hardware encoder | ffmpeg render | ASR model, Ollama, CPU scoring |
| Ollama (GPU LLM server) | LLM scoring, VLM scoring, hashtags | ASR model (different GPU processes) |
| CPU cores | ffmpeg decode, Python scoring, numpy | Everything above |
| Disk I/O | Source reads, frame cache, output writes | Everything above |

Key insight: **these resources are largely independent** — the current pipeline serializes work that could overlap.

---

## 2. Current Architecture & Bottlenecks

### Pipeline Stages (sequential per project)

```
Project A:
  [transcribe: 5-10min] → [detect: 1-2min] → [score: 2-4min] → [reframe: 2-3min] → [caption: 30s] → [render: 4-8min]
                                                                                                           ↑ parallel per clip

Project B (if pipeline_workers=2):
  [transcribe: 5-10min] → but blocked by _asr_lock while A transcribes
```

### Bottleneck Table

| # | Bottleneck | Time Lost (1hr VOD) | Resource | Fix |
|---|-----------|---------------------|----------|-----|
| 1 | Global `_asr_lock` serializes all transcription | 5-10 min (batch only) | GPU ASR | Dedicated ASR thread |
| 2 | Sequential LLM then VLM (blocking HTTP) | 1-2 min | Worker thread | Async + parallel Ollama |
| 3 | Redundant ffmpeg decode (VLM + facecam + reframe + OCR) | 2-5 min | CPU/IO | Shared frame cache |
| 4 | Thumbnail extracted from rendered output | 1-2 min | CPU/IO | Source-based thumbnail |
| 5 | Conservative semaphore/worker caps | 1-3 min | GPU/NVENC | Scale to hardware |
| 6 | GIL constrains Python scoring | 30s-1min | CPU | (Minor — addressed by overlap) |

---

## 3. Section A: Dedicated ASR Thread

### Problem

`_asr_lock` in `torch_guard.py` is a global `threading.RLock()` shared by `transcribe.py` (ASR model load + inference) and `audio_events.py` (CLAP model load). Every `transcribe()` call acquires this lock. With `pipeline_workers=2`, the second worker thread blocks at the transcribe stage until the first worker fully finishes transcription — even though the second worker could be doing detection or scoring on another project.

### Design

Add a dedicated ASR worker thread to the `Engine` class that owns the GPU ASR model and processes transcription jobs from a queue. Pipeline workers submit audio paths and get back futures.

```
┌─────────────────────────────────────────────────┐
│                  Engine                          │
│                                                  │
│  Pipeline Worker 1 ─┐                            │
│                     ├── submit_audio() ─────────►│
│  Pipeline Worker 2 ─┘         │                  │
│                               ▼                  │
│                     ┌─────────────────┐          │
│                     │  ASR Queue       │          │
│                     │  (thread.Queue)  │          │
│                     └────────┬────────┘          │
│                              │                   │
│                     ┌────────▼────────┐          │
│                     │  ASR Worker      │          │
│                     │  (daemon thread) │          │
│                     │  Owns _asr_lock  │          │
│                     │  Loads model once│          │
│                     └────────┬────────┘          │
│                              │                   │
│                     ┌────────▼────────┐          │
│                     │  Transcript     │          │
│                     │  (written to    │          │
│                     │   project store)│          │
│                     └─────────────────┘          │
└─────────────────────────────────────────────────┘
```

### Implementation Details

**New class: `AsrJob`**
```python
@dataclass
class AsrJob:
    project_id: str
    audio_path: str
    language: str | None
    power_mode: str | None
    progress: Callable | None
    future: concurrent.futures.Future
```

**Engine changes:**
- `__init__`: Create `self._asr_queue: queue.Queue[AsrJob]` and `self._asr_futures: dict[str, Future]`
- `start()`: Also start the ASR worker thread
- New `_asr_loop()`: Drains ASR queue, acquires `_asr_lock`, calls `transcribe_mod.transcribe()`, resolves future
- New `submit_asr()`: Posts an AsrJob to the queue, returns the future
- `_process()` stage 0: Calls `submit_asr()` instead of `transcribe_mod.transcribe()` directly, then does the WAV cleanup and VAD prep while ASR runs

**transcribe.py changes:**
- No change needed — the module is already thread-safe (global model cache, reentrant lock)

**audio_events.py changes:**
- CLAP model loading already uses `_asr_lock`. Since ASR now runs on its own thread, CLAP loading on the pipeline worker thread will briefly contend for the lock, but the CLAP load is fast (~2s) and rare (once per project).

### Stage Separation

The key optimization for mixed workloads: while ASR runs on the GPU thread, the pipeline worker can:

1. **For gameplay content:** Start `gameplay_mod.detect_gameplay()` immediately — it does audio energy analysis on the WAV file, which doesn't need the transcript. The WAV is already extracted and available.
2. **For talking content:** Start `classify.detect_content_type()` — it needs the source video probe info, which is already available.
3. **For batch processing:** Pick up another project from the queue and start its audio extraction + submit its ASR job.

The pipeline worker calls `future.result()` (blocks) only when it truly needs the transcript — right before the VAD refine and detection stages that depend on word timings.

### Time Impact

| Scenario | Before | After | Saving |
|----------|--------|-------|--------|
| Single 1hr VOD | ASR runs inline (5-10min), pipeline worker idle | ASR runs on own thread; pipeline worker starts detect early | 0-3 min* |
| Batch of 3×20min videos | Sequential ASR: 3×3min = 9min (blocked by lock) | Overlapped ASR + detect: ~6min total | 3 min |
| Batch of 5×10min shorts | Sequential: 5×2min = 10min serial | Overlapped: ~4min | 6 min |

*For single VOD talking content, the pipeline worker can start gameplay classification and facecam detection early, but the main detection stage still needs the transcript. The saving here is smaller — the bigger wins come from the other sections.

---

## 4. Section B: Async Ollama Calls

### Problem

`ollama_client.generate()` uses `urllib.request.urlopen()` — a synchronous blocking call. During the score stage, the pipeline makes:

1. `llm_mod.score_virals()` — 1 call per clip (6-12 clips × ~3s = 18-36s)
2. `llm_mod.suggest_titles()` — 1 call per clip (6-12 clips × ~3s = 18-36s)  
3. `vlm_mod.score_visuals()` — 1 call per clip (6-12 clips × ~5s = 30-60s)

Total blocking time: ~66-132s of wall-clock where the worker thread does nothing.

### Design

Replace synchronous HTTP calls with a `concurrent.futures.ThreadPoolExecutor` that fires all clip requests in parallel. Ollama's server-side GPU inference is the bottleneck (3-5s per request), not the network — firing 8 concurrent requests costs the same total GPU time but completes in ~5s instead of ~40s.

### Implementation

**ollama_client.py changes:**
- Add a module-level `_OLLAMA_POOL = ThreadPoolExecutor(max_workers=8)` 
- Add `generate_async()` that wraps the existing `generate()` in `_OLLAMA_POOL.submit()`
- Add `generate_batch()` that takes a list of prompts and returns a list of results, all fired concurrently

**llm.py changes:**
- `score_virals()`: Fire all clip prompts concurrently via `generate_batch()`
- `suggest_titles()`: Fire all clip prompts concurrently via `generate_batch()`
- `generate_title()`: Already single-shot; no change

**vlm.py changes:**
- `score_visuals()`: Already uses `run_budgeted()` with `ThreadPoolExecutor`. But `run_budgeted` submits tasks sequentially within the pool. Change to submit all tasks at once before the `wait()` call.

### Thread Safety

Ollama's `/api/generate` is stateless — concurrent requests are safe. The `generate()` function reads the response JSON and returns; it has no side effects. No locks needed.

### Time Impact

| Stage | Before (sequential) | After (parallel) | Saving |
|-------|-------------------|-------------------|--------|
| LLM virality re-rank | 18-36s | 4-8s | 14-28s |
| LLM title suggestions | 18-36s | 4-8s | 14-28s |
| VLM visual scoring | 30-60s | 8-15s | 22-45s |
| **Total** | **66-132s** | **16-31s** | **50-101s** |

---

## 5. Section C: Shared Frame Cache

### Problem

The same source video frames are decoded independently 5-6 times during pipeline processing:

| Consumer | Decode Pattern | Frames |
|----------|---------------|-------|
| Reframe face tracking | Full source at 3 fps → JPEGs | ~10,800 (1hr) |
| VLM scoring | 3 keyframes per clip at 384px | ~30-60 |
| Facecam detection | Full source at 1 fps → frames | ~3,600 |
| OCR | Per-event frame grabs | ~20-100 |
| Thumbnail | 1 frame per clip at 540px | ~10-20 |

Each decode pass runs `ffmpeg` which reads the source file, decompresses the video stream, applies scaling filters, and writes JPEGs. For a 1-hour 1080p H.264 source, each full decode pass takes ~30-60s of CPU time.

### Design

A process-global LRU frame cache keyed by `(source_path, bucket_index)` where `bucket_index = round(timestamp * BUCKET_HZ)`. The cache stores in-memory JPEG bytes at a configurable max width (default 480px).

```
request(t="614.7s", width=480)
  → bucket = round(614.7 * 2) = 1229
  → cache key = ("/media/proj_abc/source.mp4", 1229)
  → hit? return JPEG
  → miss? ffmpeg grab, store, return
```

### Implementation

**New file: `backend/app/media/frame_cache.py`**

```python
"""Shared frame cache — prevents redundant ffmpeg decode across pipeline stages.

Keyed by (source_path, timestamp_bucket). LRU eviction with a configurable
memory limit. Thread-safe via RLock.
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path

_BUCKET_HZ = 2.0        # 0.5s resolution — close timestamps share a cache entry
_MAX_MEM_BYTES = 256 * 1024 * 1024  # 256 MB
_MAX_WIDTH = 480        # downscale width for cached frames (keeps cx fraction valid)

_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_cache_lock = threading.RLock()
_cache_bytes = 0


def _bucket(t: float) -> int:
    return round(t * _BUCKET_HZ)


def get(source_path: str, timestamp: float, width: int = _MAX_WIDTH) -> bytes | None:
    """Return cached JPEG bytes, or None if not in cache."""
    key = (source_path, _bucket(timestamp))
    with _cache_lock:
        data = _cache.get(key)
        if data is not None:
            _cache.move_to_end(key)  # LRU refresh
            return data
        return None


def put(source_path: str, timestamp: float, data: bytes) -> None:
    """Store JPEG bytes in cache, evicting oldest entries if over budget."""
    key = (source_path, _bucket(timestamp))
    with _cache_lock:
        _cache[key] = data
        _cache.move_to_end(key)
        global _cache_bytes
        _cache_bytes += len(data)
        while _cache_bytes > _MAX_MEM_BYTES and len(_cache) > 1:
            oldest_key, oldest_data = _cache.popitem(last=False)
            _cache_bytes -= len(oldest_data)


def clear(source_path: str | None = None) -> None:
    """Clear entire cache, or entries for one source."""
    with _cache_lock:
        if source_path is None:
            _cache.clear()
            global _cache_bytes
            _cache_bytes = 0
        else:
            keys = [k for k in _cache if k[0] == source_path]
            for k in keys:
                _cache_bytes -= len(_cache[k])
                del _cache[k]
```

### Integration Points

**Reframe (`reframe.py`):**
- Before calling `ffmpeg.run()` for frame extraction, check `frame_cache.get(src, t)`
- Store each extracted frame in the cache
- The pre-compute pass (`precompute_face_tracks`) populates the cache at 3 fps across the full source — subsequent per-clip reframe calls hit the cache

**VLM (`vlm.py`):**
- `_grab_frames_b64()`: Check cache before calling `ffmpeg.grab_frame()`
- Cache hit → skip ffmpeg entirely and base64-encode from cache

**Facecam (`facecam.py`):**
- Check cache before extracting frames for face detection
- Store detected face regions alongside frame data (separate key namespace)

**OCR (`detect_ocr.py`):**
- Check cache for frame grabs at event timestamps

### Memory Budget

| Source | Frames | Size/Frame | Total |
|--------|-------|-----------|-------|
| 1hr VOD at 3 fps | 10,800 | ~60 KB (JPEG 480px) | ~648 MB (full) |
| LRU budget | — | — | **256 MB** |
| Typical active set (10 clips × 15s × 3 fps) | ~450 | ~60 KB | ~27 MB |
| + VLM keyframes + OCR frames | ~100 | ~60 KB | ~6 MB |

256 MB is sufficient to cover the active clip working set plus VLM/OCR keyframes. Older clips' frames evict naturally via LRU.

### Time Impact

| Consumer | Before | After | Saving |
|----------|--------|-------|--------|
| Reframe (per clip) | 10-30s decode per clip | <1s (cache hit) | 2-5 min total |
| VLM scoring | ~1s per keyframe grab | ~0.01s (cache hit) | 10-30s |
| Facecam detection | 30-60s full decode | 30-60s (first pass populates cache) | 0s (cached by reframe pre-compute) |
| OCR frame grabs | ~1s per grab | ~0.01s | 5-10s |
| **Total** | **3-7 min** | **30-90s** | **2-5 min** |

---

## 6. Section D: Parallel LLM + VLM

### Problem

`llm_mod.score_virals()` and `vlm_mod.score_visuals()` run sequentially in the score stage:

```
score: [LLM re-rank: 4-8s] → [VLM visual read: 8-15s] → [emotion: 5-10s]
                                                         Total: 17-33s
```

They use different Ollama models (`qwen3:14b` for text, `qwen3-vl:8b` for vision) which run as separate GPU processes. There's no reason they can't run concurrently.

### Design

Fire LLM and VLM calls at the same time using `concurrent.futures`. The orchestrator's `_process()` score stage becomes:

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    llm_future = pool.submit(_do_llm_scoring, ...) if llm_mod.available() else None
    vlm_future = pool.submit(_do_vlm_scoring, ...) if vlm_mod.available() else None
    # Do emotion scoring (CPU, can run in parallel) while LLM/VLM are in-flight
    # ...
    llm_result = llm_future.result() if llm_future else {}
    vlm_result = vlm_future.result() if vlm_future else {}
```

### Implementation

**orchestrator.py changes:**
- Wrap the LLM and VLM blocks in `_score_stage_llm()` and `_score_stage_vlm()` helper methods
- Fire both via `ThreadPoolExecutor` in `_process()`
- Run the emotion and audio-event scoring (CPU-bound, no GPU contention) in the main thread while LLM/VLM run

### Safety

Both scoring functions are read-only (they read transcript text and source frames) and write results to clip objects in memory. Each clip object is only modified by one function at a time (LLM sets `clip.score`, VLM sets `clip.score` with `apply_viral_boost` which reads-modifies-writes). The `apply_viral_boost` function is:

```python
def apply_viral_boost(score, factors, viral, reason):
    boost = (viral - 0.5) * 0.2  # ±10 pts max
    return score + boost, {**factors, "viral": reason}
```

This is a pure function — safe for concurrent calls since each clip has its own score/factors values. The LLM and VLM modify different clips or different aspects of the same clip, but to be safe we should use per-clip locks or consolidate boosts after both futures complete.

**Better approach:** Both LLM and VLM return `dict[int, tuple[float, str]]` (index → (viral, reason)). Apply both sets of boosts **after** both futures complete, in a single pass:

```python
llm_reads = llm_future.result()  # {0: (0.7, "strong hook"), ...}
vlm_reads = vlm_future.result()  # {0: (0.8, "intense action"), ...}
for i, clip in enumerate(clips):
    if i in llm_reads:
        clip.score, clip.factors = score_mod.apply_viral_boost(...)
    if i in vlm_reads:
        clip.score, clip.factors = score_mod.apply_viral_boost(...)
```

### Time Impact

| Stage | Before (sequential) | After (parallel) | Saving |
|-------|-------------------|-------------------|--------|
| LLM + VLM combined | 17-33s | 8-15s (max of both) | 9-18s |
| Other CPU scoring runs during | idle | in parallel | +5-10s hidden |
| **Net** | **17-33s** | **8-15s** | **9-18s** |

---

## 7. Section E: Source-Based Thumbnails

### Problem

`render.py:_make_thumbnail()` extracts a face-picked thumbnail from the **rendered output** clip. This requires ffmpeg to decode the freshly-encoded H.264/NVENC output, which adds a full decode pass for each clip. For 10-20 clips at 5-10s each that's 50-200s of unnecessary decode.

### Design

Extract thumbnails from the **source video** at the clip's midpoint timestamp. Store the thumbnail path in the clip model ahead of render. The thumbnail becomes available the moment the clip is detected, not after it's rendered.

### Implementation

**orchestrator.py changes:**
- After clips are created and reframed (stage 3-4), pre-generate thumbnails:
  ```python
  for clip in clips:
      thumb_path = ingest.project_dir(project_id) / "clips" / f"{clip.id}.jpg"
      at = clip.start + (clip.end - clip.start) * 0.35
      ffmpeg.make_thumbnail(src_path, thumb_path, at=at, width=540)
  ```
- Store `thumb_path` on the clip model

**render.py changes:**
- `render_clip()`: Remove the `_make_thumbnail(out_path, thumb_path, ...)` call at the end
- The thumbnail is already in place from the pre-generation

**Clip model update (models.py):**
- Add `thumb_url: str | None = None` field (already exists via `export_url`/`thumb_url` pattern)

### Quality Consideration

Source thumbnails are **better** than rendered-output thumbnails because:
- Source video has higher bitrate (not re-compressed through NVENC)
- Thumbnail is available immediately after detection (UX improvement)
- Face detection for thumbnail picking runs on the source frames, which are already in the frame cache

### Time Impact

| Clip count | Before | After | Saving |
|-----------|--------|-------|--------|
| 10 clips | 50-100s | 10-20s (source grab) | 40-80s |
| 20 clips | 100-200s | 20-40s | 80-160s |

The source frame is likely already in the **frame cache** (Section C) from the reframe pre-compute pass, making the thumbnail grab essentially free.

---

## 8. Section F: Scaling Knobs & Concurrency

### Problem

Fixed/conservative caps limit throughput even when GPU resources are available:

| Cap | Current | Max for RTX 5060 Ti | Impact |
|-----|---------|---------------------|--------|
| `_RENDER_SEMAPHORE` | 4 | 6-8 NVENC sessions | Render queue depth |
| `_auto_workers()` | max(2, min(cpu//4, 4)) | 6-8 | Parallel clip rendering |
| `render_workers_for("max_gpu")` | max(1, min(workers, 8)) | 12 | Upper bound |
| `pipeline_workers` | 1 (or env override) | 2-3 | Concurrent projects |
| whisper batch size | Auto to VRAM ✓ | 48 (16GB) | Already done |

### Changes

**`render.py`:** `_RENDER_SEMAPHORE` → dynamic based on VRAM

The current `_RENDER_SEMAPHORE = threading.Semaphore(4)` is a module-level constant initialized at import time, before settings are loaded. Replace with a lazy-initialized semaphore:

```python
_RENDER_SEMAPHORE: threading.Semaphore | None = None
_RENDER_SEM_LOCK = threading.Lock()

def _get_render_sem() -> threading.Semaphore:
    global _RENDER_SEMAPHORE
    if _RENDER_SEMAPHORE is None:
        with _RENDER_SEM_LOCK:
            if _RENDER_SEMAPHORE is None:
                vram = get_settings().vram_mb
                # RTX 5060 Ti supports ~6 NVENC sessions; scale down for smaller cards.
                n = 6 if vram >= 16000 else 4 if vram >= 8000 else 2
                _RENDER_SEMAPHORE = threading.Semaphore(n)
    return _RENDER_SEMAPHORE
```

Then replace every `with _RENDER_SEMAPHORE:` with `with _get_render_sem():`.

**`config.py`:** `_auto_workers()` → scale higher on GPU systems
```python
def _auto_workers(cpu: int, has_nvidia: bool = False) -> int:
    if has_nvidia:
        return max(2, min(cpu // 2, 8))  # NVENC offloads encode, CPU free for scoring
    return max(2, min(cpu // 4, 4))
```

Then in `get_settings()`, pass `has_nvidia`:
```python
workers_env = os.environ.get("CLIPFORGE_RENDER_WORKERS")
render_workers = int(workers_env) if workers_env else _auto_workers(cpu, has_nvidia=has_nvidia)
```

**`config.py`:** `pipeline_workers` → auto-scale by VRAM
```python
# In get_settings():
pipeline_env = os.environ.get("CLIPFORGE_PIPELINE_WORKERS")
if pipeline_env:
    pipeline_workers = int(pipeline_env)
else:
    pipeline_workers = max(1, min(vram_mb // 8000, 3)) if has_nvidia else 1
```

This gives:
- 16 GB VRAM → 2 pipeline workers
- 24 GB VRAM → 3 pipeline workers
- No GPU → 1 pipeline worker

### Time Impact

| Cap Change | Saving (1hr VOD) | Saving (batch) |
|-----------|-----------------|----------------|
| Render semaphore 4→6 | 1-2 min | 2-3 min |
| Auto workers 4→6 | 30s-1min | 1-2 min |
| Pipeline workers 1→2 | 0 (single VOD) | 30-50% batch time |
| **Total** | **1.5-3 min** | **Significant for batches** |

---

## 9. Implementation Order

| Step | Section | Effort | Impact | Risk | Dependencies |
|------|---------|--------|--------|------|-------------|
| 1 | F: Scaling knobs | Low | Medium | Low | None |
| 2 | B: Async Ollama | Medium | High | Low | None |
| 3 | D: Parallel LLM+VLM | Low | Medium | Low | Step 2 |
| 4 | C: Shared frame cache | Medium | High | Medium | None |
| 5 | E: Source thumbnails | Low | Medium | Low | Step 4 (optional) |
| 6 | A: Dedicated ASR thread | High | High | Medium | None |

Each step is independently testable and reversible. Steps 1-5 can be implemented and shipped before step 6.

---

## 10. Quality Guarantees

| Optimization | Quality Impact | Mitigation |
|-------------|---------------|------------|
| Async ASR thread | None | Same model, same params, same code path |
| Async Ollama | None | Same prompts, same temperature/num_predict |
| Shared frame cache | None | Lossless JPEG path; same frames as direct ffmpeg |
| Parallel LLM+VLM | None | Boosts applied post-merge (no race conditions) |
| Source thumbnails | **Improved** | Source has higher bitrate than re-encoded output |
| More render workers | None | Same NVENC preset/quality params |
| Bigger semaphore | None | NVENC sessions are independent hardware units |

### Testing Strategy

Each optimization includes:

1. **Unit tests** for new components (frame_cache hit/miss/eviction, ASR job lifecycle)
2. **Existing integration tests** must pass unchanged (the pipeline produces identical outputs)
3. **Smoke test** with a 5-min source video measuring end-to-end time before/after

The existing `test_smoke.py` and `test_units.py` (205 tests) provide the regression baseline.

---

*End of spec*
