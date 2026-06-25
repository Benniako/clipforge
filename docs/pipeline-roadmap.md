# ClipForge Pipeline Improvement Roadmap

## Executive Summary

Three research agents audited ClipForge across **performance**, **architecture**, and **competitive positioning**. This roadmap synthesizes their findings into a prioritized action plan.

**Total issues identified: 54**
- Performance bottlenecks: 16 (3 critical, 5 major, 6 medium, 2 minor)
- Architecture/code quality issues: 23 (5 high, 8 medium, 10 low)
- Competitive gaps vs OpusClip/Submagic/Descript: 15

---

## Phase A: Quick Wins (2-4 hours each)

### A1. Eliminate Redundant ffmpeg Decodes
**Source:** Performance bottlenecks #1, #4, #5, #6, #7
**Speedup:** 3-10x on non-transcription stages
**Effort:** 2-3 hours

**The problem:** The pipeline calls ffmpeg separately for every clip — reframe face tracking, scene cuts, audio events. On 20 clips, the same source is decoded 20+ times.

**Fix:**
1. **Reframe** (`reframe.py`): Extract all frame timestamps for all clips in one ffmpeg pass, run face detection once, then let each clip reference the pre-computed face centers
2. **Scene cuts** (`orchestrator.py:548-555`): Run `scene_cuts()` once on the full timeline, cache results, query per clip instead of per-clip ffmpeg calls
3. **Audio events** (`audio_events.py:593-612`): Decode full audio to numpy array once, slice in-memory per clip instead of per-clip ffmpeg extraction

### A2. Remove Duplicate Keyboard Handler (DONE ✓)
Already fixed in commit `535e5a7`.

### A3. Batch Progress Writes to SQLite
**Source:** Performance #16
**Effort:** 1 hour

**Fix:** In `orchestrator.py:_advance()`, only call `store.mutate()` when progress crosses integer thresholds (e.g., 10%, 20%) or every 5 seconds, not on every fractional change. Use an in-memory counter.

### A4. Fix Dead Code: `ai_boost` Not Sent to Backend
**Source:** Architecture #5.2
**Effort:** 30 min

**Fix:** In `frontend/src/lib/api.ts`, serialize `ai_boost` field into FormData, or remove the field from `CreateProjectInput`.

---

## Phase B: Pipeline Performance (8-16 hours)

### B1. Decouple the Global `TORCH_LOAD_LOCK`
**Source:** Performance #3
**Speedup:** 1.5-2x for multi-worker
**Effort:** 2-3 hours

**Problem:** Single global lock serializes all PyTorch model loading. CLAP's `torch.load` monkey-patch is the reason.
**Fix:** Replace global lock with per-provider locks. Move CLAP loading to a daemon thread that runs during pipeline init. Use a separate subprocess for CLAP audio embedding.

### B2. Parallel Audio Event Scoring
**Source:** Performance #5
**Speedup:** 3-5x
**Effort:** 2-3 hours

**Fix:** 
- Batch PANNs inference: accumulate audio arrays, call `tagger.inference()` once with shape `(N, samples)`
- Parallelize ffmpeg extraction across clips using `ThreadPoolExecutor`

### B3. Pre-compute Speech Intervals Once
**Source:** Performance #7
**Speedup:** 5-10x
**Effort:** 1 hour

**Fix:** Compute VAD speech intervals for the full timeline once after transcription, store as `list[tuple[float, float]]`, then each clip just filters and rebases.

### B4. Optimizer Render Worker Count
**Source:** Performance #14
**Speedup:** 1.5-3x on high-core CPUs
**Effort:** 30 min

**Fix:** In `config.py`, raise `render_workers` cap to `cpu_count` on x264 path. Split CPU vs NVENC worker counts.

### B5. Reduce Thumbnail Frame Samples
**Source:** Performance #10
**Speedup:** 2-3x
**Effort:** 30 min

**Fix:** Reduce `_pick_thumbnail_at` samples from 7 to 3.

---

## Phase C: Architecture (12-24 hours)

### C1. Decompose `Engine._process()` — 470-line God Function
**Source:** Architecture #1.1, #2.3
**Severity:** HIGH
**Effort:** 4-6 hours

**The problem:** `_process()` handles all 6 pipeline stages with deeply nested conditionals, inline try/except, and late imports. It's untestable in isolation.

**Fix:** Extract each stage into its own method:
```python
def _stage_transcribe(self, project_id, src_path, info) -> Transcript
def _stage_detect(self, project_id, transcript, info) -> list[Clip]
def _stage_score(self, project_id, clips, transcript) -> list[Clip]
def _stage_reframe(self, project_id, clips, src_path) -> None
def _stage_caption(self, project_id, clips, transcript) -> None
def _stage_render(self, project_id, clips, src_path, info) -> None
```

### C2. Replace Global `_write_lock` with Per-Project Locks
**Source:** Architecture #4.1
**Severity:** HIGH
**Effort:** 2-3 hours

**Fix:** Switch `store.py` from single `_write_lock` to per-project locks (`_project_locks: dict[str, threading.Lock]`). Add a lightweight progress-update path that only writes the `progress` field via narrow SQL `UPDATE` instead of full document serialization.

### C3. Add Integration Tests for Critical Paths
**Source:** Architecture #3.1, #3.2, #3.3
**Severity:** HIGH
**Effort:** 4-8 hours

**Missing tests:**
- `Engine._process()` — mock all providers, test with synthetic data
- `create_project` API endpoint — file upload, URL import, error paths
- `rerender_clip` / `rerender_all` — mocked store + ffmpeg
- `store.mutate()` — atomicity under exception, concurrent access

### C4. Remove Threading Leaks from API Layer
**Source:** Architecture #1.3
**Effort:** 2-3 hours

**Fix:** Move `threading.Thread(target=engine.rerender_...)` calls behind `Engine.rerender_clip_async()` methods. The API calls should never manage threads directly.

### C5. Fix TOCTOU Race in `_wait_if_paused`
**Source:** Architecture #2.7
**Effort:** 1 hour

**Fix:** Move the `project_id in self._pause_requested` check inside the `with self._pause_condition:` block.

---

## Phase D: Feature Gaps vs SaaS (Ongoing)

### D1. Multi-Modal Moment Detection
**Source:** Competitive gap #1
**Effort:** 3-6 months (research phase first)

**Current:** Only text/transcript-based moment detection.
**Target:** Combine audio (tone, pace, volume) + visual (facial expressions on-screen action) + text signals into a unified moment score.
**First step:** Add audio-energy as a moment signal (speaker raises voice → potential highlight).

### D2. Dynamic Caption Template Library
**Source:** Competitive gap #3
**Effort:** 2-4 weeks

**Current:** 9 static templates in `styles.py`.
**Target:** 25+ animated templates with brand kit support (logo, custom font, colors).
**Low-hanging fruit:** Port the existing ASS templates to include subtle animations (slide-in, fade, bounce for emphasis words).

### D3. One-Click Social Publishing
**Source:** Competitive gap #4
**Effort:** 2-4 weeks per platform

**Current:** Manual download and re-upload.
**Target:** Direct publish button for TikTok, YouTube Shorts, Instagram Reels.
**First step:** Build the auth + upload API for one platform (TikTok has the best developer docs).

### D4. REST API + MCP Server
**Source:** Competitive gap #7
**Effort:** 2-4 weeks

**Current:** Basic REST API for frontend consumption.
**Target:** Versioned public API + MCP server for Claude/Cursor/IDE agent integration. Enables "AI agent processes your video" use case.

### D5. Batch Export and Multi-Format Output
**Source:** Competitive gap — output section
**Effort:** 1-2 weeks

**Current:** Single-format MP4 export.
**Target:** Simultaneous export in multiple aspect ratios (9:16 + 16:9 + 1:1), configurable codec/quality, ZIP batch download.

---

## Phase E: Testing & QA (8-16 hours)

### E1. Frontend Component Tests
**Source:** Architecture #3.5
**Effort:** 4-8 hours

Add tests for:
- `api.ts` — mock fetch, test every endpoint function
- `ClipEditor.tsx` — keyboard shortcuts, undo/redo, trim logic
- `Upload.tsx` — form validation, file/URL submission

### E2. Pipeline Stage Unit Tests
**Source:** Architecture #3.1
**Effort:** 4-8 hours

Test each stage in isolation with mocked dependencies:
- `_stage_transcribe`: Feed a WAV file path, expect a Transcript
- `_stage_detect`: Feed a Transcript, expect a list of Clips
- `_stage_score`: Feed Clips, verify score ranges and factors

### E3. Store Edge Case Tests
**Source:** Architecture #3.4
**Effort:** 1-2 hours

Test `mutate()` under: concurrent access, exception during mutation, nested contexts, deletion during mutation.

---

## Priority Matrix

```
                    LOW EFFORT           HIGH EFFORT
                   ┌──────────────────────────────────┐
  HIGH IMPACT      │ A1 ffmpeg decode    │ C1 Decompose _process()  │
                   │ A3 Batch progress   │ C2 Per-project locks     │
                   │ B3 Speech intervals │ C3 Integration tests     │
                   │ B4 Render workers   │ D2 Caption templates     │
                   │ B5 Thumbnail frames │ D4 REST API + MCP       │
                   ├──────────────────────────────────┤
  MEDIUM IMPACT    │ A4 Dead code fix    │ D1 Multi-modal detection │
                   │ B1 TORCH_LOAD_LOCK  │ D3 Social publishing     │
                   │ B2 Parallel scoring │ E1 Frontend tests        │
                   │ C4 Thread leak fix  │ E2 Pipeline unit tests   │
                   │ C5 TOCTOU fix       │ D5 Multi-format export   │
                   └──────────────────────────────────┘
```

## Recommended Sprint Plan

| Sprint | Focus | Items | Est. Hours |
|--------|-------|-------|------------|
| **1** | Quick wins | A1, A3, A4, B3, B4, B5 | 6-8 |
| **2** | Performance | B1, B2, C5 | 6-8 |
| **3** | Architecture | C1 (decompose _process) | 6-8 |
| **4** | Architecture | C2 (per-project locks), C4 | 6-8 |
| **5** | Testing | C3, E1, E2, E3 | 8-12 |
| **6** | Features | D2, D5 | 8-12 |
| **7** | Features | D4 (REST API + MCP) | 8-12 |
| **8+** | Moonshots | D1, D3 | Ongoing |

**First sprint priority:** A1 (eliminate redundant ffmpeg decodes) — single biggest performance win with minimal code change.
