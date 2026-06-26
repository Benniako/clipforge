"""Shared LRU frame cache — prevents redundant ffmpeg decode across pipeline stages.

Keyed by ``(source_path, timestamp_bucket)`` where the bucket rounds to the
nearest 0.5 s window.  Stores in-memory JPEG bytes at 480 px width.  Thread-safe
via an :class:`threading.RLock`.

The typical decode chain is:

    VLM score  ── 3 keyframes/clip  → grab_frame()
    Facecam    ── 1 fps samples     → grab_frame()
    Reframe    ── 3 fps full source → custom ffmpeg.run()
    OCR batch  ── per-project frames → grab_frame()

Without a cache every caller re-decodes the same video spans independently —
for a 60 min VOD that's several minutes of redundant CPU/GPU decode.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
_BUCKET_HZ = 2.0          # 0.5 s resolution — close timestamps share an entry
_MAX_MEM_BYTES = 256 * 1024 * 1024  # 256 MiB
_MAX_WIDTH = 480          # downscale width for cached frames


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
_cache_lock = threading.RLock()
_cache_bytes = 0


def _bucket(t: float) -> int:
    return round(t * _BUCKET_HZ)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def get(source_path: str, timestamp: float) -> bytes | None:
    """Return cached JPEG bytes at ``(source_path, timestamp)``, or ``None``."""
    key = (source_path, _bucket(timestamp))
    with _cache_lock:
        data = _cache.get(key)
        if data is not None:
            _cache.move_to_end(key)  # LRU refresh
            return data
        return None


def put(source_path: str, timestamp: float, data: bytes) -> None:
    """Store JPEG bytes, evicting oldest entries if over budget."""
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
    """Drop all entries, or only those belonging to a specific source."""
    with _cache_lock:
        global _cache_bytes
        if source_path is None:
            _cache.clear()
            _cache_bytes = 0
        else:
            keys = [k for k in _cache if k[0] == source_path]
            for k in keys:
                _cache_bytes -= len(_cache[k])
                del _cache[k]
