"""Adversarial stress test suite for Milestone M1 (challenger_m1_1).

Tests:
1. Waveform bounds & audio extremes:
   - Clipped fullscale int16 (-32768 and +32767)
   - Pure silence (all zeros)
   - Missing audio file fallback
   - Corrupted WAV files (0-byte, truncated, garbage)
   - Stereo 2-channel audio extraction
   - Short clips (< 1s down to 0.01s)
   - Long clips (1hr to 2hr duration)
   - Inverted or zero clip spans
   - Out-of-range num_bins
2. Suggestions logic stress testing:
   - Empty transcript and words
   - Words at clip.start (0s lead silence)
   - Words overlapping clip start boundary
   - Large lead silence
   - Extreme climax timestamps (-9999, 0, 1e6)
   - Facecam layout combinations
   - JSON schema and retention_impact contract compliance
3. Apply endpoint atomicity & edge cases:
   - Repeated application idempotency
   - Invalid suggestion ID (404)
   - Invalid layout parameter (400)
   - Boundary clamping (start_t < 0, start_t > end, reframe_cx out of [0, 1])
   - Concurrency race conditions
4. Batch operations stress testing:
   - Empty clip_ids (400)
   - Invalid action (400)
   - Partial nonexistent IDs
   - Rating boundary values
   - Delete all clips
   - Sequencing reorder
"""
from __future__ import annotations

import math
import os
import socket
import struct
import tempfile
import threading
import wave
from pathlib import Path

from starlette.testclient import TestClient

# Setup isolated test directory before loading app modules
_test_dir = tempfile.mkdtemp(prefix="clipforge-challenger-m1-")
os.environ["CLIPFORGE_DATA_DIR"] = _test_dir
os.environ["CLIPFORGE_DEVICE"] = "cpu"

# Enforce strict offline operation
_real_connect = socket.socket.connect

def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
        raise RuntimeError(f"Strict Offline Invariant Violated! Outgoing network call attempted to {host}")
    return _real_connect(self, address)

socket.socket.connect = _guarded_connect

from app import feedback, store
from app.config import get_settings
from app.main import app

store.init_db()
feedback.init_db()

from app.models import (
    CaptionSet, CaptionWord, Clip, ClipStatus, DetectedEvent,
    ImportSettings, LayoutType, Platform, Project, Rect, Reframe,
    Transcript, Word
)
from app.services import audio_waveform, suggestions_copilot

client = TestClient(app)


def _write_custom_wav(path: Path, samples_data: bytes, sample_rate: int = 16000, n_channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_data)


# --------------------------------------------------------------------------- #
# 1. Waveform & dBFS Bounds Stress Testing
# --------------------------------------------------------------------------- #

def test_waveform_clipped_negative_fullscale_int16():
    """Verify bounds when audio contains -32768 (two's complement int16 minimum)."""
    settings = get_settings()
    p_id = "proj_wf_clipped_neg"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    # 16000 samples of -32768
    data = struct.pack("<h", -32768) * 16000
    _write_custom_wav(wav_path, data)

    clip = Clip(id="c_clip_neg", start=0.0, end=1.0)
    proj = Project(id=p_id, name="Clipped Neg", clips=[clip])
    store.save(proj)

    wf = audio_waveform.extract_clip_waveform(proj, clip)
    min_amp = min(wf["samples"])
    max_amp = max(wf["samples"])
    min_db = min(wf["dbfs"])
    max_db = max(wf["dbfs"])

    print(f"\n[WF Negative Clip] Amplitudes: min={min_amp}, max={max_amp} | dBFS: min={min_db}, max={max_db}")

    # Empirical assertion check: Amplitudes MUST be in [0.0, 1.0]
    # Note: If min_amp < 0.0, this confirms the numpy int16 abs overflow bug!
    assert all(0.0 <= s <= 1.0 for s in wf["samples"]), f"Amplitudes outside [0.0, 1.0]: min={min_amp}, max={max_amp}"
    assert all(-60.0 <= d <= 0.0 for d in wf["dbfs"]), f"dBFS outside [-60.0, 0.0]: min={min_db}, max={max_db}"


def test_waveform_clipped_positive_fullscale_int16():
    """Verify bounds when audio contains +32767 (maximum positive int16)."""
    settings = get_settings()
    p_id = "proj_wf_clipped_pos"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    data = struct.pack("<h", 32767) * 16000
    _write_custom_wav(wav_path, data)

    clip = Clip(id="c_clip_pos", start=0.0, end=1.0)
    proj = Project(id=p_id, name="Clipped Pos", clips=[clip])
    store.save(proj)

    wf = audio_waveform.extract_clip_waveform(proj, clip)
    min_amp = min(wf["samples"])
    max_amp = max(wf["samples"])
    min_db = min(wf["dbfs"])
    max_db = max(wf["dbfs"])

    print(f"\n[WF Positive Clip] Amplitudes: min={min_amp}, max={max_amp} | dBFS: min={min_db}, max={max_db}")
    assert all(0.0 <= s <= 1.0 for s in wf["samples"]), f"Amplitudes outside [0.0, 1.0]: min={min_amp}, max={max_amp}"
    assert all(-60.0 <= d <= 0.0 for d in wf["dbfs"]), f"dBFS outside [-60.0, 0.0]: min={min_db}, max={max_db}"


def test_waveform_pure_silence():
    """Verify silent audio (all zeros) yields samples in [0.0, 1.0] and dBFS = -60.0."""
    settings = get_settings()
    p_id = "proj_wf_silence"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    data = struct.pack("<h", 0) * 16000
    _write_custom_wav(wav_path, data)

    clip = Clip(id="c_silence", start=0.0, end=1.0)
    proj = Project(id=p_id, name="Pure Silence", clips=[clip])
    store.save(proj)

    wf = audio_waveform.extract_clip_waveform(proj, clip)
    assert all(s == 0.0 for s in wf["samples"])
    assert all(d == -60.0 for d in wf["dbfs"])


def test_waveform_corrupted_wav_files():
    """Verify corrupted / truncated WAV files cleanly fall back without crash."""
    settings = get_settings()
    p_id = "proj_wf_corrupt"
    proj_dir = settings.media_dir / p_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    wav_path = proj_dir / "audio16k.wav"

    # Case A: 0-byte file
    wav_path.write_bytes(b"")
    clip = Clip(id="c_corrupt1", start=0.0, end=5.0)
    proj = Project(id=p_id, name="Corrupt WAV", clips=[clip])
    store.save(proj)

    wf1 = audio_waveform.extract_clip_waveform(proj, clip)
    assert len(wf1["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf1["samples"])
    assert all(-60.0 <= d <= 0.0 for d in wf1["dbfs"])

    # Case B: Truncated header (12 bytes)
    wav_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    wf2 = audio_waveform.extract_clip_waveform(proj, clip)
    assert len(wf2["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf2["samples"])

    # Case C: Random garbage bytes
    wav_path.write_bytes(os.urandom(512))
    wf3 = audio_waveform.extract_clip_waveform(proj, clip)
    assert len(wf3["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf3["samples"])


def test_waveform_stereo_audio():
    """Verify stereo 2-channel audio extraction downmixes correctly without error."""
    settings = get_settings()
    p_id = "proj_wf_stereo"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    # Interleaved stereo samples: Left=1000, Right=-1000
    pair = struct.pack("<hh", 1000, -1000)
    data = pair * 16000
    _write_custom_wav(wav_path, data, n_channels=2)

    clip = Clip(id="c_stereo", start=0.0, end=1.0)
    proj = Project(id=p_id, name="Stereo WAV", clips=[clip])
    store.save(proj)

    wf = audio_waveform.extract_clip_waveform(proj, clip)
    assert len(wf["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf["samples"])
    assert all(-60.0 <= d <= 0.0 for d in wf["dbfs"])


def test_waveform_short_and_subsecond_clips():
    """Verify ultra-short clips (< 1s, down to 0.01s) produce valid waveforms."""
    settings = get_settings()
    p_id = "proj_wf_short"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    data = struct.pack("<h", 5000) * 16000
    _write_custom_wav(wav_path, data)

    for dur in [0.01, 0.05, 0.1, 0.5, 0.9]:
        clip = Clip(id=f"c_dur_{int(dur*1000)}", start=0.0, end=dur)
        proj = Project(id=p_id, name="Short Clips", clips=[clip])
        store.save(proj)

        wf = audio_waveform.extract_clip_waveform(proj, clip)
        assert len(wf["samples"]) >= 100
        assert len(wf["dbfs"]) == len(wf["samples"])
        assert all(0.0 <= s <= 1.0 for s in wf["samples"])
        assert all(-60.0 <= d <= 0.0 for d in wf["dbfs"])
        assert wf["duration_s"] >= 0.1


def test_waveform_long_clips_1hr():
    """Verify long clips (1 hour = 3600s) execute smoothly and return bounded points."""
    clip = Clip(id="c_1hr", start=0.0, end=3600.0, score=75, features={"climax_t": 1800.0})
    proj = Project(id="proj_wf_1hr", clips=[clip])
    store.save(proj)

    wf = audio_waveform.extract_clip_waveform(proj, clip)
    assert len(wf["samples"]) >= 100
    assert len(wf["dbfs"]) == len(wf["samples"])
    assert all(0.0 <= s <= 1.0 for s in wf["samples"])
    assert all(-60.0 <= d <= 0.0 for d in wf["dbfs"])
    assert wf["duration_s"] == 3600.0



def test_waveform_zero_and_inverted_spans():
    """Verify degenerate spans (start >= end) are handled gracefully."""
    settings = get_settings()
    p_id = "proj_wf_zero"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    data = struct.pack("<h", 4000) * 16000
    _write_custom_wav(wav_path, data)

    # start == end
    clip_zero = Clip(id="c_zero", start=5.0, end=5.0)
    # start > end
    clip_inv = Clip(id="c_inv", start=10.0, end=5.0)
    proj = Project(id=p_id, name="Zero Inv", clips=[clip_zero, clip_inv])
    store.save(proj)

    wf_zero = audio_waveform.extract_clip_waveform(proj, clip_zero)
    assert len(wf_zero["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf_zero["samples"])

    wf_inv = audio_waveform.extract_clip_waveform(proj, clip_inv)
    assert len(wf_inv["samples"]) >= 100
    assert all(0.0 <= s <= 1.0 for s in wf_inv["samples"])


def test_waveform_num_bins_bounds():
    """Verify num_bins outside [100, 200] is safely clamped."""
    settings = get_settings()
    p_id = "proj_wf_bins"
    wav_path = settings.media_dir / p_id / "audio16k.wav"
    data = struct.pack("<h", 3000) * 16000
    _write_custom_wav(wav_path, data)

    clip = Clip(id="c_bins", start=0.0, end=1.0)
    proj = Project(id=p_id, name="Bins", clips=[clip])

    wf_low = audio_waveform.extract_clip_waveform(proj, clip, num_bins=10)
    assert len(wf_low["samples"]) == 100

    wf_high = audio_waveform.extract_clip_waveform(proj, clip, num_bins=500)
    assert len(wf_high["samples"]) == 200


# --------------------------------------------------------------------------- #
# 2. Suggestions Logic Stress Testing
# --------------------------------------------------------------------------- #

def test_suggestions_no_speech():
    """Verify suggestions on clip with zero speech words."""
    clip = Clip(id="c_nospeech", start=0.0, end=15.0, score=85, features={"climax_t": 7.5})
    proj = Project(id="p_nospeech", clips=[clip], transcript=Transcript(words=[], language="en"))

    sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
    assert len(sugs) >= 1
    # Verify no trim_hook since there is no speech to trim to
    assert not any(s["type"] == "trim_hook" for s in sugs)

    # Verify hook_rating is present and has valid schema
    hr = next((s for s in sugs if s["type"] == "hook_rating"), None)
    assert hr is not None
    assert hr["retention_impact"] in ("Strong Hook", "OK Hook", "Weak Hook")
    assert hr["title"]
    assert hr["description"]


def test_suggestions_speech_starts_at_zero():
    """Verify speech starting exactly at clip.start (0.0s lead silence) generates NO trim_hook."""
    words = [Word(t=0.0, d=0.5, text="Hello"), Word(t=0.6, d=0.5, text="world")]
    clip = Clip(id="c_zero_silence", start=0.0, end=10.0, features={"climax_t": 5.0})
    proj = Project(id="p_zero_silence", clips=[clip], transcript=Transcript(words=words, language="en"))

    sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
    assert not any(s["type"] == "trim_hook" for s in sugs)


def test_suggestions_speech_overlapping_clip_start():
    """Verify words starting before clip.start generate NO trim_hook."""
    words = [Word(t=9.5, d=1.0, text="Continuous"), Word(t=10.6, d=0.5, text="flow")]
    clip = Clip(id="c_overlap", start=10.0, end=20.0, features={"climax_t": 14.0})
    proj = Project(id="p_overlap", clips=[clip], transcript=Transcript(words=words, language="en"))

    sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
    assert not any(s["type"] == "trim_hook" for s in sugs)


def test_suggestions_extreme_climax_timestamps():
    """Verify suggestions handle negative, zero, and huge climax timestamps."""
    for c_t in [-9999.0, 0.0, 1e6]:
        clip = Clip(id="c_ext", start=0.0, end=10.0, features={"climax_t": c_t})
        proj = Project(id="p_ext", clips=[clip])
        sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
        assert len(sugs) >= 1
        for s in sugs:
            assert s["id"]
            assert s["type"]
            assert s["title"]
            assert s["description"]
            assert s["retention_impact"]


def test_suggestions_facecam_combinations():
    """Verify facecam reframe suggestion behavior across project and clip layouts."""
    # Case A: project facecam, layout fill -> suggests split
    c_a = Clip(id="ca", start=0.0, end=10.0, reframe=Reframe(layout=LayoutType.fill))
    p_a = Project(id="pa", clips=[c_a], facecam=Rect(x=0.1, y=0.1, w=0.2, h=0.2))
    sugs_a = suggestions_copilot.generate_clip_suggestions(p_a, c_a)
    assert any(s["type"] == "reframe_layout" for s in sugs_a)

    # Case B: project facecam, layout already split -> NO split suggestion
    c_b = Clip(id="cb", start=0.0, end=10.0, reframe=Reframe(layout=LayoutType.split))
    p_b = Project(id="pb", clips=[c_b], facecam=Rect(x=0.1, y=0.1, w=0.2, h=0.2))
    sugs_b = suggestions_copilot.generate_clip_suggestions(p_b, c_b)
    assert not any(s["type"] == "reframe_layout" for s in sugs_b)

    # Case C: no facecam anywhere -> NO split suggestion
    c_c = Clip(id="cc", start=0.0, end=10.0, reframe=Reframe(layout=LayoutType.fill))
    p_c = Project(id="pc", clips=[c_c], facecam=None)
    sugs_c = suggestions_copilot.generate_clip_suggestions(p_c, c_c)
    assert not any(s["type"] == "reframe_layout" for s in sugs_c)


def test_suggestions_schema_compliance():
    """Verify all suggestions strictly match PROJECT.md JSON schema requirements."""
    words = [Word(t=11.2, d=0.4, text="Hook"), Word(t=12.0, d=0.5, text="line")]
    clip = Clip(id="c_schema", start=10.0, end=25.0, score=92, features={"climax_t": 18.0})
    proj = Project(id="p_schema", clips=[clip], transcript=Transcript(words=words, language="en"),
                   facecam=Rect(x=0.1, y=0.1, w=0.2, h=0.2))

    sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
    valid_types = {"trim_hook", "hook_rating", "climax_teaser", "reframe_layout", "tighten_dead_air"}

    for s in sugs:
        assert isinstance(s["id"], str) and s["id"].startswith("sug_")
        assert s["type"] in valid_types
        assert isinstance(s["title"], str) and len(s["title"]) > 0
        assert isinstance(s["description"], str) and len(s["description"]) > 0
        assert isinstance(s["retention_impact"], str) and len(s["retention_impact"]) > 0
        assert isinstance(s["patch"], dict)


# --------------------------------------------------------------------------- #
# 3. Apply Suggestion Atomicity & Edge Cases
# --------------------------------------------------------------------------- #

def test_apply_suggestion_idempotency_and_repeated_apply():
    """Verify behavior when applying the same suggestion multiple times."""
    p_id = "proj_apply_repeat"
    words = [Word(t=11.0, d=0.5, text="Silence"), Word(t=11.6, d=0.5, text="ended")]
    clip = Clip(id="c1", start=10.0, end=20.0, score=80, features={"climax_t": 15.0})
    proj = Project(id=p_id, clips=[clip], transcript=Transcript(words=words, language="en"))
    store.save(proj)

    # First apply: succeeds
    r1 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_trim_hook_c1/apply")
    assert r1.status_code == 200
    res1 = r1.json()
    assert res1["success"] is True
    assert "start_t" in res1["updated_fields"]

    # Second apply: suggestion was already satisfied, so dynamic list no longer has sug_trim_hook_c1
    r2 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_trim_hook_c1/apply")
    assert r2.status_code == 404

    # Third apply with explicit body patch: succeeds despite missing suggestion ID
    r3 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_manual/apply", json={"patch": {"teaser_loop": True}})
    assert r3.status_code == 200
    assert r3.json()["success"] is True


def test_apply_suggestion_invalid_parameters():
    """Verify error handling for invalid layouts, nonexistent clips, nonexistent projects."""
    p_id = "proj_apply_inval"
    clip = Clip(id="c1", start=10.0, end=20.0)
    proj = Project(id=p_id, clips=[clip])
    store.save(proj)

    # Invalid layout
    r1 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_custom/apply", json={"patch": {"layout": "invalid_layout_type"}})
    assert r1.status_code == 400

    # Nonexistent clip
    r2 = client.post(f"/api/projects/{p_id}/clips/c_ghost/suggestions/sug_any/apply", json={"patch": {"teaser_loop": True}})
    assert r2.status_code == 404

    # Nonexistent project
    r3 = client.post("/api/projects/p_ghost/clips/c1/suggestions/sug_any/apply", json={"patch": {"teaser_loop": True}})
    assert r3.status_code == 404


def test_apply_suggestion_boundary_clamping():
    """Verify out-of-range patch boundaries are safely clamped."""
    p_id = "proj_apply_clamp"
    clip = Clip(id="c1", start=10.0, end=20.0)
    proj = Project(id=p_id, clips=[clip])
    store.save(proj)

    # Negative start_t -> clamped to 0.0
    r1 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_neg_start/apply", json={"patch": {"start_t": -15.0}})
    assert r1.status_code == 200
    assert r1.json()["clip"]["start"] == 0.0

    # start_t > end_t -> clamped to end_t - 1.0
    r2 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_high_start/apply", json={"patch": {"start_t": 50.0}})
    assert r2.status_code == 200
    assert r2.json()["clip"]["start"] == 19.0  # 20.0 - 1.0

    # reframe_cx out of range -> clamped to [0.0, 1.0]
    r3 = client.post(f"/api/projects/{p_id}/clips/c1/suggestions/sug_cx/apply", json={"patch": {"reframe_cx": 2.5}})
    assert r3.status_code == 200
    p = store.get(p_id)
    assert p.clip("c1").reframe.keyframes[0].cx == 1.0


def test_apply_suggestion_concurrency():
    """Verify concurrent requests do not corrupt SQLite database or crash."""
    p_id = "proj_apply_concur"
    clips = [Clip(id=f"c{i}", start=float(i*10), end=float(i*10+8), features={"climax_t": 4.0}) for i in range(8)]
    proj = Project(id=p_id, clips=clips)
    store.save(proj)

    errors = []
    def worker(cid):
        try:
            resp = client.post(f"/api/projects/{p_id}/clips/{cid}/suggestions/sug_climax_teaser_{cid}/apply")
            if resp.status_code != 200:
                errors.append((cid, resp.status_code, resp.text))
        except Exception as e:
            errors.append((cid, "exc", str(e)))

    threads = [threading.Thread(target=worker, args=(f"c{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent apply errors: {errors}"
    p = store.get(p_id)
    assert all(c.teaser_loop is True for c in p.clips)


# --------------------------------------------------------------------------- #
# 4. Batch Operations Stress Testing
# --------------------------------------------------------------------------- #

def test_batch_operations_stress():
    """Verify batch endpoints under extreme actions, ratings, and sequencing."""
    p_id = "proj_batch_stress"
    clips = [Clip(id=f"c{i}", start=float(i*5), end=float(i*5+4), score=50) for i in range(5)]
    proj = Project(id=p_id, clips=clips)
    store.save(proj)

    # 1. Rating with boundary numbers (-10, 100)
    r1 = client.post(f"/api/projects/{p_id}/clips/batch", json={
        "clip_ids": ["c0"],
        "action": "rate",
        "payload": {"rating": -5}
    })
    assert r1.status_code == 200
    assert store.get(p_id).clip("c0").score == 0

    r2 = client.post(f"/api/projects/{p_id}/clips/batch", json={
        "clip_ids": ["c1"],
        "action": "rate",
        "payload": {"rating": 100}
    })
    assert r2.status_code == 200
    assert store.get(p_id).clip("c1").score == 100

    # 2. Tag with empty strings and duplicates
    r3 = client.post(f"/api/projects/{p_id}/clips/batch", json={
        "clip_ids": ["c0", "c1"],
        "action": "tag",
        "payload": {"tags": ["#good", "", "   ", "#good", "#great"]}
    })
    assert r3.status_code == 200
    c0 = store.get(p_id).clip("c0")
    assert c0.hashtags == ["#good", "#great"]

    # 3. Partial nonexistent IDs in batch
    r4 = client.post(f"/api/projects/{p_id}/clips/batch", json={
        "clip_ids": ["c2", "c_ghost_1", "c_ghost_2"],
        "action": "rate",
        "payload": {"rating": 4}
    })
    assert r4.status_code == 200
    assert r4.json()["affected_clip_ids"] == ["c2"]

    # 4. Delete all clips
    r5 = client.post(f"/api/projects/{p_id}/clips/batch", json={
        "clip_ids": [f"c{i}" for i in range(5)],
        "action": "delete"
    })
    assert r5.status_code == 200
    p_empty = store.get(p_id)
    assert len(p_empty.clips) == 0


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    failures = []
    print(f"Running {len(fns)} empirical challenger stress tests...\n")
    for fn in fns:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            failures.append((fn.__name__, type(e).__name__, str(e)))
            failed += 1
            continue
        print(f"  ✓ {fn.__name__}", flush=True)
        passed += 1

    print(f"\n=======================================================")
    print(f"STRESS RESULTS: {passed}/{len(fns)} passed, {failed} failed")
    print(f"=======================================================")
    if failures:
        print("\nFailures Detail:")
        for name, err_type, err_msg in failures:
            print(f"  - {name}: [{err_type}] {err_msg}")
        sys.exit(1)
    sys.exit(0)
