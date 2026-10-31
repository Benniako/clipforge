"""Automated test suite for Interactive Studio Backend APIs and Offline Services (Milestone M1).

Covers:
1. Audio Waveform & Climax Heatmap extraction service and GET /api/projects/{id}/clips/{id}/waveform.
2. Video Storyboard Thumbnail Scrub endpoint GET /api/projects/{id}/clips/{id}/storyboard.
3. Smart AI Suggestions & Co-Pilot service and GET /api/projects/{id}/clips/{id}/suggestions.
4. One-Click Suggestion Application POST /api/projects/{id}/clips/{id}/suggestions/{suggestion_id}/apply.
5. Batch Clip Operations POST /api/projects/{id}/clips/batch (rate, tag, sequence, delete, export).
6. Error handling, input validation, and edge cases.
7. Strict local-first offline execution verification (zero outbound network calls).
"""
from __future__ import annotations

import math
import os
import socket
import struct
import tempfile
import wave
from pathlib import Path

from starlette.testclient import TestClient

# Setup isolated test directory before loading app modules
_test_dir = tempfile.mkdtemp(prefix="clipforge-studio-test-")
os.environ["CLIPFORGE_DATA_DIR"] = _test_dir
os.environ["CLIPFORGE_DEVICE"] = "cpu"

# Enforce strict offline operation (R7): any outbound non-local network attempt fails
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


def _create_synthetic_wav(path: Path, duration_s: float = 20.0, sample_rate: int = 16000) -> None:
    """Create a real 16 kHz 16-bit mono PCM WAV with silence, dialogue, and a loud scream peak."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            # 0.0 - 1.0s: silence (~ -50 dBFS)
            if t < 1.0:
                val = int(math.sin(2 * math.pi * 200 * t) * 100)
            # 1.0 - 12.0s: normal speech dialogue (~ -16 dBFS)
            elif t < 12.0:
                val = int(math.sin(2 * math.pi * 400 * t) * 5000)
            # 12.0 - 13.5s: loud scream peak (> -4 dBFS, near full-scale)
            elif t < 13.5:
                val = int(math.sin(2 * math.pi * 850 * t) * 31000)
            # 13.5 - 20.0s: crowd roar cheering (~ -10 dBFS)
            else:
                val = int(math.sin(2 * math.pi * 320 * t) * 12000)
            samples.append(struct.pack("<h", max(-32768, min(32767, val))))

        wf.writeframes(b"".join(samples))


def _build_test_project(project_id: str = "proj_studio_test") -> Project:
    """Construct a full Project with clips, speech transcript, facecam, and detected cues."""
    settings = get_settings()
    proj_dir = settings.media_dir / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create real audio file
    wav_path = proj_dir / "audio16k.wav"
    _create_synthetic_wav(wav_path, duration_s=30.0)

    # 2. Setup transcript with leading dead air for clip 1
    # Clip 1 span: 10.0s to 25.0s. First spoken word starts at 10.9s (0.9s lead silence)
    words = [
        Word(t=10.9, d=0.4, text="Wait"),
        Word(t=11.4, d=0.5, text="did"),
        Word(t=12.0, d=0.4, text="you"),
        Word(t=12.5, d=0.6, text="see"),
        Word(t=13.2, d=0.5, text="that"),
        Word(t=14.0, d=0.8, text="triple"),
        Word(t=15.0, d=0.9, text="kill"),
        Word(t=16.0, d=1.2, text="unbelievable"),
    ]
    transcript = Transcript(words=words, language="en", provider="whisper")

    clip1 = Clip(
        id="c1",
        start=10.0,
        end=25.0,
        title="Epic Triple Kill",
        kind="speech",
        score=88,
        features={"climax_t": 8.2, "lead_silence": 0.9},
        hashtags=["#gaming"],
        captions=CaptionSet(words=[
            CaptionWord(t=0.9, d=0.4, text="Wait"),
            CaptionWord(t=1.4, d=0.5, text="did"),
            CaptionWord(t=2.0, d=0.4, text="you"),
        ]),
        reframe=Reframe(layout=LayoutType.fill),
        status=ClipStatus.ready,
        export_url=f"/media/{project_id}/clips/c1.mp4",
        thumb_url=f"/media/{project_id}/clips/c1_thumb.jpg",
    )

    clip2 = Clip(
        id="c2",
        start=1.0,
        end=9.5,
        title="Opening Round",
        kind="gameplay",
        score=72,
        features={"climax_t": 4.0},
        hashtags=["#valorant"],
        reframe=Reframe(layout=LayoutType.fill),
        status=ClipStatus.ready,
        export_url=f"/media/{project_id}/clips/c2.mp4",
        thumb_url=f"/media/{project_id}/clips/c2_thumb.jpg",
    )

    events = [
        DetectedEvent(t=18.2, label="excited shouting", source="audio", confidence=0.95),
        DetectedEvent(t=19.5, label="crowd cheering", source="audio", confidence=0.92),
        DetectedEvent(t=17.8, label="triple_kill", source="cue", confidence=0.98),
    ]

    project = Project(
        id=project_id,
        name="Interactive Studio Test Project",
        settings=ImportSettings(platform=Platform.tiktok, language="en"),
        transcript=transcript,
        clips=[clip1, clip2],
        facecam=Rect(x=0.05, y=0.05, w=0.25, h=0.25),
        events=events,
    )
    store.save(project)
    return project


# --------------------------------------------------------------------------- #
# Waveform and Climax Heatmap Tests
# --------------------------------------------------------------------------- #

def test_waveform_service_and_endpoint():
    """Verify waveform extraction with real WAV samples, decibels, and event markers."""
    proj = _build_test_project("proj_wf_test")
    clip = proj.clip("c1")
    assert clip is not None

    # Test direct service function
    wf_data = audio_waveform.extract_clip_waveform(proj, clip)
    assert wf_data["clip_id"] == "c1"
    assert wf_data["duration_s"] == 15.0
    assert 100 <= len(wf_data["samples"]) <= 200
    assert len(wf_data["samples"]) == len(wf_data["dbfs"])
    assert all(0.0 <= s <= 1.0 for s in wf_data["samples"])
    assert all(-60.0 <= d <= 0.0 for d in wf_data["dbfs"])
    assert 0.0 <= wf_data["climax_t"] <= 15.0

    # Verify event markers (scream, crowd_roar, kill_streak)
    events = wf_data["events"]
    assert len(events) >= 3

    scream_ev = next((e for e in events if e["type"] == "scream"), None)
    assert scream_ev is not None
    assert scream_ev["color"] == "#FF4500"

    roar_ev = next((e for e in events if e["type"] == "crowd_roar"), None)
    assert roar_ev is not None
    assert roar_ev["color"] == "#9D00FF"

    kill_ev = next((e for e in events if e["type"] == "kill_streak"), None)
    assert kill_ev is not None
    assert kill_ev["color"] == "#00F0FF"

    # Test API endpoint
    resp = client.get(f"/api/projects/{proj.id}/clips/{clip.id}/waveform")
    assert resp.status_code == 200
    res = resp.json()
    assert res["clip_id"] == "c1"
    assert len(res["samples"]) >= 100
    assert "events" in res


def test_waveform_fallback_without_audio_file():
    """Verify clean deterministic fallback synthesis when audio16k.wav is missing."""
    settings = get_settings()
    p_id = "proj_wf_fallback"
    project = Project(
        id=p_id,
        name="No Audio Project",
        clips=[Clip(id="c_synth", start=5.0, end=15.0, title="Synth Clip", features={"climax_t": 6.0})],
    )
    store.save(project)

    resp = client.get(f"/api/projects/{p_id}/clips/c_synth/waveform")
    assert resp.status_code == 200
    res = resp.json()
    assert res["clip_id"] == "c_synth"
    assert res["duration_s"] == 10.0
    assert len(res["samples"]) >= 100
    assert len(res["dbfs"]) == len(res["samples"])
    assert res["climax_t"] == 6.0


def test_waveform_not_found_handling():
    """Verify 404 responses for missing project or clip IDs."""
    resp = client.get("/api/projects/proj_nonexistent/clips/c1/waveform")
    assert resp.status_code == 404

    proj = _build_test_project("proj_wf_404")
    resp = client.get(f"/api/projects/{proj.id}/clips/c_invalid/waveform")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Video Storyboard Thumbnail Scrub Tests
# --------------------------------------------------------------------------- #

def test_storyboard_manifest_endpoint():
    """Verify storyboard manifest layout, dimensions, and scrub coordinates."""
    proj = _build_test_project("proj_sb_test")
    resp = client.get(f"/api/projects/{proj.id}/clips/c1/storyboard")
    assert resp.status_code == 200
    res = resp.json()

    assert res["clip_id"] == "c1"
    assert "sprite_url" in res
    assert res["cols"] == 5
    assert res["rows"] == 2
    assert res["tile_w"] == 160
    assert res["tile_h"] == 90
    assert res["total_frames"] == 10
    assert res["duration_s"] == 15.0
    assert res["interval_s"] > 0.0


def test_storyboard_not_found():
    """Verify 404 for invalid project or clip in storyboard endpoint."""
    resp = client.get("/api/projects/proj_unknown/clips/c1/storyboard")
    assert resp.status_code == 404

    proj = _build_test_project("proj_sb_404")
    resp = client.get(f"/api/projects/{proj.id}/clips/c_unknown/storyboard")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Smart AI Suggestions & Co-Pilot Engine Tests
# --------------------------------------------------------------------------- #

def test_suggestions_service_and_endpoint():
    """Verify AI suggestion cards for hook trimming, hook rating, virality, and facecam."""
    proj = _build_test_project("proj_sug_test")
    clip = proj.clip("c1")
    assert clip is not None

    # Test direct engine
    sugs = suggestions_copilot.generate_clip_suggestions(proj, clip)
    assert len(sugs) >= 3

    # 1. Hook trim card
    trim_card = next((s for s in sugs if s["type"] == "trim_hook"), None)
    assert trim_card is not None
    assert trim_card["id"] == "sug_trim_hook_c1"
    assert "Cut Dead Air Hook" in trim_card["title"]
    assert "retention" in trim_card["retention_impact"]
    assert "start_t" in trim_card["patch"]
    assert 10.8 <= trim_card["patch"]["start_t"] <= 10.9

    # 2. Hook rating card
    rate_card = next((s for s in sugs if s["type"] == "hook_rating"), None)
    assert rate_card is not None
    assert rate_card["id"] == "sug_hook_rating_c1"
    assert "Hook Strength" in rate_card["title"]
    assert rate_card["retention_impact"] in ("Strong Hook", "OK Hook", "Weak Hook")

    # 3. Split facecam card
    reframe_card = next((s for s in sugs if s["type"] == "reframe_layout"), None)
    assert reframe_card is not None
    assert reframe_card["id"] == "sug_split_reframe_c1"
    assert reframe_card["patch"] == {"layout": "split"}
    assert "engagement" in reframe_card["retention_impact"]

    # 4. Cold-open teaser loop card
    climax_card = next((s for s in sugs if s["type"] == "climax_teaser"), None)
    assert climax_card is not None
    assert climax_card["patch"] == {"teaser_loop": True}

    # Test HTTP endpoint
    resp = client.get(f"/api/projects/{proj.id}/clips/c1/suggestions")
    assert resp.status_code == 200
    res = resp.json()
    assert res["clip_id"] == "c1"
    assert len(res["suggestions"]) >= 3


# --------------------------------------------------------------------------- #
# One-Click Suggestion Application Tests
# --------------------------------------------------------------------------- #

def test_apply_suggestion_hook_trim():
    """Verify one-click apply of hook trim updates clip start timestamp atomically."""
    proj = _build_test_project("proj_apply_trim")

    resp = client.post(f"/api/projects/{proj.id}/clips/c1/suggestions/sug_trim_hook_c1/apply")
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert "start_t" in res["updated_fields"]

    updated_clip = store.get(proj.id).clip("c1")
    assert updated_clip.start == res["clip"]["start"]
    assert updated_clip.start > 10.0  # Successfully trimmed initial silence!
    assert updated_clip.status == ClipStatus.rendering


def test_apply_suggestion_split_reframe():
    """Verify one-click apply of split facecam reframe updates layout and facecam."""
    proj = _build_test_project("proj_apply_reframe")

    resp = client.post(f"/api/projects/{proj.id}/clips/c1/suggestions/sug_split_reframe_c1/apply")
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert "layout" in res["updated_fields"]

    updated_clip = store.get(proj.id).clip("c1")
    assert updated_clip.reframe.layout == LayoutType.split
    assert updated_clip.reframe.facecam is not None


def test_apply_suggestion_teaser_loop():
    """Verify one-click apply of cold-open climax teaser loop."""
    proj = _build_test_project("proj_apply_teaser")

    resp = client.post(f"/api/projects/{proj.id}/clips/c1/suggestions/sug_climax_teaser_c1/apply")
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert "teaser_loop" in res["updated_fields"]

    updated_clip = store.get(proj.id).clip("c1")
    assert updated_clip.teaser_loop is True


def test_apply_suggestion_invalid_id():
    """Verify 404 for nonexistent suggestion ID."""
    proj = _build_test_project("proj_apply_404")
    resp = client.post(f"/api/projects/{proj.id}/clips/c1/suggestions/sug_fake_unknown/apply")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Batch Operations & Montage Sequencing Tests
# --------------------------------------------------------------------------- #

def test_batch_rate_clips():
    """Verify batch rating action sets score and feedback across multiple clips."""
    proj = _build_test_project("proj_batch_rate")

    body = {
        "clip_ids": ["c1", "c2"],
        "action": "rate",
        "payload": {"rating": 5}
    }
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json=body)
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert res["action"] == "rate"
    assert set(res["affected_clip_ids"]) == {"c1", "c2"}

    p = store.get(proj.id)
    assert p.clip("c1").score == 100
    assert p.clip("c1").feedback == "up"
    assert p.clip("c2").score == 100
    assert p.clip("c2").feedback == "up"


def test_batch_tag_clips():
    """Verify batch tagging adds hashtags to specified clips."""
    proj = _build_test_project("proj_batch_tag")

    body = {
        "clip_ids": ["c1", "c2"],
        "action": "tag",
        "payload": {"tags": ["#trending", "#viral"]}
    }
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json=body)
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True

    p = store.get(proj.id)
    assert "#trending" in p.clip("c1").hashtags
    assert "#viral" in p.clip("c1").hashtags
    assert "#trending" in p.clip("c2").hashtags


def test_batch_sequence_and_delete_clips():
    """Verify batch reordering (montage sequencing) and batch deletion."""
    proj = _build_test_project("proj_batch_seq_del")
    assert [c.id for c in proj.clips] == ["c1", "c2"]

    # 1. Sequence: reverse order
    body_seq = {
        "clip_ids": ["c2", "c1"],
        "action": "sequence",
        "payload": {"order": ["c2", "c1"]}
    }
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json=body_seq)
    assert resp.status_code == 200

    p = store.get(proj.id)
    assert [c.id for c in p.clips] == ["c2", "c1"]

    # 2. Delete: remove c1
    body_del = {
        "clip_ids": ["c1"],
        "action": "delete"
    }
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json=body_del)
    assert resp.status_code == 200

    p_after = store.get(proj.id)
    assert len(p_after.clips) == 1
    assert p_after.clip("c1") is None
    assert p_after.clip("c2") is not None


def test_batch_validation_and_errors():
    """Verify validation for empty clip_ids and unsupported actions."""
    proj = _build_test_project("proj_batch_err")

    # Empty clip_ids
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={"clip_ids": [], "action": "rate"})
    assert resp.status_code == 400

    # Unsupported action
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={"clip_ids": ["c1"], "action": "fly_to_moon"})
    assert resp.status_code == 400

    # Project not found
    resp = client.post("/api/projects/proj_ghost/clips/batch", json={"clip_ids": ["c1"], "action": "rate"})
    assert resp.status_code == 404


def setup_offline_guard():
    """Enforce strict offline socket check outside pytest runner."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
            raise RuntimeError(f"Strict Offline Invariant Violated! Outgoing network call attempted to {host}")
        return real_connect(self, address)

    socket.socket.connect = guarded_connect


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_offline_guard()
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            failed += 1
            continue
        print(f"  ✓ {fn.__name__}", flush=True)
        passed += 1
    print(f"\n{passed}/{len(fns)} interactive studio unit tests passed"
          + (f", {failed} failed" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)

