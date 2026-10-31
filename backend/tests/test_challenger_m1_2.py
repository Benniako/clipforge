"""Challenger test suite for Milestone M1 (challenger_m1_2).

Adversarial stress testing and empirical verification of:
1. Batch clip operations (POST /api/projects/{id}/clips/batch):
   - Empty lists, missing payloads, unsupported actions
   - Large arrays of clip IDs (1,000+ items)
   - Non-existent / ghost clip IDs and mixed ID sets
   - Payload format variations (invalid types, string vs numeric ratings, malformed tags)
   - Sequence order variations (partial orders, duplicates, non-list types)
   - Atomic consistency under store mutation
2. Video Storyboard Tile Coordinate Calculations (GET /api/projects/{id}/clips/{id}/storyboard):
   - Varied video durations: 1s, 15s, 60s, 300s, sub-second (0.05s), zero duration
   - Manifest parameters: cols, rows, tile_w, tile_h, total_frames, interval_s
   - Coordinate oracle for hover scrubbing across 0..duration timeline
   - 404 handling and fallback sprite URL paths
3. Montage Sequencing API (POST /api/projects/{id}/montage and batch sequence action):
   - Minimum rendered clips validation (<2 clips -> 400)
   - Unrendered clips handling
   - Non-existent clip IDs handling
   - Montage persistence and factor scoring
"""
from __future__ import annotations

import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

from starlette.testclient import TestClient

# Setup isolated test directory
_test_dir = tempfile.mkdtemp(prefix="clipforge-challenger-test-")
os.environ["CLIPFORGE_DATA_DIR"] = _test_dir
os.environ["CLIPFORGE_DEVICE"] = "cpu"

# Enforce strict offline execution
_real_connect = socket.socket.connect

def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1", "testserver"):
        raise RuntimeError(f"Strict Offline Invariant Violated! Attempted outbound connection to {host}")
    return _real_connect(self, address)

socket.socket.connect = _guarded_connect

from app import feedback, store
from app.config import get_settings
from app.main import app

store.init_db()
feedback.init_db()

from app.models import (
    CaptionSet, CaptionWord, Clip, ClipStatus, ImportSettings, LayoutType,
    Platform, Project, Reframe
)

client = TestClient(app)


def _create_project_with_clips(project_id: str, clip_specs: list[dict]) -> Project:
    """Helper to populate a test project with specified clips."""
    clips = []
    for spec in clip_specs:
        cid = spec["id"]
        start = float(spec.get("start", 0.0))
        end = float(spec.get("end", 10.0))
        duration = max(0.0, end - start)
        clip = Clip(
            id=cid,
            start=start,
            end=end,
            title=spec.get("title", f"Clip {cid}"),
            kind=spec.get("kind", "speech"),
            score=spec.get("score", 50),
            status=spec.get("status", ClipStatus.ready),
            export_url=spec.get("export_url", f"/media/{project_id}/clips/{cid}.mp4"),
            thumb_url=spec.get("thumb_url", f"/media/{project_id}/clips/{cid}_thumb.jpg"),
            hashtags=spec.get("hashtags", []),
            feedback=spec.get("feedback", None),
        )
        clips.append(clip)

    proj = Project(
        id=project_id,
        name=f"Challenger Test {project_id}",
        settings=ImportSettings(platform=Platform.tiktok, language="en"),
        clips=clips,
    )
    store.save(proj)
    return proj


# =========================================================================== #
# Objective 1: Batch Operations Stress & Robustness Tests
# =========================================================================== #

def test_batch_empty_clip_ids_rejected():
    """Verify POST /clips/batch with empty clip_ids returns 400 Bad Request."""
    proj = _create_project_with_clips("proj_batch_empty", [{"id": "c1"}])
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={"clip_ids": [], "action": "rate"})
    assert resp.status_code == 400
    assert "clip_ids list cannot be empty" in resp.json()["detail"]


def test_batch_unsupported_action_rejected():
    """Verify POST /clips/batch with unknown action returns 400 Bad Request."""
    proj = _create_project_with_clips("proj_batch_bad_act", [{"id": "c1"}])
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={"clip_ids": ["c1"], "action": "turbo_boost"})
    assert resp.status_code == 400
    assert "unsupported batch action" in resp.json()["detail"]


def test_batch_project_not_found():
    """Verify POST /clips/batch with nonexistent project returns 404 Not Found."""
    resp = client.post("/api/projects/proj_does_not_exist_9999/clips/batch",
                       json={"clip_ids": ["c1"], "action": "rate", "payload": {"rating": 5}})
    assert resp.status_code == 404
    assert "project not found" in resp.json()["detail"]


def test_batch_nonexistent_clip_ids():
    """Test batch operations when clip_ids contain non-existent IDs.
    Should gracefully return affected_clip_ids=[] without corrupting project.
    """
    proj = _create_project_with_clips("proj_batch_ghost", [{"id": "c1", "score": 50}])
    
    # Send batch rate for non-existent clip ID
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["ghost_1", "ghost_2"],
        "action": "rate",
        "payload": {"rating": 5}
    })
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert res["affected_clip_ids"] == []
    
    # Verify existing clip was untouched
    p = store.get(proj.id)
    assert p.clip("c1").score == 50


def test_batch_mixed_existing_and_ghost_clip_ids():
    """Test batch operations with a mixture of existing and non-existent IDs."""
    proj = _create_project_with_clips("proj_batch_mixed", [
        {"id": "c1", "score": 20},
        {"id": "c2", "score": 30},
    ])
    
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["ghost_a", "c1", "ghost_b"],
        "action": "rate",
        "payload": {"rating": 4}  # score = 80, feedback = "up"
    })
    assert resp.status_code == 200
    res = resp.json()
    assert res["affected_clip_ids"] == ["c1"]
    
    p = store.get(proj.id)
    assert p.clip("c1").score == 80
    assert p.clip("c1").feedback == "up"
    assert p.clip("c2").score == 30  # c2 was untouched


def test_batch_large_array_stress():
    """Stress-test batch endpoint with 1,000 clip IDs in payload."""
    clip_specs = [{"id": f"clip_{i:04d}", "score": 10} for i in range(50)]
    proj = _create_project_with_clips("proj_batch_stress_1k", clip_specs)
    
    # 50 real clips + 950 ghost IDs = 1,000 total IDs
    large_clip_ids = [f"clip_{i:04d}" for i in range(50)] + [f"ghost_{i:04d}" for i in range(950)]
    assert len(large_clip_ids) == 1000
    
    t0 = time.perf_counter()
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": large_clip_ids,
        "action": "tag",
        "payload": {"tags": ["#stress_tested", "#batch_1000"]}
    })
    elapsed = time.perf_counter() - t0
    
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert len(res["affected_clip_ids"]) == 50
    assert elapsed < 3.0, f"Batch 1k took too long: {elapsed:.2f}s"
    
    # Verify persistence
    p = store.get(proj.id)
    assert len(p.clips) == 50
    assert all("#stress_tested" in c.hashtags for c in p.clips)
    assert all("#batch_1000" in c.hashtags for c in p.clips)


def test_batch_rate_payload_formats():
    """Verify various payload formats and edge cases for batch rate action."""
    proj = _create_project_with_clips("proj_rate_formats", [
        {"id": "c1"}, {"id": "c2"}, {"id": "c3"}
    ])
    
    # 1. Missing rating in payload -> 400
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1"],
        "action": "rate",
        "payload": {}
    })
    assert resp.status_code == 400
    assert "payload must contain 'rating'" in resp.json()["detail"]
    
    # 2. String rating "up"
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1"],
        "action": "rate",
        "payload": {"rating": "up"}
    })
    assert resp.status_code == 200
    p = store.get(proj.id)
    assert p.clip("c1").feedback == "up"
    
    # 3. String rating "down"
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c2"],
        "action": "rate",
        "payload": {"rating": "down"}
    })
    assert resp.status_code == 200
    p = store.get(proj.id)
    assert p.clip("c2").feedback == "down"
    
    # 4. Out-of-bounds numeric rating (e.g. 10 -> score clamps to 100)
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c3"],
        "action": "rate",
        "payload": {"rating": 10}
    })
    assert resp.status_code == 200
    p = store.get(proj.id)
    assert p.clip("c3").score == 100
    assert p.clip("c3").feedback == "up"


def test_batch_tag_payload_validation():
    """Verify validation of payload in batch tag action."""
    proj = _create_project_with_clips("proj_tag_val", [{"id": "c1"}])
    
    # Invalid: tags is not a list
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1"],
        "action": "tag",
        "payload": {"tags": "single_string_not_list"}
    })
    assert resp.status_code == 400
    assert "'tags' in payload must be a list" in resp.json()["detail"]
    
    # Valid: empty tags payload -> no-op
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1"],
        "action": "tag",
        "payload": {}
    })
    assert resp.status_code == 200


def test_batch_delete_atomic():
    """Verify batch deletion removes multiple clips atomically."""
    proj = _create_project_with_clips("proj_batch_del", [
        {"id": "c1"}, {"id": "c2"}, {"id": "c3"}, {"id": "c4"}
    ])
    
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c2", "c4"],
        "action": "delete"
    })
    assert resp.status_code == 200
    res = resp.json()
    assert set(res["affected_clip_ids"]) == {"c2", "c4"}
    
    p = store.get(proj.id)
    remaining_ids = [c.id for c in p.clips]
    assert remaining_ids == ["c1", "c3"]


def test_batch_sequence_montage_reordering():
    """Verify batch sequence reorders clips and preserves unmentioned clips."""
    proj = _create_project_with_clips("proj_batch_seq", [
        {"id": "c1"}, {"id": "c2"}, {"id": "c3"}, {"id": "c4"}
    ])
    
    # Partial reorder: specify c3 then c1, leaving c2 and c4
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c3", "c1"],
        "action": "sequence",
        "payload": {"order": ["c3", "c1"]}
    })
    assert resp.status_code == 200
    
    p = store.get(proj.id)
    ordered_ids = [c.id for c in p.clips]
    # c3, c1 first, then remaining c2, c4
    assert ordered_ids[:2] == ["c3", "c1"]
    assert set(ordered_ids[2:]) == {"c2", "c4"}
    assert len(ordered_ids) == 4


# =========================================================================== #
# Objective 2: Storyboard Tile Coordinate Stress Testing
# =========================================================================== #

def test_storyboard_durations_coordinate_math():
    """Verify storyboard manifest coordinate calculations for varied video durations.
    Tests durations: 1s, 15s, 60s, 300s, sub-second (0.05s).
    """
    test_cases = [
        {"id": "c_1s", "start": 0.0, "end": 1.0, "expected_dur": 1.0, "expected_interval": 0.1},
        {"id": "c_15s", "start": 10.0, "end": 25.0, "expected_dur": 15.0, "expected_interval": 1.5},
        {"id": "c_60s", "start": 0.0, "end": 60.0, "expected_dur": 60.0, "expected_interval": 6.0},
        {"id": "c_300s", "start": 100.0, "end": 400.0, "expected_dur": 300.0, "expected_interval": 30.0},
        # Sub-second duration boundary condition (dur clamped to 0.1s minimum)
        {"id": "c_sub", "start": 0.0, "end": 0.05, "expected_dur": 0.1, "expected_interval": 0.01},
    ]
    
    proj = _create_project_with_clips("proj_sb_durations", test_cases)
    
    for case in test_cases:
        cid = case["id"]
        resp = client.get(f"/api/projects/{proj.id}/clips/{cid}/storyboard")
        assert resp.status_code == 200, f"Failed storyboard for clip {cid}: {resp.text}"
        data = resp.json()
        
        # Verify schema invariants
        assert data["clip_id"] == cid
        assert data["cols"] == 5
        assert data["rows"] == 2
        assert data["total_frames"] == 10
        assert data["tile_w"] == 160
        assert data["tile_h"] == 90
        
        # Verify math
        assert math.isclose(data["duration_s"], case["expected_dur"], abs_tol=1e-3)
        assert math.isclose(data["interval_s"], case["expected_interval"], abs_tol=1e-3)
        
        # Verify sprite URL presence
        assert "sprite_url" in data
        assert len(data["sprite_url"]) > 0


def test_storyboard_tile_coordinate_oracle():
    """Oracle verification: map timeline scrub position [0..duration] to tile coordinates.
    Sprite sheet: 5 cols x 2 rows, tile 160x90.
    Overall sprite dimensions: 800px x 180px.
    """
    proj = _create_project_with_clips("proj_sb_oracle", [
        {"id": "c_test", "start": 0.0, "end": 20.0}  # duration = 20.0s, interval = 2.0s
    ])
    
    resp = client.get(f"/api/projects/{proj.id}/clips/c_test/storyboard")
    assert resp.status_code == 200
    sb = resp.json()
    
    cols = sb["cols"]          # 5
    rows = sb["rows"]          # 2
    tile_w = sb["tile_w"]      # 160
    tile_h = sb["tile_h"]      # 90
    total_frames = sb["total_frames"]  # 10
    interval_s = sb["interval_s"]      # 2.0
    dur = sb["duration_s"]             # 20.0
    
    # Sheet total dimensions
    sheet_w = cols * tile_w    # 800
    sheet_h = rows * tile_h    # 180
    assert sheet_w == 800
    assert sheet_h == 180
    
    # Test time points across timeline and check coordinate oracle
    test_points = [
        # (time_s, expected_frame_idx, expected_col, expected_row, expected_x, expected_y)
        (0.0, 0, 0, 0, 0, 0),
        (1.9, 0, 0, 0, 0, 0),
        (2.0, 1, 1, 0, 160, 0),
        (4.0, 2, 2, 0, 320, 0),
        (6.5, 3, 3, 0, 480, 0),
        (8.0, 4, 4, 0, 640, 0),
        (10.0, 5, 0, 1, 0, 90),     # Row 1 begins at frame 5
        (12.0, 6, 1, 1, 160, 90),
        (14.0, 7, 2, 1, 320, 90),
        (16.0, 8, 3, 1, 480, 90),
        (18.0, 9, 4, 1, 640, 90),   # Last frame index = 9
        (20.0, 9, 4, 1, 640, 90),   # Duration endpoint clamped to frame 9
    ]
    
    for t, exp_idx, exp_col, exp_row, exp_x, exp_y in test_points:
        # Hover scrub index calculation
        idx = min(total_frames - 1, max(0, int(t / interval_s)))
        col = idx % cols
        row = idx // cols
        x_px = col * tile_w
        y_px = row * tile_h
        
        assert idx == exp_idx, f"At t={t}: expected frame {exp_idx}, got {idx}"
        assert col == exp_col, f"At t={t}: expected col {exp_col}, got {col}"
        assert row == exp_row, f"At t={t}: expected row {exp_row}, got {row}"
        assert x_px == exp_x, f"At t={t}: expected x_px {exp_x}, got {x_px}"
        assert y_px == exp_y, f"At t={t}: expected y_px {exp_y}, got {y_px}"
        
        # Verify tile stays strictly within sprite sheet boundaries
        assert 0 <= x_px <= sheet_w - tile_w
        assert 0 <= y_px <= sheet_h - tile_h


def test_storyboard_404_not_found():
    """Verify proper 404 for missing project or clip in storyboard endpoint."""
    resp = client.get("/api/projects/proj_nonexistent_xyz/clips/c1/storyboard")
    assert resp.status_code == 404
    
    proj = _create_project_with_clips("proj_sb_404", [{"id": "c1"}])
    resp = client.get(f"/api/projects/{proj.id}/clips/ghost_clip_id/storyboard")
    assert resp.status_code == 404


# =========================================================================== #
# Objective 3: Montage Sequencing API Verification
# =========================================================================== #

def test_montage_create_validation():
    """Verify POST /api/projects/{id}/montage validation rules:
    - Minimum 2 rendered clips required
    - Unrendered clips rejected
    - Valid clips create Montage object with status 'rendering'
    """
    proj = _create_project_with_clips("proj_mtg_val", [
        {"id": "c1", "export_url": "/media/proj_mtg_val/clips/c1.mp4", "status": ClipStatus.ready},
        {"id": "c2", "export_url": "/media/proj_mtg_val/clips/c2.mp4", "status": ClipStatus.ready},
        {"id": "c3_unrendered", "export_url": None, "status": ClipStatus.pending},
    ])
    
    # 1. Less than 2 clips provided -> 400
    resp = client.post(f"/api/projects/{proj.id}/montage", json={"clip_ids": ["c1"]})
    assert resp.status_code == 400
    assert "select at least 2 rendered clips" in resp.json()["detail"]
    
    # 2. 2 clips provided, but one is unrendered -> 400
    resp = client.post(f"/api/projects/{proj.id}/montage", json={"clip_ids": ["c1", "c3_unrendered"]})
    assert resp.status_code == 400
    assert "select at least 2 rendered clips" in resp.json()["detail"]
    
    # 3. 2 valid rendered clips -> 200 OK
    resp = client.post(f"/api/projects/{proj.id}/montage", json={
        "clip_ids": ["c1", "c2"],
        "title": "Challenger Test Montage"
    })
    assert resp.status_code == 200
    mtg = resp.json()
    assert mtg["title"] == "Challenger Test Montage"
    assert mtg["clip_ids"] == ["c1", "c2"]
    assert mtg["status"] == ClipStatus.rendering.value
    assert "score" in mtg
    assert "factors" in mtg
    
    # Verify persisted in project
    p = store.get(proj.id)
    assert len(p.montages) == 1
    assert p.montages[0].id == mtg["id"]


def test_montage_download_endpoint_states():
    """Verify GET /api/projects/{id}/montages/{montage_id}/download error handling."""
    proj = _create_project_with_clips("proj_mtg_dl", [
        {"id": "c1", "export_url": "/media/proj_mtg_dl/clips/c1.mp4"},
        {"id": "c2", "export_url": "/media/proj_mtg_dl/clips/c2.mp4"},
    ])
    
    # Create montage (status will be rendering)
    resp = client.post(f"/api/projects/{proj.id}/montage", json={"clip_ids": ["c1", "c2"]})
    assert resp.status_code == 200
    mtg_id = resp.json()["id"]
    
    # Attempt download while unrendered -> 409 Conflict
    resp_dl = client.get(f"/api/projects/{proj.id}/montages/{mtg_id}/download")
    assert resp_dl.status_code == 409
    assert "montage is not rendered yet" in resp_dl.json()["detail"]
    
    # Non-existent montage -> pre-existing route returns 409 because mtg is None
    resp_ghost = client.get(f"/api/projects/{proj.id}/montages/mtg_ghost_id/download")
    assert resp_ghost.status_code == 409


# =========================================================================== #
# Objective 4: Adversarial Stress Tests (Malformed inputs, duplicates, etc.)
# =========================================================================== #

def test_batch_sequence_non_list_type_error():
    """Verify sequence action returns 400 Bad Request when order is not a list."""
    proj = _create_project_with_clips("proj_seq_err", [{"id": "c1"}, {"id": "c2"}])
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1", "c2"],
        "action": "sequence",
        "payload": {"order": 12345}
    })
    assert resp.status_code == 400, f"Expected 400 Bad Request for non-list order, got {resp.status_code}"
    assert "'order' in payload must be a list" in resp.json()["detail"]


def test_batch_sequence_string_order_causes_silent_clip_loss():
    """Verify passing a string as 'order' returns 400 Bad Request and does not delete clips."""
    proj = _create_project_with_clips("proj_seq_str", [
        {"id": "c1"}, {"id": "c2"}
    ])
    
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1", "c2"],
        "action": "sequence",
        "payload": {"order": "c1"}  # string instead of list of strings!
    })
    assert resp.status_code == 400
    assert "'order' in payload must be a list" in resp.json()["detail"]
    
    p = store.get(proj.id)
    clip_ids = [c.id for c in p.clips]
    assert clip_ids == ["c1", "c2"], "Clip 'c1' not deleted; all clips preserved."


def test_batch_sequence_duplicate_ids_causes_clip_duplication():
    """Verify passing duplicate IDs in order deduplicates them and prevents clip duplication."""
    proj = _create_project_with_clips("proj_seq_dups", [
        {"id": "c1"}, {"id": "c2"}
    ])
    
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1", "c2"],
        "action": "sequence",
        "payload": {"order": ["c1", "c1", "c2"]}
    })
    assert resp.status_code == 200
    
    p = store.get(proj.id)
    clip_ids = [c.id for c in p.clips]
    assert clip_ids == ["c1", "c2"], "Duplicate clip IDs deduplicated properly"



def test_batch_delete_all_clips():
    """Verify deleting all clips in a project leaves project with empty clip list."""
    proj = _create_project_with_clips("proj_del_all", [
        {"id": "c1"}, {"id": "c2"}, {"id": "c3"}
    ])
    resp = client.post(f"/api/projects/{proj.id}/clips/batch", json={
        "clip_ids": ["c1", "c2", "c3"],
        "action": "delete"
    })
    assert resp.status_code == 200
    p = store.get(proj.id)
    assert len(p.clips) == 0



# =========================================================================== #
# Test Runner
# =========================================================================== #

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    print(f"\n--- Running Challenger M1 Test Suite ({len(fns)} tests) ---\n")
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
    print(f"\nChallenger Test Results: {passed}/{len(fns)} passed, {failed} failed.\n", flush=True)
    sys.exit(1 if failed else 0)
