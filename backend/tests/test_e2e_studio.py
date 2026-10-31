"""ClipForge Interactive Studio End-to-End Automated Test Suite.

Covers Tiers 1-4 for R1-R8 requirements:
- API endpoint contracts (clips, waveforms, suggestions, apply, batch, montage)
- Strict sovereign offline integrity verification using Python socket audit hook
- Boundary and error handling
- Cross-feature workflows and end-to-end streamer curation pipeline
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any, List

# Ensure backend directory is in sys.path when running as standalone script
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Reconfigure stdout for UTF-8 symbols on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Strict Offline Socket Audit Hook Setup
# ============================================================================

_EXTERNAL_CONNECTION_ATTEMPTS: List[Any] = []
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testserver"}


def _offline_socket_audit_hook(event: str, args: tuple) -> None:
    """Audit hook that traps and disallows any outbound network connections."""
    if event == "socket.connect":
        sock, address = args
        host = address[0] if isinstance(address, tuple) and len(address) > 0 else str(address)
        if str(host) not in _LOOPBACK_HOSTS:
            _EXTERNAL_CONNECTION_ATTEMPTS.append(address)
            raise RuntimeError(
                f"Offline integrity violation: attempted outbound connection to {address}"
            )


# Register socket audit hook once
try:
    sys.addaudithook(_offline_socket_audit_hook)
except Exception:
    pass

# Import backend application and models
from starlette.testclient import TestClient
from app.main import app
from app import store, feedback
from app.models import (
    Project, Clip, ImportSettings, Platform, CaptionWord, CaptionSet,
    Rect, Reframe, LayoutType, Transcript, SourceMedia
)


def _create_test_client() -> TestClient:
    """Create a test client with initialized database."""
    store.init_db()
    feedback.init_db()
    return TestClient(app, raise_server_exceptions=False)


def _seed_test_project_and_clips() -> tuple[Project, Clip, Clip]:
    """Helper to seed a test project with realistic gaming highlight data and rendered URLs."""
    store.init_db()
    p = Project(
        name="Apex Legends Ranked Highlights",
        source=SourceMedia(filename="test.mp4", path="media/test_source.mp4", duration=120.0),
        transcript=Transcript(provider="whisper", language="en", words=[]),
        settings=ImportSettings(platform=Platform.tiktok, game_profile="apex_legends"),
    )
    c1 = Clip(
        id="c_clutch_ace_1",
        start=10.0,
        end=34.5,
        title="1v3 Clutch Climax Ace",
        score=94,
        kind="gameplay",
        reframe=Reframe(layout=LayoutType.split, facecam=Rect(x=0.05, y=0.05, w=0.3, h=0.3)),
        export_url="/media/test1.mp4",
        features={"climax_t": 22.4, "kill_count": 3, "lead_silence": 0.8},
    )
    c1.captions = CaptionSet(
        style_id="bold-pop",
        words=[
            CaptionWord(t=10.5, d=0.8, text="ONE"),
            CaptionWord(t=11.4, d=0.7, text="DOWN"),
            CaptionWord(t=21.8, d=1.2, text="CLUTCH"),
            CaptionWord(t=23.2, d=1.0, text="VICTORY"),
        ],
    )

    c2 = Clip(
        id="c_sniper_headshot_2",
        start=40.0,
        end=55.0,
        title="Kraber 360 Headshot",
        score=88,
        kind="gameplay",
        export_url="/media/test2.mp4",
        features={"climax_t": 48.0},
    )

    p.clips = [c1, c2]
    store.save(p)
    return p, c1, c2


# ============================================================================
# Tier 1: Feature Coverage (Happy Path Tests)
# ============================================================================

def test_t1_strict_offline_integrity_zero_external_network():
    """T1-OFF-01: Proves socket audit hook intercepts external calls and allows loopback."""
    # Loopback connection test (should not raise offline violation)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 8000))
    except (ConnectionRefusedError, OSError):
        pass
    finally:
        s.close()

    assert len(_EXTERNAL_CONNECTION_ATTEMPTS) == 0, (
        f"Unexpected external connections detected: {_EXTERNAL_CONNECTION_ATTEMPTS}"
    )


def test_t1_health_and_capabilities_endpoints():
    """T1-API-01: Verifies offline health check and capability probe."""
    client = _create_test_client()
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json().get("ok") is True

    res_caps = client.get("/api/capabilities")
    assert res_caps.status_code == 200
    body = res_caps.json()
    assert "flat" in body and "detail" in body


def test_t1_project_and_clip_metadata_retrieval():
    """T1-API-02: Verifies project and clip metadata retrieval via GET /api/projects/{id}."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.get(f"/api/projects/{p.id}")
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == p.id
    assert len(body["clips"]) == 2
    clip_data = next(c for c in body["clips"] if c["id"] == c1.id)
    assert clip_data["title"] == "1v3 Clutch Climax Ace"
    assert clip_data["score"] == 94
    assert clip_data["reframe"]["facecam"]["x"] == 0.05


def test_t1_clip_publish_content_endpoint():
    """T1-API-03: Verifies social copy generation for clips."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.get(f"/api/projects/{p.id}/clips/{c1.id}/publish-content")
    assert res.status_code == 200
    body = res.json()
    assert "titles" in body
    assert "hashtags" in body
    assert isinstance(body["titles"], list)


def test_t1_montage_creation_endpoint():
    """T1-API-04: Verifies montage sequencing endpoint with 2 rendered clips."""
    client = _create_test_client()
    p, c1, c2 = _seed_test_project_and_clips()

    payload = {"clip_ids": [c1.id, c2.id], "title": "Apex Ranked Montage"}
    res = client.post(f"/api/projects/{p.id}/montage", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "Apex Ranked Montage"
    assert c1.id in body["clip_ids"]
    assert c2.id in body["clip_ids"]


def test_t1_clip_feedback_endpoint():
    """T1-API-05: Verifies streamer rating feedback recording."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.post(
        f"/api/projects/{p.id}/clips/{c1.id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["id"] == c1.id


def test_t1_waveform_endpoint_contract():
    """T1-WAVE-01: Verifies waveform & climax heatmap endpoint (PROJECT.md § 1)."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.get(f"/api/projects/{p.id}/clips/{c1.id}/waveform")
    assert res.status_code == 200
    data = res.json()
    assert data["clip_id"] == c1.id
    assert "samples" in data and len(data["samples"]) >= 100
    assert "dbfs" in data and len(data["dbfs"]) >= 100
    assert "events" in data
    assert data["climax_t"] > 0
    # Validate event colors match PROJECT.md
    for ev in data.get("events", []):
        if ev["type"] == "scream":
            assert ev["color"] == "#FF4500"
        elif ev["type"] == "crowd_roar":
            assert ev["color"] == "#9D00FF"
        elif ev["type"] == "kill_streak":
            assert ev["color"] == "#00F0FF"


def test_t1_storyboard_thumbnail_scrub_contract():
    """T1-SCRUB-01: Verifies storyboard thumbnail scrub manifest endpoint."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.get(f"/api/projects/{p.id}/clips/{c1.id}/storyboard")
    assert res.status_code == 200
    data = res.json()
    assert data["clip_id"] == c1.id
    assert "sprite_url" in data
    assert data["cols"] >= 1
    assert data["rows"] >= 1
    assert data["tile_w"] > 0
    assert data["tile_h"] > 0


def test_t1_ai_suggestions_copilot_contract():
    """T1-SUGG-01: Verifies AI suggestions endpoint (PROJECT.md § 2)."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    res = client.get(f"/api/projects/{p.id}/clips/{c1.id}/suggestions")
    assert res.status_code == 200
    data = res.json()
    assert data["clip_id"] == c1.id
    assert "suggestions" in data
    suggestions = data["suggestions"]
    assert len(suggestions) > 0
    for s in suggestions:
        assert "id" in s
        assert "type" in s
        assert "title" in s
        assert "patch" in s


def test_t1_suggestion_one_click_apply_contract():
    """T1-SUGG-02: Verifies applying an AI suggestion patches clip atomically."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    # Get suggestions
    res_sug = client.get(f"/api/projects/{p.id}/clips/{c1.id}/suggestions")
    assert res_sug.status_code == 200
    suggestions = res_sug.json()["suggestions"]
    assert len(suggestions) > 0
    first_sug = suggestions[0]

    # Apply suggestion
    res_apply = client.post(
        f"/api/projects/{p.id}/clips/{c1.id}/suggestions/{first_sug['id']}/apply",
        json={"patch": first_sug.get("patch", {})},
    )
    assert res_apply.status_code == 200
    apply_data = res_apply.json()
    assert apply_data.get("success") is True
    assert "clip" in apply_data


def test_t1_batch_operations_endpoint_contract():
    """T1-BAT-01: Verifies batch clip operations (PROJECT.md § 4)."""
    client = _create_test_client()
    p, c1, c2 = _seed_test_project_and_clips()

    payload = {
        "clip_ids": [c1.id, c2.id],
        "action": "rate",
        "payload": {"rating": 5},
    }
    res = client.post(f"/api/projects/{p.id}/clips/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data.get("success") is True
    assert len(data.get("affected_clip_ids", [])) == 2


# ============================================================================
# Tier 2: Boundary & Corner Cases (Limits, 0, Negative, Overflow)
# ============================================================================

def test_t2_nonexistent_project_returns_404():
    """T2-API-01: Requesting non-existent project returns 404 Not Found."""
    client = _create_test_client()
    res = client.get("/api/projects/proj_does_not_exist_999")
    assert res.status_code == 404
    assert "detail" in res.json()


def test_t2_nonexistent_clip_returns_404():
    """T2-API-02: Requesting non-existent clip returns 404 Not Found."""
    client = _create_test_client()
    p, _, _ = _seed_test_project_and_clips()
    res = client.get(f"/api/projects/{p.id}/clips/c_nonexistent_888/publish-content")
    assert res.status_code == 404


def test_t2_invalid_feedback_rating_value_rejected():
    """T2-API-03: Invalid rating body payload returns 422 Unprocessable Entity."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()
    res = client.post(
        f"/api/projects/{p.id}/clips/{c1.id}/feedback",
        json={"wrong_field": "missing_rating"},
    )
    assert res.status_code == 422


def test_t2_empty_montage_clip_ids_rejected():
    """T2-API-04: Creating montage with empty clip IDs list returns 400 or 422."""
    client = _create_test_client()
    p, _, _ = _seed_test_project_and_clips()
    res = client.post(f"/api/projects/{p.id}/montage", json={"clip_ids": []})
    assert res.status_code in (400, 422)


def test_t2_empty_batch_clip_ids_rejected():
    """T2-BAT-01: Batch operations with empty clip list returns 400."""
    client = _create_test_client()
    p, _, _ = _seed_test_project_and_clips()
    res = client.post(
        f"/api/projects/{p.id}/clips/batch",
        json={"clip_ids": [], "action": "rate", "payload": {"rating": 5}},
    )
    assert res.status_code == 400


def test_t2_external_network_socket_connect_trapped():
    """T2-OFF-01: Proves socket audit hook successfully traps external IP connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    trapped = False
    try:
        s.connect(("8.8.8.8", 53))
    except RuntimeError as e:
        if "Offline integrity violation" in str(e):
            trapped = True
    except Exception:
        pass
    finally:
        s.close()

    assert trapped or len(_EXTERNAL_CONNECTION_ATTEMPTS) > 0, (
        "Audit hook must trap external connection attempt"
    )


# ============================================================================
# Tier 3: Cross-Feature Combinations
# ============================================================================

def test_t3_clip_edit_and_project_persistence():
    """T3-COMB-01: Editing clip parameters (trim, layout, facecam) persists atomically."""
    client = _create_test_client()
    p, c1, _ = _seed_test_project_and_clips()

    edit_payload = {
        "title": "New Epic Ace Title",
        "description": "Clutch play in ranked.",
        "layout": "framed",
        "reframe_cx": 0.42,
    }
    res = client.patch(f"/api/projects/{p.id}/clips/{c1.id}", json=edit_payload)
    assert res.status_code == 200

    reloaded = store.get(p.id)
    assert reloaded is not None
    updated_clip = next(c for c in reloaded.clips if c.id == c1.id)
    assert updated_clip.title == "New Epic Ace Title"
    assert updated_clip.description == "Clutch play in ranked."
    assert updated_clip.reframe.layout == LayoutType.framed


def test_t3_batch_tag_and_retrieval():
    """T3-COMB-02: Batch tagging multiple clips updates metadata across all targets."""
    client = _create_test_client()
    p, c1, c2 = _seed_test_project_and_clips()

    payload = {
        "clip_ids": [c1.id, c2.id],
        "action": "tag",
        "payload": {"tags": ["#ViralMoment", "#ApexLegends"]},
    }
    res = client.post(f"/api/projects/{p.id}/clips/batch", json=payload)
    assert res.status_code == 200

    reloaded = store.get(p.id)
    assert reloaded is not None
    for c in reloaded.clips:
        assert "#ViralMoment" in c.hashtags


# ============================================================================
# Tier 4: Real-World Streamer Workload
# ============================================================================

def test_t4_full_streamer_backend_workload():
    """T4-WORKLOAD-01: Full streamer workflow: Seed -> Suggestions -> Apply -> Batch -> Montage."""
    client = _create_test_client()
    store.init_db()

    # 1. Project Creation
    p = Project(
        name="Stream Review 2026-10-30",
        settings=ImportSettings(platform=Platform.tiktok, game_profile="valorant"),
    )
    c1 = Clip(
        id="c_val_1",
        start=5.0,
        end=25.0,
        title="Valorant Ace #1",
        score=96,
        export_url="/media/val1.mp4",
        features={"lead_silence": 0.9, "climax_t": 18.0},
    )
    c2 = Clip(
        id="c_val_2",
        start=30.0,
        end=50.0,
        title="Valorant Clutch #2",
        score=88,
        export_url="/media/val2.mp4",
        features={"climax_t": 44.0},
    )
    p.clips = [c1, c2]
    store.save(p)

    # 2. Streamer loads project
    res_proj = client.get(f"/api/projects/{p.id}")
    assert res_proj.status_code == 200
    assert len(res_proj.json()["clips"]) == 2

    # 3. Streamer gets AI suggestions for Clip 1
    res_sug = client.get(f"/api/projects/{p.id}/clips/{c1.id}/suggestions")
    assert res_sug.status_code == 200
    suggestions = res_sug.json()["suggestions"]
    assert len(suggestions) > 0

    # 4. Streamer applies first recommendation
    sug = suggestions[0]
    res_apply = client.post(
        f"/api/projects/{p.id}/clips/{c1.id}/suggestions/{sug['id']}/apply",
        json={"patch": sug.get("patch", {})},
    )
    assert res_apply.status_code == 200

    # 5. Streamer batch rates both clips as 5 Stars
    res_batch = client.post(
        f"/api/projects/{p.id}/clips/batch",
        json={"clip_ids": [c1.id, c2.id], "action": "rate", "payload": {"rating": 5}},
    )
    assert res_batch.status_code == 200

    # 6. Streamer builds highlight montage with both clips
    res_montage = client.post(
        f"/api/projects/{p.id}/montage",
        json={"clip_ids": [c1.id, c2.id], "title": "Session Highlight Reel"},
    )
    assert res_montage.status_code == 200
    montage_body = res_montage.json()
    assert len(montage_body["clip_ids"]) == 2

    # 7. Streamer requests publish content
    res_pub = client.get(f"/api/projects/{p.id}/clips/{c1.id}/publish-content")
    assert res_pub.status_code == 200
    assert "hashtags" in res_pub.json()


# ============================================================================
# Standalone Test Runner Execution
# ============================================================================

if __name__ == "__main__":
    test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0

    print(f"\n=======================================================")
    print(f" ClipForge E2E Studio Automated Backend Test Suite")
    print(f"=======================================================\n")

    for fn in test_fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}", flush=True)
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            failed += 1

    print(f"\n-------------------------------------------------------")
    print(f"Results: {passed}/{len(test_fns)} tests passed"
          + (f", {failed} failed" if failed else " (100% SUCCESS)"), flush=True)
    print(f"-------------------------------------------------------\n")

    sys.exit(1 if failed else 0)
