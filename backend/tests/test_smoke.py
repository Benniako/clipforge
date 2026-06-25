import shutil
import subprocess

import pytest

from app import store
from app.config import get_settings
from app.models import ClipStatus, ImportSettings, Platform, Project, ProjectStatus
from app.pipeline import ingest
from app.pipeline.orchestrator import engine
from app.media import ffmpeg

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not shutil.which(get_settings().ffmpeg or "ffmpeg"),
    reason="ffmpeg required for pipeline smoke test",
)]


@pytest.mark.timeout(120)
def test_pipeline_produces_clips_from_synthetic_source(tmp_path):
    """End-to-end: generate a short synthetic video → run pipeline → verify clips.

    This is a *slow* integration test (creates a real video, runs the full
    pipeline) so it's marked ``slow`` and excluded from ``pytest -m 'not slow'``.
    """
    src = tmp_path / "source.mp4"
    ff = get_settings().ffmpeg
    subprocess.run([
        ff, "-hide_banner", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=10",
        "-vf", "drawtext=text='Test':x=40:y=40:fontsize=48:fontcolor=white",
        "-c:v", "libx264", "-preset", "ultrafast", str(src),
    ], check=True, capture_output=True)

    project = Project(name="Smoke",
                      settings=ImportSettings(platform=Platform.tiktok,
                                              min_len=3, max_len=9, target_clips=3))
    store.save(project)
    project.source = ingest.attach_source_file(project, src, "source.mp4")
    store.save(project)

    engine._process(project.id)

    final = store.get(project.id)
    assert final.status == ProjectStatus.complete, f"pipeline failed: {final.status}"
    assert len(final.clips) >= 1, "expected at least one clip"
    assert any(c.export_url for c in final.clips if c.status == ClipStatus.rendered), \
        "expected at least one rendered clip"
