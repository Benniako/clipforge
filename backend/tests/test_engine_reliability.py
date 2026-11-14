"""Tests for engine reliability: emotion2vec from the local cache, loading the
Ollama model before a time budget starts, background prewarming, and the
runtime problems shown in the diagnostics panel.

No model, no Ollama server and no FunASR are needed: they are replaced by fakes.

    cd backend && python -m pytest tests/test_engine_reliability.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import pytest  # noqa: E402

from app import engine_status  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.providers import emotion, llm, ollama_client, vlm  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    for key in list(engine_status.problems()):
        engine_status.clear(key)
    ollama_client._LOADED.clear()
    ollama_client._PREWARMING.clear()
    yield
    emotion._model = None


# --- the problem registry ----------------------------------------------------------

def test_problems_are_reported_shortened_and_cleared():
    engine_status.report("ollama", "Could not load x:\n  timed out " + "y" * 400)
    text = engine_status.problems()["ollama"]
    assert "\n" not in text and len(text) == engine_status.MAX_LEN and text.endswith("…")
    engine_status.clear("ollama")
    assert engine_status.problems() == {}


def test_capability_detail_carries_the_problem():
    engine_status.report("emotion", "The model could not be loaded: 403")
    items = {i["key"]: i for c in get_settings().capability_detail()["categories"]
             for i in c["items"]}
    assert items["emotion"]["problem"] == "The model could not be loaded: 403"
    assert items["ffmpeg"]["problem"] is None


# --- emotion2vec ----------------------------------------------------------------------

def _cached_model(root):
    folder = root / "iic" / "emotion2vec_plus_base"
    folder.mkdir(parents=True)
    (folder / "model.pt").write_bytes(b"weights")
    (folder / "config.yaml").write_text("model: Emotion2vec\n")
    return folder


def test_emotion_cache_roots_honour_modelscope_cache(monkeypatch):
    monkeypatch.setenv("MODELSCOPE_CACHE", "D:/ms-cache")
    roots = emotion._cache_roots()
    assert roots[0] == emotion.Path("D:/ms-cache") and roots[-1].name == "modelscope"


def test_emotion_model_comes_from_the_local_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_EMOTION_MODEL", raising=False)
    monkeypatch.setattr(emotion, "_cache_roots", lambda: [tmp_path / "ms"])
    assert emotion.model_source() == emotion.MODEL_ID                 # nothing cached
    folder = _cached_model(tmp_path / "ms")
    assert emotion.model_source() == str(folder)
    monkeypatch.setenv("CLIPFORGE_EMOTION_MODEL", "C:/models/emo")
    assert emotion.model_source() == "C:/models/emo"


def test_emotion_load_uses_the_local_copy_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.delenv("CLIPFORGE_EMOTION_MODEL", raising=False)
    monkeypatch.setattr(emotion, "_cache_roots", lambda: [tmp_path / "ms"])
    folder = _cached_model(tmp_path / "ms")
    calls = []

    class AutoModel:
        def __init__(self, *, model, device, disable_update):
            calls.append(model)
            if model == emotion.MODEL_ID:
                raise RuntimeError("model 'iic/emotion2vec_plus_base' is not registered")

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=AutoModel))
    emotion._model = None
    assert emotion._load() is not None and calls == [str(folder)]
    assert "emotion" not in engine_status.problems()

    (folder / "model.pt").unlink()                     # no local copy: by name, which fails
    emotion._model = None
    assert emotion._load() is None
    problem = engine_status.problems()["emotion"]
    assert "not registered" in problem and "No local copy" in problem


# --- Ollama: load before the budget, prewarm ---------------------------------------

def test_ensure_loaded_reports_and_clears_the_ollama_problem(monkeypatch):
    def refused(*_a, **_kw):
        raise OSError("connection refused")

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", refused)
    assert not ollama_client.ensure_loaded("qwen2.5vl:7b", 16384)
    assert "connection refused" in engine_status.problems()["ollama"]

    class Ok:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", lambda *a, **kw: Ok())
    assert ollama_client.ensure_loaded("qwen2.5vl:7b", 16384)
    assert "ollama" not in engine_status.problems()


def test_ready_uses_the_vision_context_for_images(monkeypatch):
    seen = []
    monkeypatch.setattr(ollama_client, "ensure_loaded", lambda m, ctx: seen.append((m, ctx)) or True)
    assert ollama_client.ready("vl", images=True) and ollama_client.ready("txt")
    assert seen == [("vl", ollama_client._num_ctx(True)), ("txt", ollama_client._num_ctx(False))]


def test_prewarm_loads_in_the_background_once(monkeypatch):
    started, release = threading.Event(), threading.Event()
    loads = []

    def slow_load(model, ctx):
        loads.append(model)
        started.set()
        release.wait(5)
        return True

    monkeypatch.setattr(ollama_client, "ensure_loaded", slow_load)
    t0 = time.monotonic()
    assert ollama_client.prewarm("vl", images=True)                # returns at once
    assert time.monotonic() - t0 < 1.0 and started.wait(5)
    assert not ollama_client.prewarm("vl", images=True)            # already loading
    release.set()
    for _ in range(50):
        if not ollama_client._PREWARMING:
            break
        time.sleep(0.02)
    assert loads == ["vl"] and not ollama_client._PREWARMING
    assert not ollama_client.prewarm("")


def test_the_vlm_batch_loads_the_model_before_its_budget(monkeypatch):
    order = []
    monkeypatch.setattr(vlm, "active_model", lambda: "vl")
    monkeypatch.setattr(vlm._oc, "ready", lambda m, images=False: order.append(("ready", m, images)) or True)
    monkeypatch.setattr(vlm, "score_visual", lambda *a, **kw: order.append("clip") or (0.8, "ok"))
    reads = vlm.score_visuals("v.mp4", [(0.0, 10.0), (20.0, 30.0)], budget=5.0, max_workers=1)
    assert reads == {0: (0.8, "ok"), 1: (0.8, "ok")}
    assert order[0] == ("ready", "vl", True) and order.count("clip") == 2


def test_no_vlm_or_llm_batch_without_a_loaded_model(monkeypatch):
    monkeypatch.setattr(vlm, "active_model", lambda: "vl")
    monkeypatch.setattr(vlm._oc, "ready", lambda *a, **kw: False)
    called = []
    monkeypatch.setattr(vlm, "score_visual", lambda *a, **kw: called.append(1))
    t0 = time.monotonic()
    assert vlm.score_visuals("v.mp4", [(0.0, 10.0)], budget=30.0) == {}
    assert time.monotonic() - t0 < 1.0 and called == []            # no budget spent waiting

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "active_model", lambda: "txt")
    monkeypatch.setattr(llm._oc, "ready", lambda *a, **kw: False)
    assert llm.score_virals(["a clip"], budget=30.0) == {}
    assert llm.suggest_titles(["a clip"], budget=30.0) == {}


def test_the_pipeline_prewarms_only_when_the_vlm_will_run(monkeypatch):
    from app.pipeline import orchestrator as O

    warmed = []
    monkeypatch.setattr(ollama_client, "prewarm", lambda m, images=False: warmed.append((m, images)) or True)
    monkeypatch.setattr(vlm, "available", lambda: True)
    monkeypatch.setattr(vlm, "active_model", lambda: "vl")
    assert O._prewarm_vlm(types.SimpleNamespace(use_vlm=True))
    assert not O._prewarm_vlm(types.SimpleNamespace(use_vlm=False))
    monkeypatch.setattr(vlm, "available", lambda: False)
    assert not O._prewarm_vlm(types.SimpleNamespace(use_vlm=True))
    assert warmed == [("vl", True)]
