"""Tests for the RapidOCR engine: the HUD reader, the keyword-OCR engine and
how the engine is chosen. A fake ``rapidocr`` module stands in for the
package, so these run without it (CI does not install the optional extras).

    cd backend && python -m pytest tests/test_ocr_rapid.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

os.environ.setdefault("CLIPFORGE_DATA_DIR", tempfile.mkdtemp(prefix="clipforge-test-"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app import config, engine_status  # noqa: E402
from app.providers import detect_ocr  # noqa: E402
from app.providers.hud import readers as R  # noqa: E402


class FakeOutput:
    def __init__(self, txts, scores):
        self.txts, self.scores = txts, scores


def fake_rapidocr(monkeypatch, *, read=None, fail=False) -> list:
    """Install a fake ``rapidocr``; returns the params each engine was made with."""
    made: list = []

    class RapidOCR:
        def __init__(self, params=None):
            if fail:
                raise RuntimeError("onnxruntime missing")
            made.append(params)

        def __call__(self, img, **kw):
            return read(img, kw) if read else FakeOutput((), ())

    monkeypatch.setitem(sys.modules, "rapidocr", types.SimpleNamespace(RapidOCR=RapidOCR))
    return made


@pytest.fixture(autouse=True)
def _fresh_keyword_reader(monkeypatch):
    monkeypatch.setattr(detect_ocr, "_reader", None)
    yield
    engine_status.clear("rapidocr")


# --- HUD reader --------------------------------------------------------------------

def test_restrict_maps_digit_lookalikes_and_drops_the_rest():
    assert R.restrict("1O:2S", R.CLOCK_CHARS) == "10:25"
    assert R.restrict(" 3 5 ", R.DIGITS) == "35"
    assert R.restrict("l2'", R.DIGITS) == "12"
    assert R.restrict("90:56+2", R.CLOCK_CHARS) == "90:56+2"
    assert R.restrict("", R.DIGITS) == ""


def test_rapid_reader_reads_without_the_detector(monkeypatch):
    calls = []

    def read(img, kw):
        calls.append(kw)
        return FakeOutput(("1O:23",), (0.97,))

    made = fake_rapidocr(monkeypatch, read=read)
    reader = R.RapidOcrReader()
    assert made[0]["Global.use_det"] is False
    assert reader.read_line(np.zeros((30, 90, 3), np.uint8), R.CLOCK_CHARS) == ("10:23", 0.97)
    assert calls[-1] == {"use_det": False, "use_cls": False, "use_rec": True}
    assert reader.read_digits(None) == ("", 0.0)
    assert reader.read_digits(np.zeros((0, 0), np.uint8)) == ("", 0.0)


def test_rapid_reader_with_nothing_recognised(monkeypatch):
    fake_rapidocr(monkeypatch, read=lambda img, kw: FakeOutput(None, None))
    assert R.RapidOcrReader().read_digits(np.zeros((30, 30), np.uint8)) == ("", 0.0)


def test_pick_readers_prefers_rapidocr_and_falls_back_without_it(monkeypatch):
    fake_rapidocr(monkeypatch)
    digit, line = R.pick_readers()
    assert digit is line and digit.name == "rapidocr"

    fake_rapidocr(monkeypatch, fail=True)
    easy = types.SimpleNamespace(name="easyocr")
    monkeypatch.setattr(R, "EasyOcrReader", lambda: easy)

    def no_tesseract():
        raise RuntimeError("no tesseract")

    monkeypatch.setattr(R, "TesseractCliReader", no_tesseract)
    assert R.pick_readers() == (easy, easy)


# --- keyword OCR engine -----------------------------------------------------------

def test_rapid_read_joins_lines_and_averages_confidence():
    reader = lambda path: FakeOutput(("VERLOREN", "", "TEAM ELIMINIERT"), (0.9, 0.1, 1.0))  # noqa: E731
    text, conf = detect_ocr._rapid_read(reader, "x.png")
    assert text == "VERLOREN TEAM ELIMINIERT" and conf == pytest.approx(0.95)
    assert detect_ocr._rapid_read(lambda p: FakeOutput(None, None), "x.png") == ("", 0.0)


def test_rapidocr_engine_reads_frames_and_falls_back_to_easyocr(monkeypatch, tmp_path):
    made = fake_rapidocr(monkeypatch, read=lambda img, kw: FakeOutput(("KAUFPHASE",), (0.99,)))
    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), np.zeros((40, 120, 3), np.uint8))
    assert detect_ocr._ocr_image_conf(str(frame), "rapidocr", "de") == ("KAUFPHASE", 0.99)
    assert made[0]["Det.limit_type"] == "max"           # frames are only ever shrunk

    # RapidOCR cannot start: the problem is reported and EasyOCR takes over.
    fake_rapidocr(monkeypatch, fail=True)
    monkeypatch.setattr(detect_ocr, "_reader", None)
    monkeypatch.setattr(detect_ocr, "_make_easyocr", lambda gpu, langs=None: "easy-reader")
    monkeypatch.setattr(detect_ocr, "_try_with_timeout", lambda fn, timeout=15.0: fn())
    assert detect_ocr._get_reader("rapidocr", "de") == ("easyocr", "easy-reader")
    assert "onnxruntime missing" in engine_status.problems()["rapidocr"]


# --- engine selection -------------------------------------------------------------

def test_rapidocr_is_the_preferred_engine(monkeypatch):
    present = {"rapidocr", "easyocr", "paddleocr"}
    monkeypatch.setattr(config, "_has_module", lambda name: name in present)
    assert config._detect_ocr(has_gpu=True) == "rapidocr"
    present.discard("rapidocr")
    assert config._detect_ocr(has_gpu=True) == "easyocr"
    assert config._detect_ocr(has_gpu=False) == "paddleocr"


def test_rapidocr_heads_the_ocr_capabilities():
    ocr = next(c for c in config.get_settings().capability_detail()["categories"]
               if c["name"] == "ocr")
    assert ocr["items"][0]["key"] == "rapidocr"
    assert "rapidocr" in config.get_settings().capability_report()
