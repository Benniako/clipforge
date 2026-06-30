"""Local OCR benchmark candidates for gameplay HUD/cue crops.

This harness is intentionally opt-in. It runs the same OCR adapter functions
used by the gameplay detector so benchmarks reflect production behavior.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import detect_ocr


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class OcrCandidate:
    engine: str
    label: str
    available: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OcrBenchRegion:
    label: str
    x: float
    y: float
    w: float
    h: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OcrBenchSample:
    path: str
    source: str
    region: str = "full"
    box: OcrBenchRegion | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.box is not None:
            d["box"] = self.box.to_dict()
        return d


@dataclass(frozen=True)
class OcrBenchResult:
    candidate: OcrCandidate
    sample: OcrBenchSample
    seconds: float
    confidence: float
    text: str
    matches: list[dict[str, str]]
    expected_hit: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "sample": self.sample.to_dict(),
            "seconds": self.seconds,
            "confidence": self.confidence,
            "text": self.text,
            "matches": self.matches,
            "expected_hit": self.expected_hit,
            "error": self.error,
        }


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _canon_engine(engine: str) -> str:
    aliases = {
        "paddle": "paddleocr",
        "easy": "easyocr",
        "rapid": "rapidocr",
        "rapidocr_onnxruntime": "rapidocr",
    }
    return aliases.get((engine or "").strip().lower(), (engine or "").strip().lower())


def candidate_matrix(*, engines: list[str] | None = None) -> list[OcrCandidate]:
    """Installed-aware OCR candidates worth benchmarking locally."""
    requested = engines or ["paddleocr", "easyocr", "rapidocr", "surya", "tesseract"]
    seen: set[str] = set()
    out: list[OcrCandidate] = []
    for raw in requested:
        engine = _canon_engine(raw)
        if not engine or engine in seen:
            continue
        seen.add(engine)
        if engine == "paddleocr":
            ok = _has_module("paddleocr")
            reason = "" if ok else "paddleocr is not installed"
            label = "PaddleOCR"
        elif engine == "easyocr":
            ok = _has_module("easyocr")
            reason = "" if ok else "easyocr is not installed"
            label = "EasyOCR"
        elif engine == "rapidocr":
            ok = _has_module("rapidocr") or _has_module("rapidocr_onnxruntime")
            reason = "" if ok else "rapidocr is not installed"
            label = "RapidOCR"
        elif engine == "surya":
            ok = _has_module("surya")
            reason = "" if ok else "surya is not installed"
            label = "Surya"
        elif engine == "tesseract":
            ok = _has_module("pytesseract") and bool(shutil.which("tesseract"))
            reason = "" if ok else "pytesseract or the tesseract binary is missing"
            label = "Tesseract"
        else:
            ok = False
            reason = f"unknown OCR engine: {raw}"
            label = raw
        out.append(OcrCandidate(engine=engine, label=label, available=ok, reason=reason))
    return out


def parse_roi(raw: str) -> OcrBenchRegion:
    """Parse ``label:x,y,w,h`` or ``x,y,w,h`` normalized ROI syntax."""
    raw = (raw or "").strip()
    label = "roi"
    coords = raw
    if ":" in raw:
        label, coords = raw.split(":", 1)
        label = _safe_label(label)
    parts = [p.strip() for p in coords.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be label:x,y,w,h or x,y,w,h")
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("ROI coordinates must be numbers") from exc
    return _region(label, x, y, w, h)


def iter_image_paths(inputs: list[str | Path]) -> list[Path]:
    """Expand files/directories into sorted image paths."""
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            out.extend(x for x in sorted(p.rglob("*")) if x.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
        elif not p.exists():
            raise FileNotFoundError(str(p))
    return out


def prepare_samples(inputs: list[str | Path], *, tmp_dir: Path,
                    regions: list[OcrBenchRegion] | None = None,
                    use_saved_regions: bool = False,
                    profile: str | None = None,
                    limit: int | None = None) -> list[OcrBenchSample]:
    """Build full-image or cropped OCR samples."""
    paths = iter_image_paths(inputs)
    if limit and limit > 0:
        paths = paths[:limit]
    boxes = list(regions or [])
    if use_saved_regions:
        from .. import visual_cues

        for label, saved in visual_cues.regions_extra(profile).items():
            for idx, raw in enumerate(saved):
                boxes.append(_region(
                    str(raw.get("name") or f"{label}_{idx}"),
                    float(raw.get("x", 0.0)),
                    float(raw.get("y", 0.0)),
                    float(raw.get("w", 1.0)),
                    float(raw.get("h", 1.0)),
                ))

    if not boxes:
        return [OcrBenchSample(path=str(p), source=str(p), region="full") for p in paths]

    tmp_dir.mkdir(parents=True, exist_ok=True)
    samples: list[OcrBenchSample] = []
    for img_i, src in enumerate(paths):
        for roi_i, box in enumerate(boxes):
            crop = tmp_dir / f"ocr_bench_{img_i:04d}_{roi_i:02d}_{box.label}.png"
            _crop_image(src, box, crop)
            samples.append(OcrBenchSample(
                path=str(crop), source=str(src), region=box.label, box=box))
    return samples


def run_candidate(candidate: OcrCandidate, sample: OcrBenchSample, *,
                  language: str = "en", profile: str | None = None,
                  expected: list[str] | None = None) -> OcrBenchResult:
    """Run one OCR candidate against one sample."""
    started = time.perf_counter()
    if not candidate.available:
        return OcrBenchResult(
            candidate, sample, 0.0, 0.0, "", [],
            expected_hit=None,
            error=f"unavailable: {candidate.reason}",
        )
    try:
        text, confidence = detect_ocr._ocr_image_conf(sample.path, candidate.engine, language)
        matches = [
            {"label": label, "phrase": phrase}
            for label, phrase in detect_ocr.match_keywords(text, profile)
        ]
        return OcrBenchResult(
            candidate=candidate,
            sample=sample,
            seconds=round(time.perf_counter() - started, 3),
            confidence=round(float(confidence or 0.0), 4),
            text=text,
            matches=matches,
            expected_hit=_expected_hit(text, expected),
        )
    except Exception as exc:
        return OcrBenchResult(
            candidate, sample, round(time.perf_counter() - started, 3),
            0.0, "", [], expected_hit=None, error=str(exc)[:300])


def benchmark(inputs: list[str | Path], *,
              candidates: list[OcrCandidate] | None = None,
              profile: str | None = None,
              language: str = "en",
              regions: list[OcrBenchRegion] | None = None,
              use_saved_regions: bool = False,
              expected: list[str] | None = None,
              include_unavailable: bool = False,
              limit: int | None = None,
              tmp_dir: Path | None = None) -> list[OcrBenchResult]:
    """Benchmark OCR candidates over image files or directories."""
    import tempfile

    cand = candidates or candidate_matrix()
    if not include_unavailable:
        cand = [c for c in cand if c.available]
    if tmp_dir is not None:
        samples = prepare_samples(
            inputs, tmp_dir=tmp_dir, regions=regions,
            use_saved_regions=use_saved_regions, profile=profile, limit=limit)
        return [
            run_candidate(c, s, language=language, profile=profile, expected=expected)
            for c in cand for s in samples
        ]

    with tempfile.TemporaryDirectory() as td:
        samples = prepare_samples(
            inputs, tmp_dir=Path(td), regions=regions,
            use_saved_regions=use_saved_regions, profile=profile, limit=limit)
        return [
            run_candidate(c, s, language=language, profile=profile, expected=expected)
            for c in cand for s in samples
        ]


def _safe_label(label: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", (label or "roi").strip().lower()).strip("_") or "roi"


def _region(label: str, x: float, y: float, w: float, h: float) -> OcrBenchRegion:
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    w = max(0.01, min(1.0 - x, w))
    h = max(0.01, min(1.0 - y, h))
    return OcrBenchRegion(_safe_label(label), round(x, 4), round(y, 4),
                          round(w, 4), round(h, 4))


def _crop_image(src: Path, region: OcrBenchRegion, dst: Path) -> None:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("Pillow is required for ROI cropping") from exc

    with Image.open(src) as img:
        im = img.convert("RGB")
        iw, ih = im.size
        x0 = max(0, min(iw - 1, int(region.x * iw)))
        y0 = max(0, min(ih - 1, int(region.y * ih)))
        x1 = max(x0 + 1, min(iw, int((region.x + region.w) * iw)))
        y1 = max(y0 + 1, min(ih, int((region.y + region.h) * ih)))
        crop = ImageOps.autocontrast(im.crop((x0, y0, x1, y1)))
        crop.save(dst)


def _expected_hit(text: str, expected: list[str] | None) -> bool | None:
    phrases = [p for p in (expected or []) if p.strip()]
    if not phrases:
        return None
    norm_text = detect_ocr._norm(text)
    return any(detect_ocr._norm(p) in norm_text for p in phrases)

