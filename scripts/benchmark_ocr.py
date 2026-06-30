#!/usr/bin/env python
"""Benchmark local OCR engines on screenshots or saved Cue Lab crops.

Examples:
  python scripts/benchmark_ocr.py crops/ --profile valorant --language de
  python scripts/benchmark_ocr.py frame.png --roi killfeed:0.55,0.02,0.35,0.22
  python scripts/benchmark_ocr.py frames/ --saved-regions --profile valorant --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.providers.ocr_benchmark import (  # noqa: E402
    benchmark,
    candidate_matrix,
    parse_roi,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="*",
                    help="Image file(s) or directories containing OCR crop images")
    ap.add_argument("--profile", default="generic",
                    help="Game profile for keyword matching and saved Cue Lab regions")
    ap.add_argument("--language", default="en",
                    help="OCR language hint, e.g. en or de")
    ap.add_argument("--engines", default="",
                    help="Comma-separated engines: paddleocr,easyocr,rapidocr,surya,tesseract")
    ap.add_argument("--roi", action="append", default=[],
                    help="Normalized crop box, label:x,y,w,h. Repeat for several ROIs.")
    ap.add_argument("--saved-regions", action="store_true",
                    help="Use saved Cue Lab regions for --profile/common")
    ap.add_argument("--expect", action="append", default=[],
                    help="Expected phrase. Repeat to score expected_hit.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of source images before ROI expansion")
    ap.add_argument("--include-unavailable", action="store_true",
                    help="Emit unavailable engines as errored rows")
    ap.add_argument("--list", action="store_true",
                    help="Only print candidate availability")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a table")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()] or None
    candidates = candidate_matrix(engines=engines)
    if args.list:
        rows = [c.to_dict() for c in candidates]
        print(json.dumps(rows, indent=2) if args.json else _candidate_table(candidates))
        return 0

    if not args.images:
        ap.error("at least one image file or directory is required unless --list is used")

    try:
        rois = [parse_roi(raw) for raw in args.roi]
    except ValueError as exc:
        ap.error(str(exc))

    rows = benchmark(
        args.images,
        candidates=candidates,
        profile=args.profile,
        language=args.language,
        regions=rois,
        use_saved_regions=args.saved_regions,
        expected=args.expect,
        include_unavailable=args.include_unavailable,
        limit=args.limit or None,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in rows], indent=2))
    else:
        print(_result_table(rows))
    return 0


def _candidate_table(candidates) -> str:
    lines = ["available  engine      label       reason"]
    for c in candidates:
        lines.append(f"{str(c.available):9}  {c.engine:10}  {c.label:10}  {c.reason}")
    return "\n".join(lines)


def _result_table(results) -> str:
    lines = [
        "seconds  conf   hit    engine      region       matches                  source  text/error"
    ]
    for r in results:
        hit = "" if r.expected_hit is None else str(r.expected_hit)
        matches = ",".join(m["label"] for m in r.matches) or "-"
        text = (r.error or r.text or "").replace("\n", " ")
        if len(text) > 80:
            text = text[:77] + "..."
        lines.append(
            f"{r.seconds:7.3f}  {r.confidence:5.2f}  {hit:5}  "
            f"{r.candidate.engine:10}  {r.sample.region[:11]:11}  "
            f"{matches[:22]:22}  {Path(r.sample.source).name[:24]:24}  {text}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

