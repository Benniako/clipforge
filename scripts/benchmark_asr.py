#!/usr/bin/env python
"""Benchmark local ASR engines on one audio/video file.

Example:
  python scripts/benchmark_asr.py path/to/audio.wav --language de --device cuda
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

from app.providers.asr_benchmark import benchmark, candidate_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="?",
                    help="Audio/video file readable by the ASR backend")
    ap.add_argument("--language", default=None, help="Language code, or omit/auto")
    ap.add_argument("--device", default=None, choices=("cpu", "cuda"),
                    help="Override device for faster-whisper/whisperX")
    ap.add_argument("--compute-type", default=None,
                    help="Override compute type, e.g. int8, float16")
    ap.add_argument("--models", default="",
                    help="Comma-separated faster-whisper models to compare")
    ap.add_argument("--list", action="store_true",
                    help="Only print candidate availability")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a table")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    candidates = candidate_matrix(models=models)
    if args.list:
        rows = [c.to_dict() for c in candidates]
        print(json.dumps(rows, indent=2) if args.json else _candidate_table(candidates))
        return 0

    if not args.audio:
        ap.error("audio is required unless --list is used")

    rows = benchmark(args.audio, candidates=candidates, language=args.language,
                     device=args.device, compute_type=args.compute_type)
    if args.json:
        print(json.dumps([r.to_dict() for r in rows], indent=2))
    else:
        print(_result_table(rows))
    return 0


def _candidate_table(candidates) -> str:
    lines = ["available  engine          model                         label"]
    for c in candidates:
        lines.append(f"{str(c.available):9}  {c.engine:14}  {c.model[:28]:28}  {c.label}")
        if c.reason:
            lines.append(f"           reason: {c.reason}")
    return "\n".join(lines)


def _result_table(results) -> str:
    lines = ["seconds  words  language  engine          model                         error"]
    for r in results:
        err = r.error or ""
        lines.append(
            f"{r.seconds:7.3f}  {r.words:5d}  {(r.language or '')[:8]:8}  "
            f"{r.candidate.engine:14}  {r.candidate.model[:28]:28}  {err}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
