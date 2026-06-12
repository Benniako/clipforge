"""The asynchronous processing pipeline.

Stage order (PRD §5.2 — transcript before detection before scoring; reframe and
render parallelise per clip):

    transcribe → detect → score → reframe → caption → render

Each stage is a small module; :mod:`orchestrator` sequences them and reports
honest, per-stage progress.
"""
