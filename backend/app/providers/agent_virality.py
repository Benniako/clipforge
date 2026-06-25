"""Optional tool-calling agent for multi-modal virality assessment.

The existing ``llm.score_viral`` reasons from a truncated transcript only.
When a tool-calling agent model (e.g. Qwen-AgentWorld-35B-A3B served via
vLLM) is available, this provider runs a full multi-step loop: the model
can call ``read_ocr_frames()``, ``analyze_audio_events()``, and
``check_facecam_reaction()`` before producing a final virality judgement.

Fully optional: no agent model available ⇒ falls back to the existing
single-shot LLM scorer without any behavioural change.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("clipforge.agent_virality")


def detected() -> bool:
    """True when a tool-calling agent model is wired.

    Currently a stub. Integration would:
    1. Check for a vLLM/Qwen-AgentWorld endpoint
    2. Define tool schemas for the existing provider functions
    3. Run a ReAct loop: call tool → get results → reason → call again → final score
    """
    return False


def score_clip(transcript_excerpt: str, *,
               src_path: str = "",
               start: float = 0.0, end: float = 0.0,
               lang: str = "de") -> tuple[float, str] | None:
    """Multi-step virality score using tool-calling.

    Returns ``(potential 0..1, reason)`` or None if unavailable/failed.
    Currently returns None — the orchestrator keeps its existing LLM path.
    """
    return None
