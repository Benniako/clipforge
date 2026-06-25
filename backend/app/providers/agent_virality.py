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
import os
import urllib.request

log = logging.getLogger("clipforge.agent_virality")

# vLLM default endpoint. Override via CLIPFORGE_VLLM_URL.
_VLLM_URL = os.environ.get("CLIPFORGE_VLLM_URL", "http://127.0.0.1:8001")


def detected() -> bool:
    """True when a tool-calling agent model is reachable.

    Probes for:
    1. A vLLM server at the configured URL (default port 8001).
    2. The Qwen-AgentWorld model in Ollama (via /api/tags).
    """
    try:
        req = urllib.request.Request(
            _VLLM_URL + "/v1/models",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as r:
            if r.status == 200:
                return True
    except Exception:
        pass
    # Fallback: check if Ollama has a capable agent model.
    try:
        ollama_url = os.environ.get("CLIPFORGE_OLLAMA_URL",
                                     "http://127.0.0.1:11434")
        with urllib.request.urlopen(ollama_url + "/api/tags", timeout=1.5) as r:
            import json
            data = json.loads(r.read())
            for m in data.get("models", []):
                name = (m.get("name") or "").lower()
                if any(kw in name for kw in ("agentworld", "qwen3", "deepseek-r1")):
                    return True
    except Exception:
        pass
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
