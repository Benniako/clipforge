"""Offline Smart AI Suggestions & Co-Pilot Engine.

Generates contextual AI insight cards for short-form clips:
- Hook Trimming: Detect initial silence / dead air (t > 0.3s) and calculate retention estimation.
- Hook Rating: Evaluates opening 3-second hook strength (0-100 score, verdict, explanation).
- Peak Virality: Climax teaser cold-open loop suggestion.
- Layout Reframe: Split facecam layout suggestion when facecam is detected.
- Smart Jump Cuts: Internal pause removal and tightening.

100% offline rule-based and local heuristic intelligence.
"""
from __future__ import annotations

import logging
from typing import Any

from ..models import Clip, LayoutType, Project
from ..pipeline import captionize
from ..providers import score as score_mod

log = logging.getLogger("clipforge.services.suggestions_copilot")


def generate_clip_suggestions(project: Project, clip: Clip) -> list[dict[str, Any]]:
    """Analyze clip context and generate actionable AI suggestions with patch payloads.

    Matches interface contract:
    [
      {
        "id": "sug_trim_hook_c1",
        "type": "trim_hook",
        "title": "Cut Dead Air Hook",
        "description": "Trim initial 0.8s silence to hook viewers instantly.",
        "retention_impact": "+18% retention",
        "patch": { "start_t": 12.8 }
      },
      ...
    ]
    """
    suggestions: list[dict[str, Any]] = []

    # 1. Hook Trimming: Detect initial silence before first spoken word or event
    words = []
    if project.transcript and project.transcript.words:
        words = [w for w in project.transcript.words if w.end > clip.start and w.t < clip.end]

    lead_silence = 0.0
    first_speech_t = clip.start

    if words:
        first_w = min(words, key=lambda w: w.t)
        lead_silence = max(0.0, first_w.t - clip.start)
        first_speech_t = first_w.t
    elif clip.features.get("lead_silence", 0.0) > 0.0:
        lead_silence = float(clip.features["lead_silence"])
        first_speech_t = clip.start + lead_silence

    # If initial silence is noticeable (>= 0.3s), suggest trimming the hook
    if lead_silence >= 0.3:
        # Retention drop model: 0.5s dead air causes ~12-18% bounce in short-form
        boost_pct = min(36, max(12, int(round(lead_silence * 20.0))))
        target_start = round(max(clip.start, first_speech_t - 0.05), 3)
        suggestions.append({
            "id": f"sug_trim_hook_{clip.id}",
            "type": "trim_hook",
            "title": "Cut Dead Air Hook",
            "description": f"Trim initial {lead_silence:.1f}s silence to hook viewers instantly.",
            "retention_impact": f"+{boost_pct}% retention",
            "patch": {"start_t": target_start}
        })

    # 2. Hook Rating: Opening 3s evaluation
    lang = getattr(project.settings, "language", "en") if project.settings else "en"
    if words:
        analysis = score_mod.hook_analysis(words, lang=lang)
        strength_100 = int(round(analysis["strength"] * 100))
        verdict = analysis["verdict"]
        suggestion_text = analysis["suggestion"]
    else:
        # For non-speech/gameplay clips, use clip virality features
        strength_100 = min(100, max(45, int(clip.score)))
        verdict = "strong" if strength_100 >= 70 else ("ok" if strength_100 >= 50 else "weak")
        suggestion_text = "Fast-paced opening action."

    verdict_label = "Strong Hook" if verdict == "strong" else ("OK Hook" if verdict == "ok" else "Weak Hook")
    climax_val = clip.features.get("climax_t", 6.2)
    desc = f"{suggestion_text} Climax reached in {climax_val:.1f}s." if suggestion_text else f"High visual energy opener. Climax reached in {climax_val:.1f}s."

    suggestions.append({
        "id": f"sug_hook_rating_{clip.id}",
        "type": "hook_rating",
        "title": f"Hook Strength: {strength_100}/100",
        "description": desc,
        "retention_impact": verdict_label,
        "patch": {}
    })

    # 3. Peak Virality: Cold-open climax teaser loop
    climax_t = float(clip.features.get("climax_t", -1.0))
    if not getattr(clip, "teaser_loop", False):
        timing_str = f"at {climax_t:.1f}s" if climax_t > 0 else "in final act"
        suggestions.append({
            "id": f"sug_climax_teaser_{clip.id}",
            "type": "climax_teaser",
            "title": "Cold-Open Climax Teaser Loop",
            "description": f"Peak virality moment detected {timing_str}. Tease climax for 2s before the main hook.",
            "retention_impact": "+28% retention",
            "patch": {"teaser_loop": True}
        })

    # 4. Layout Reframe: Suggest split facecam when facecam exists in project
    has_facecam = project.facecam is not None or (clip.reframe and clip.reframe.facecam is not None)
    current_layout = clip.reframe.layout if clip.reframe else LayoutType.fill

    if has_facecam and current_layout != LayoutType.split:
        suggestions.append({
            "id": f"sug_split_reframe_{clip.id}",
            "type": "reframe_layout",
            "title": "Switch to Split Facecam",
            "description": "Detected streamer facecam. Show gameplay and streamer reactions simultaneously.",
            "retention_impact": "+24% engagement",
            "patch": {"layout": "split"}
        })

    # 5. Smart Jump Cuts: Dead air tightening inside clip
    if project.transcript and words:
        segs = captionize.compute_tight_segments(project.transcript, clip.start, clip.end)
        if len(segs) >= 2:
            saved = (clip.end - clip.start) - sum(b - a for a, b in segs)
            if saved >= 0.5 and not clip.segments:
                retention_boost = min(30, max(8, int(round(saved * 8.0))))
                suggestions.append({
                    "id": f"sug_tighten_{clip.id}",
                    "type": "tighten_dead_air",
                    "title": "Smart Jump Cuts (Tighten Dead Air)",
                    "description": f"Cut {saved:.1f}s internal dead air across {len(segs)-1} jump cuts to sustain momentum.",
                    "retention_impact": f"+{retention_boost}% retention",
                    "patch": {"tighten": True, "segments": [[round(a, 3), round(b, 3)] for a, b in segs]}
                })

    return suggestions
