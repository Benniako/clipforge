"""Audio event detection via PANNs — score the *sounds* that signal a highlight.

Cheering, laughter, applause, a scream, an explosion, gunfire — these are the
sounds that make a moment go viral, and the transcript can't see them. PANNs
(Pretrained Audio Neural Networks, CNN14 on AudioSet's 527 classes) tags a clip's
audio in one fast forward pass. We sum the probabilities of the "hype" classes
into a 0..1 score and fold it into virality as an explainable factor, naming the
loudest class as the reason ("crowd cheering", "laughter").

Unlike the cue/OCR matchers this needs no per-game reference — it's zero-shot, so
it solves the cold-start problem for a brand-new game or an IRL clip.

Optional and graceful: no ``panns_inference`` ⇒ :func:`event_score` returns None
and scoring is unchanged. The class→score reduction is a pure function, so it's
unit-tested without the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import get_settings
from ..models import ScoreFactor

log = logging.getLogger("clipforge.audio_events")

_tagger = None  # cached (AudioTagging, labels) or False
_clap = None    # cached LAION-CLAP model or False

# AudioSet class-name substrings that read as a short-form highlight, grouped by
# the human reason we'll show. Matched case-insensitively against the 527 labels.
HYPE_CLASSES: dict[str, tuple[str, ...]] = {
    "crowd cheering": ("cheering", "crowd", "applause", "clapping"),
    "laughter": ("laughter", "giggle", "chuckle"),
    "excited shouting": ("screaming", "shout", "yell", "children shouting"),
    "explosive action": ("explosion", "gunshot", "gunfire", "machine gun",
                         "artillery", "boom"),
    "impact": ("smash", "crash", "shatter", "glass", "slam"),
}

CLAP_PROMPTS: dict[str, str] = {
    "crowd cheering": "people cheering and applauding loudly",
    "laughter": "people laughing hard",
    "excited shouting": "excited shouting and screaming",
    "explosive action": "explosions gunshots and intense action",
    "impact": "a loud crash smash or impact",
    "victory fanfare": "a triumphant game victory fanfare or win sound",
    "surprise reaction": "a streamer gasping in surprise or shock",
}

NEGATIVE_CLAP_PROMPTS: dict[str, str] = {
    "menu click": "video game menu click user interface sound",
    "lobby music": "calm video game lobby menu background music",
    "loading sound": "loading screen ambient loop or transition sound",
    "keyboard/mouse": "keyboard typing and mouse clicking at a desktop",
    "low energy ambience": "quiet low energy silence room tone or ambience",
}


@dataclass
class AudioEvent:
    t: float
    label: str
    confidence: float
    detail: str = ""


def available() -> bool:
    s = get_settings()
    return s.has_audio_events or s.has_clap


def _load():
    global _tagger
    if _tagger is not None:
        return _tagger or None
    try:
        from panns_inference import AudioTagging
        from panns_inference.config import labels

        device = "cuda" if get_settings().device == "cuda" else "cpu"
        _tagger = (AudioTagging(checkpoint_path=None, device=device), list(labels))
        log.info("PANNs audio tagging loaded (%d classes)", len(labels))
    except Exception as e:
        log.info("PANNs unavailable (%s)", e)
        _tagger = False
    return _tagger or None


def _load_clap():
    """Best-effort LAION-CLAP loader.

    The exact package is optional and heavy. Keep all imports inside the loader
    so normal ClipForge runs never pay for it unless the user installed it.
    """
    global _clap
    if _clap is not None:
        return _clap or None
    try:
        import laion_clap

        device = "cuda" if get_settings().device == "cuda" else "cpu"
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt()
        _clap = model
        log.info("LAION-CLAP audio tagging loaded")
    except Exception as e:
        log.info("CLAP audio tagging unavailable (%s)", e)
        _clap = False
    return _clap or None


def reduce_scores(probs: dict[str, float]) -> tuple[float, str] | None:
    """Collapse a {label: probability} map into (hype 0..1, reason).

    Pure (no model) so it's unit-testable. ``probs`` is the per-class output keyed
    by AudioSet label; we take, per hype group, its strongest member, then combine
    groups so several different cues stack but one alone still reads clearly.
    """
    group_best: dict[str, float] = {}
    for label, p in probs.items():
        low = label.lower()
        for reason, needles in HYPE_CLASSES.items():
            if any(n in low for n in needles):
                group_best[reason] = max(group_best.get(reason, 0.0), float(p))
    if not group_best:
        return None
    top_reason = max(group_best, key=group_best.get)
    # Combine groups so two distinct cues (cheer + laughter) beat one, but cap at
    # 1.0; the dominant group anchors the score.
    combined = 1.0 - 1.0
    for p in group_best.values():
        combined = combined + p - combined * p  # probabilistic OR
    return max(0.0, min(1.0, combined)), top_reason


def reduce_clap_similarities(sims: dict[str, float]) -> tuple[float, str] | None:
    """Collapse CLAP prompt similarities into a hype score.

    CLAP cosine values are not probabilities; this maps a useful similarity
    band into 0..1 and ignores weak matches so ambient noise is not rewarded.
    """
    if not sims:
        return None
    reason, sim = max(sims.items(), key=lambda kv: kv[1])
    if sim < 0.20:
        return None
    hype = (sim - 0.20) / 0.28
    return max(0.0, min(1.0, hype)), reason


def reduce_negative_similarities(sims: dict[str, float]) -> tuple[float, str] | None:
    """Collapse CLAP negative prompt similarities into a non-highlight risk."""
    if not sims:
        return None
    reason, sim = max(sims.items(), key=lambda kv: kv[1])
    if sim < 0.22:
        return None
    risk = (sim - 0.22) / 0.28
    return max(0.0, min(1.0, risk)), reason


def reduce_clap_window(pos_sims: dict[str, float],
                       neg_sims: dict[str, float]) -> tuple[float, str] | None:
    """Positive CLAP cue gated by non-highlight prompts."""
    pos = reduce_clap_similarities(pos_sims)
    if pos is None:
        return None
    hype, reason = pos
    neg = reduce_negative_similarities(neg_sims)
    if neg is not None:
        risk, neg_reason = neg
        if risk >= max(0.45, hype * 0.85):
            log.debug("CLAP rejected %s because %s risk %.2f", reason, neg_reason, risk)
            return None
        hype = max(0.0, hype - risk * 0.45)
        if hype < 0.25:
            return None
    return hype, reason


def _clap_similarity_rows(paths: list[str]) -> list[tuple[dict[str, float], dict[str, float]]]:
    if not paths:
        return []
    model = _load_clap()
    if model is None:
        return []
    try:
        import numpy as np

        all_prompts = [*CLAP_PROMPTS.values(), *NEGATIVE_CLAP_PROMPTS.values()]
        audio = model.get_audio_embedding_from_filelist(paths, use_tensor=False)
        text = model.get_text_embedding(all_prompts, use_tensor=False)
        audio = np.asarray(audio, dtype=np.float32)
        text = np.asarray(text, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if text.ndim == 1:
            text = text[None, :]
        audio = audio / np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-6)
        text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-6)
        vals = audio @ text.T
        pos_labels = list(CLAP_PROMPTS)
        neg_labels = list(NEGATIVE_CLAP_PROMPTS)
        rows: list[tuple[dict[str, float], dict[str, float]]] = []
        for row in vals:
            pos = {reason: float(row[i]) for i, reason in enumerate(pos_labels)}
            neg_off = len(pos_labels)
            neg = {reason: float(row[neg_off + i]) for i, reason in enumerate(neg_labels)}
            rows.append((pos, neg))
        return rows
    except Exception as e:
        log.warning("CLAP audio scoring failed (%s)", e)
        return []


def _clap_score(seg_path: str) -> tuple[float, str] | None:
    rows = _clap_similarity_rows([seg_path])
    if not rows:
        return None
    return reduce_clap_window(*rows[0])


def find_events(wav_path: str, duration: float, *, window: float = 6.0,
                hop: float = 3.0, threshold: float = 0.35,
                limit: int = 20) -> list[AudioEvent]:
    """Zero-shot CLAP search over audio windows.

    Returns highlight-like audio events (cheer/laugh/action/etc.) while filtering
    common non-highlights such as menu clicks, lobby music, and loading loops.
    """
    if not get_settings().has_clap or _load_clap() is None:
        return []
    if duration <= 0:
        return []
    try:
        import tempfile
        from pathlib import Path

        from ..media import ffmpeg

        window = max(1.0, float(window))
        hop = max(0.5, float(hop))
        max_windows = 180
        estimated = max(int(duration / hop), 1)
        if estimated > max_windows:
            hop = max(hop, duration / max_windows)
        starts: list[float] = []
        t = 0.0
        while t < duration:
            starts.append(round(t, 3))
            t += hop

        pairs: list[tuple[float, str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for i, start in enumerate(starts):
                seg = tmpdir / f"w{i:04d}.wav"
                dur = min(window, max(duration - start, 0.2))
                if dur < 0.4:
                    continue
                try:
                    ffmpeg.run(["-ss", f"{start:.3f}", "-i", wav_path,
                                "-t", f"{dur:.3f}", "-ac", "1", "-ar", "48000",
                                "-c:a", "pcm_s16le", str(seg)], timeout=45)
                    pairs.append((start + dur / 2.0, str(seg)))
                except Exception as e:
                    log.debug("CLAP window extract failed at %.1fs: %s", start, e)
            rows = _clap_similarity_rows([p for _, p in pairs])

        events: list[AudioEvent] = []
        for (t_mid, _path), row in zip(pairs, rows):
            res = reduce_clap_window(*row)
            if res is None:
                continue
            hype, reason = res
            if hype < threshold:
                continue
            events.append(AudioEvent(
                t=round(t_mid, 3), label=reason.replace(" ", "_"),
                confidence=round(hype, 4), detail=f"CLAP heard {reason}"))

        kept: list[AudioEvent] = []
        min_gap = max(window * 0.75, 3.0)
        for ev in sorted(events, key=lambda e: e.confidence, reverse=True):
            if all(abs(ev.t - k.t) >= min_gap for k in kept):
                kept.append(ev)
            if len(kept) >= limit:
                break
        kept.sort(key=lambda e: e.t)
        return kept
    except Exception as e:
        log.warning("CLAP event search failed (%s)", e)
        return []


def event_score(wav_path: str, start: float, end: float) -> tuple[float, str] | None:
    """(hype 0..1, reason) for a clip's audio span, or None.

    Uses PANNs when available, then CLAP as a zero-shot fallback. Reads only the
    clip's span so it stays cheap per clip.
    """
    if not available():
        return None
    try:
        import tempfile
        from pathlib import Path

        import numpy as np

        from ..media import ffmpeg

        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "seg.wav"
            # PANNs likes 32 kHz mono; CLAP accepts normal wav files too.
            ffmpeg.run(["-ss", f"{max(start, 0):.3f}", "-i", wav_path,
                        "-t", f"{max(end - start, 0.2):.3f}", "-ac", "1",
                        "-ar", "32000", "-c:a", "pcm_s16le", str(seg)], timeout=60)
            loaded = _load() if get_settings().has_audio_events else None
            if loaded is not None:
                import wave

                tagger, labels = loaded
                with wave.open(str(seg), "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if audio.size:
                    clipwise, _ = tagger.inference(audio[None, :])
                    probs = {labels[i]: float(clipwise[0][i]) for i in range(len(labels))}
                    pann = reduce_scores(probs)
                    if pann is not None:
                        return pann
            if get_settings().has_clap:
                return _clap_score(str(seg))
            return None
    except Exception as e:
        log.warning("audio-event scoring failed (%s)", e)
        return None


def apply_event_bonus(score: int, factors: list[ScoreFactor],
                      hype: float, reason: str, *, max_bonus: float = 10.0
                      ) -> tuple[int, list[ScoreFactor]]:
    """Lift a clip whose audio carries a viral sound (cheer/laugh/explosion).

    Positive-only — a quiet clip shouldn't be punished, it just won't get the
    boost. Shown as an explainable factor. Pure."""
    bonus = int(round(max(0.0, min(1.0, hype)) * max_bonus))
    if bonus <= 0:
        return score, factors
    new_score = int(max(1, min(99, score + bonus)))
    return new_score, [ScoreFactor(
        label=reason.capitalize(), weight=float(bonus),
        detail=f"Audio event detection heard {reason} ({int(hype*100)}/100)"),
        *factors]
