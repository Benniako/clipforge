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
import sys
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

CLAP_PROMPTS: dict[str, tuple[str, ...]] = {
    "crowd cheering": (
        "people cheering and applauding loudly",
        "a crowd erupts after an exciting sports or game moment",
        "German viewers cheering after a win",
    ),
    "laughter": (
        "people laughing hard",
        "a streamer laughing at a funny moment",
        "German streamer laughter",
    ),
    "excited shouting": (
        "excited shouting and screaming",
        "a streamer yells loudly during intense gameplay",
        "German excited gaming commentary",
    ),
    "explosive action": (
        "explosions gunshots and intense action",
        "rapid shooter combat with gunfire and impacts",
        "intense gameplay fight sounds",
    ),
    "impact": (
        "a loud crash smash or impact",
        "a sudden hit impact or heavy thud",
    ),
    "victory fanfare": (
        "a triumphant game victory fanfare or win sound",
        "round won music sting in a competitive game",
    ),
    "surprise reaction": (
        "a streamer gasping in surprise or shock",
        "shocked voice reaction after an unexpected moment",
    ),
}

GAME_CLAP_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "valorant": {
        "valorant kill": (
            "Valorant kill confirmation sound",
            "tactical shooter headshot kill sound",
            "enemy eliminated sound in Valorant",
        ),
        "spike objective": (
            "Valorant spike planted announcer",
            "Valorant spike defused announcer",
            "tactical shooter bomb plant or defuse announcer",
        ),
        "round win": (
            "Valorant round victory announcer",
            "team won the round in a tactical shooter",
        ),
    },
    "cs2": {
        "cs2 kill": (
            "Counter Strike headshot kill sound",
            "Counter Strike enemy killed gunshot impact",
        ),
        "bomb objective": (
            "Counter Strike bomb planted announcer",
            "Counter Strike bomb defused announcer",
        ),
        "round win": (
            "Counter Terrorists win announcer",
            "Terrorists win announcer in Counter Strike",
        ),
    },
    "eafc": {
        "goal celebration": (
            "football goal crowd celebration",
            "soccer commentator screams goal",
            "German football goal celebration commentary",
        ),
        "referee whistle": (
            "football referee whistle at an important play",
            "penalty whistle in a football match",
        ),
    },
    "rocketleague": {
        "goal explosion": (
            "Rocket League goal explosion",
            "arcade car soccer goal scored sound",
        ),
        "epic save": (
            "Rocket League epic save announcer",
            "arcade car soccer save announcer",
        ),
    },
}
_PROFILE_ALIAS = {"auto": "generic", "cs": "cs2", "fifa": "eafc"}

NEGATIVE_CLAP_PROMPTS: dict[str, tuple[str, ...]] = {
    "menu click": (
        "video game menu click user interface sound",
        "game settings menu navigation clicks",
        "agent select or inventory menu user interface",
    ),
    "lobby music": (
        "calm video game lobby menu background music",
        "ambient waiting room music before a match starts",
        "non gameplay menu music loop",
    ),
    "loading sound": (
        "loading screen ambient loop or transition sound",
        "matchmaking loading screen with no action",
    ),
    "keyboard/mouse": (
        "keyboard typing and mouse clicking at a desktop",
        "computer mouse clicks without gameplay action",
    ),
    "low energy ambience": (
        "quiet low energy silence room tone or ambience",
        "steady background hum with no exciting event",
    ),
}


@dataclass
class AudioEvent:
    t: float
    label: str
    confidence: float
    detail: str = ""


def available() -> bool:
    s = get_settings()
    return ((s.has_audio_events and _tagger is not False)
            or (s.has_clap and _clap is not False))


def capability_flags() -> dict[str, bool]:
    """Runtime-aware health flags for optional audio detectors.

    ``get_settings`` can only know that a package is installed. Model downloads
    and checkpoint compatibility are discovered lazily when the detector loads,
    so a failed load should turn the UI dot off instead of staying green.
    """
    s = get_settings()
    panns = bool(s.has_audio_events and _tagger is not False)
    clap = bool(s.has_clap and _clap is not False)
    return {
        "audio_events": panns or clap,
        "panns_audio": panns,
        "clap_audio": clap,
    }


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
        old_argv = sys.argv[:]
        try:
            # laion_clap imports training argparse helpers; hide uvicorn args.
            sys.argv = [old_argv[0] if old_argv else "clipforge"]
            import laion_clap

            device = "cuda" if get_settings().device == "cuda" else "cpu"
            model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
            _load_clap_checkpoint(model)
            _clap = model
            log.info("LAION-CLAP audio tagging loaded")
        finally:
            sys.argv = old_argv
    except BaseException as e:
        log.info("CLAP audio tagging unavailable (%s)", e)
        _clap = False
    return _clap or None


def _load_clap_checkpoint(model) -> None:
    """Load LAION-CLAP weights across package / PyTorch version mismatches."""
    try:
        model.load_ckpt()
        return
    except Exception as first:
        err = first

    if "weights_only" in str(err):
        try:
            _call_with_torch_load_compat(model.load_ckpt)
            return
        except Exception as second:
            err = second

    # Some current CLAP checkpoints contain a harmless RoBERTa position-id
    # buffer that older package builds did not expect. Loading non-strictly keeps
    # the actual weights while avoiding a hard failure on that metadata tensor.
    if "Unexpected key(s) in state_dict" in str(err):
        core = getattr(model, "model", None)
        original = getattr(core, "load_state_dict", None)
        if original is not None:
            def _load_state_dict_compat(state_dict, *args, **kwargs):
                kwargs["strict"] = False
                return original(state_dict, *args, **kwargs)

            core.load_state_dict = _load_state_dict_compat
            try:
                try:
                    model.load_ckpt()
                except Exception as second:
                    if "weights_only" not in str(second):
                        raise
                    _call_with_torch_load_compat(model.load_ckpt)
                return
            finally:
                core.load_state_dict = original

    raise err


def _call_with_torch_load_compat(fn) -> None:
    import torch

    old_load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old_load(*args, **kwargs)

    torch.load = _load_compat
    try:
        fn()
    finally:
        torch.load = old_load


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
    combined = 0.0  # identity for the probabilistic-OR fold below
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
    raw_pos = float(pos_sims.get(reason, 0.0))
    raw_neg = max((float(v) for v in neg_sims.values()), default=-1.0)
    if raw_pos - raw_neg < 0.06:
        log.debug("CLAP rejected %s because positive/negative margin was %.3f",
                  reason, raw_pos - raw_neg)
        return None
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


def _normalize_profile(profile: str | None) -> str:
    name = (profile or "generic").lower().replace(" ", "")
    return _PROFILE_ALIAS.get(name, name)


def _prompt_sets(profile: str | None = None, language: str | None = None
                 ) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    pos = {label: tuple(prompts) for label, prompts in CLAP_PROMPTS.items()}
    name = _normalize_profile(profile)
    for label, prompts in GAME_CLAP_PROMPTS.get(name, {}).items():
        pos[label] = tuple(dict.fromkeys(pos.get(label, ()) + prompts))
    lang = (language or "").lower()
    if lang.startswith("de"):
        pos["excited shouting"] = tuple(dict.fromkeys(
            pos.get("excited shouting", ()) + (
                "German streamer shouts after an intense play",
                "German gaming voice gets loud and excited",
            )))
        pos["surprise reaction"] = tuple(dict.fromkeys(
            pos.get("surprise reaction", ()) + (
                "German streamer gasps in surprise",
                "German voice shocked by a gameplay moment",
            )))
    return pos, NEGATIVE_CLAP_PROMPTS


def _flatten_prompts(groups: dict[str, tuple[str, ...]]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    prompts: list[str] = []
    for label, variants in groups.items():
        for prompt in variants:
            labels.append(label)
            prompts.append(prompt)
    return labels, prompts


def _group_prompt_scores(labels: list[str], values) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, val in zip(labels, values):
        scores[label] = max(scores.get(label, -1.0), float(val))
    return scores


def _clap_similarity_rows(paths: list[str], *, profile: str | None = None,
                          language: str | None = None
                          ) -> list[tuple[dict[str, float], dict[str, float]]]:
    if not paths:
        return []
    model = _load_clap()
    if model is None:
        return []
    try:
        import numpy as np

        pos_prompts, neg_prompts = _prompt_sets(profile, language)
        pos_labels, pos_texts = _flatten_prompts(pos_prompts)
        neg_labels, neg_texts = _flatten_prompts(neg_prompts)
        all_prompts = [*pos_texts, *neg_texts]
        audio = _embedding_array(model.get_audio_embedding_from_filelist, paths)
        text = _embedding_array(model.get_text_embedding, all_prompts)
        if audio.ndim == 1:
            audio = audio[None, :]
        if text.ndim == 1:
            text = text[None, :]
        audio = audio / np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-6)
        text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-6)
        vals = audio @ text.T
        rows: list[tuple[dict[str, float], dict[str, float]]] = []
        for row in vals:
            pos = _group_prompt_scores(pos_labels, row[:len(pos_labels)])
            neg_off = len(pos_labels)
            neg = _group_prompt_scores(neg_labels, row[neg_off:neg_off + len(neg_labels)])
            rows.append((pos, neg))
        return rows
    except Exception as e:
        log.warning("CLAP audio scoring failed (%s)", e)
        return []


def _embedding_array(fn, values):
    import numpy as np

    try:
        out = fn(values, use_tensor=False)
    except TypeError as e:
        if "use_tensor" not in str(e):
            raise
        out = fn(values)
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32)


def _clap_score(seg_path: str, *, profile: str | None = None,
                language: str | None = None) -> tuple[float, str] | None:
    rows = _clap_similarity_rows([seg_path], profile=profile, language=language)
    if not rows:
        return None
    return reduce_clap_window(*rows[0])


def find_events(wav_path: str, duration: float, *, window: float = 6.0,
                hop: float = 3.0, threshold: float = 0.35,
                limit: int = 20, profile: str | None = None,
                language: str | None = None) -> list[AudioEvent]:
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
            rows = _clap_similarity_rows([p for _, p in pairs],
                                         profile=profile, language=language)

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


def event_score(wav_path: str, start: float, end: float, *,
                profile: str | None = None,
                language: str | None = None) -> tuple[float, str] | None:
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
                return _clap_score(str(seg), profile=profile, language=language)
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
