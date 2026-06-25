"""Job orchestration — sequence the stages and report honest progress.

A single background worker thread drains a queue of project ids and runs the
pipeline for each. Stages run in order; the render stage fans out across a thread
pool so clips encode in parallel (PRD §5.2). Progress is persisted after every
stage and every finished clip so the UI shows real, legible movement rather than
a fake spinner.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .. import feedback, store
from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import (ASPECTS, Clip, ClipStatus, ContentType, JobProgress,
                      LayoutType, ProjectStatus, Reframe, ReframeKeyframe,
                      now)
from ..providers import detect as detect_mod
from ..providers import detect_gameplay as gameplay_mod
from ..providers import hashtags as hashtags_mod
from ..providers import llm as llm_mod
from ..providers import scenes as scenes_mod
from ..providers import score as score_mod
from ..providers import transcribe as transcribe_mod
from ..styles import get_style
from . import captionize, classify, ingest
from . import facecam as facecam_mod
from . import montage as montage_mod
from . import reframe as reframe_mod
from . import render as render_mod

log = logging.getLogger("clipforge.engine")

# Known error patterns mapped to user-friendly messages.
_FRIENDLY_ERRORS: dict[str, str] = {
    "ffmpeg": (
        "The media engine (ffmpeg) encountered a problem. This can happen when the "
        "source video uses an uncommon codec or is corrupted. Try re-encoding to "
        "H.264 MP4 with HandBrake or ffmpeg first."
    ),
    "whisper": (
        "Speech-to-text (Whisper) failed. This can happen with very long or noisy "
        "audio. Try a shorter video, or set CLIPFORGE_WHISPER_MODEL=small for a "
        "lighter model."
    ),
    "memory": (
        "ClipForge ran out of memory. Try closing other applications, "
        "setting CLIPFORGE_WHISPER_MODEL=base or tiny, or processing a shorter video."
    ),
    "out of memory": (
        "ClipForge ran out of memory. Try closing other applications, "
        "setting CLIPFORGE_WHISPER_MODEL=base or tiny, or processing a shorter video."
    ),
    "CUDA": (
        "GPU acceleration failed. This can happen with outdated drivers or "
        "insufficient VRAM. Set CLIPFORGE_DEVICE=cpu to fall back to CPU processing "
        "(slower but more stable)."
    ),
    "disk": (
        "ClipForge ran out of disk space. Free up space on the drive where "
        "backend/data/ is located, or set CLIPFORGE_DATA_DIR to a different drive."
    ),
    "No space": (
        "ClipForge ran out of disk space. Free up space on the drive where "
        "backend/data/ is located, or set CLIPFORGE_DATA_DIR to a different drive."
    ),
    "url": (
        "Failed to import the video URL. Check the URL is correct and accessible. "
        "For private videos, download the file first and upload it directly."
    ),
    "timeout": (
        "An operation timed out. For very long videos this is expected; try a "
        "shorter video or increase the timeout by setting CLIPFORGE_TIMEOUT."
    ),
}


def _friendly_error(exc: BaseException) -> str:
    """Map an exception to a human-readable, actionable error message.

    Checks the exception message, its cause chain, and known patterns to produce
    output a non-expert user can act on.
    """
    msg = str(exc)
    for pattern, hint in _FRIENDLY_ERRORS.items():
        if pattern.lower() in msg.lower():
            return hint
    # Check the cause chain for wrapped ffmpeg errors.
    cause = exc
    while hasattr(cause, "__cause__") and cause.__cause__:
        cause = cause.__cause__
        for pattern, hint in _FRIENDLY_ERRORS.items():
            if pattern.lower() in str(cause).lower():
                return hint
    # If the exception has a 'category' attribute (FFmpegError), use it.
    if hasattr(exc, "category"):
        return str(exc.category)
    # Generic fallback — strip file paths and truncate for readability.
    clean = re.sub(r"[A-Za-z]:\\[^\s,)]+", "[path]", msg)
    clean = re.sub(r"/[^\s,)]+", "[path]", clean)
    if len(clean) > 300:
        clean = clean[:300] + "..."
    return clean


# Used instead of str.removeprefix() for Python <3.9 compatibility.
def _strip_media_prefix(url: str) -> str:
    prefix = "/media/"
    return url[len(prefix):] if url.startswith(prefix) else url


def _media_url(path) -> str:
    return f"/media/{path.relative_to(get_settings().media_dir).as_posix()}"


def _speech_intervals(transcript, start: float, end: float
                      ) -> list[tuple[float, float]] | None:
    """Clip-relative speech spans for speech-aware reframing.

    None for synthetic transcripts — their filler timing would mark the whole
    clip as speech and defeat the hold-during-silence behaviour.
    """
    if transcript is None or transcript.provider == "synthetic":
        return None
    segs = captionize.compute_tight_segments(transcript, start, end)
    return [(a - start, b - start) for a, b in segs]


# Cache for pre-computed full-timeline speech intervals.
_full_speech_intervals: list[tuple[float, float]] | None = None
_full_speech_transcript_id: int = 0


def _precompute_speech_intervals(transcript) -> None:
    """Compute speech intervals for the entire timeline once.

    Subsequent per-clip calls to ``_speech_intervals`` still rebase the
    intervals (subtract clip start), but the full-timeline scan of the
    transcript is done here once instead of per-clip.
    """
    global _full_speech_intervals, _full_speech_transcript_id
    if transcript is None or transcript.provider == "synthetic":
        _full_speech_intervals = None
        return
    # Cheap identity check — skip if already computed for this transcript.
    tid = id(transcript)
    if tid == _full_speech_transcript_id and _full_speech_intervals is not None:
        return
    _full_speech_transcript_id = tid
    _full_speech_intervals = transcript.words  # store words list for filtering
    log.debug("pre-computed speech intervals for transcript (%d words)",
              len(transcript.words))


def _score_visual_reads(src_path: str, clips: list[Clip], *,
                        power_mode: str | None = None,
                        lang: str | None = None,
                        cues: list[str] | None = None) -> dict[int, tuple[float, str]]:
    """Optional VLM reads for clip spans, isolated so this contract stays tested."""
    from ..providers import vlm as vlm_mod

    if not vlm_mod.available():
        return {}
    opts = get_settings().vlm_options_for(power_mode)
    return vlm_mod.score_visuals(src_path, [(c.start, c.end) for c in clips],
                                 budget=float(opts["budget"]),
                                 max_workers=int(opts["max_workers"]),
                                 n_frames=int(opts["n_frames"]),
                                 timeout=float(opts["timeout"]),
                                 lang=lang or "en", cues=cues)


STAGES = ["transcribe", "detect", "score", "reframe", "caption", "render"]
STAGE_LABELS = {
    "transcribe": "Transcribing audio",
    "detect": "Finding moments",
    "score": "Scoring & ranking",
    "reframe": "Reframing to 9:16",
    "caption": "Writing captions",
    "render": "Rendering clips",
}


class Engine:
    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._started = False
        # One lock per clip so a batch render and editor re-renders can never
        # run two ffmpeg processes against the same output file (on Windows
        # the loser errors on the locked file and marks a good clip "failed").
        # Entries are tiny and bounded by clips-per-session; never pruned.
        self._clip_locks: dict[tuple[str, str], threading.Lock] = {}
        self._clip_locks_guard = threading.Lock()
        self._pause_condition = threading.Condition()
        self._pause_requested: set[str] = set()
        # Per-stage timing anchors (project_id -> timestamp / stage index) so the
        # render window can show how long the *current* stage has been running.
        self._stage_started: dict[str, float] = {}
        self._stage_idx: dict[str, int] = {}
        self._last_progress_pct: dict[str, float] = {}
        self._last_progress_ts: dict[str, float] = {}
        self._queued_projects: set[str] = set()
        self._active_projects: set[str] = set()

    def _clip_lock(self, project_id: str, clip_id: str) -> threading.Lock:
        with self._clip_locks_guard:
            return self._clip_locks.setdefault((project_id, clip_id),
                                               threading.Lock())

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        n = max(1, get_settings().pipeline_workers)
        for i in range(n):
            threading.Thread(target=self._loop, name=f"clipforge-worker-{i}",
                             daemon=True).start()
        log.info("pipeline started with %d worker(s)", n)

    def enqueue(self, project_id: str) -> None:
        with store.mutate(project_id) as p:
            p.status = ProjectStatus.queued
            p.progress = JobProgress(stage="queued", total_stages=len(STAGES),
                                     message="Waiting to start",
                                     stages=self._stage_view(-1, 0.0))
        with self._pause_condition:
            self._queued_projects.add(project_id)
        self._q.put(project_id)

    def pause(self, project_id: str) -> None:
        with self._pause_condition:
            self._pause_requested.add(project_id)
            self._pause_condition.notify_all()
        self._mark_paused(project_id)

    def resume(self, project_id: str) -> None:
        with self._pause_condition:
            active = project_id in self._active_projects
            queued = project_id in self._queued_projects
            self._pause_requested.discard(project_id)
            self._pause_condition.notify_all()
        if active:
            with store.mutate(project_id) as p:
                if p.status == ProjectStatus.paused:
                    p.status = ProjectStatus.processing
                    p.progress.message = "Resuming..."
                    p.progress.updated_at = now()
            return
        if queued:
            with store.mutate(project_id) as p:
                if p.status == ProjectStatus.paused:
                    p.status = ProjectStatus.queued
                    p.progress.message = "Waiting to resume"
                    p.progress.updated_at = now()
            return
        p = store.get(project_id)
        if p and p.status == ProjectStatus.paused and p.source is not None:
            self.enqueue(project_id)
        elif p and p.status == ProjectStatus.paused:
            with store.mutate(project_id) as project:
                project.status = ProjectStatus.created
                project.progress.message = "Paused project reset"
                project.progress.updated_at = now()

    def resume_incomplete(self) -> int:
        """Requeue projects stranded mid-run by a restart or dead worker."""
        resumed = 0
        for summary in store.list_summaries(limit=1000):
            if summary.status not in (ProjectStatus.queued, ProjectStatus.processing):
                continue
            try:
                with store.mutate(summary.id) as p:
                    p.clips = []
                    p.events = []
                    p.montages = []
                self.enqueue(summary.id)
                resumed += 1
            except Exception:
                log.warning("could not resume project %s", summary.id, exc_info=True)
        if resumed:
            log.info("requeued %d incomplete project(s)", resumed)
        return resumed

    # -- worker loop ------------------------------------------------------
    def _loop(self) -> None:
        while True:
            project_id = self._q.get()
            with self._pause_condition:
                self._queued_projects.discard(project_id)
                self._active_projects.add(project_id)
            try:
                self._process(project_id)
            except BaseException as e:  # never let the worker die
                error_msg = _friendly_error(e)
                log.error("pipeline failed for %s: %s\n%s", project_id, error_msg,
                          traceback.format_exc())
                try:
                    with store.mutate(project_id) as p:
                        p.status = ProjectStatus.failed
                        p.error = error_msg
                        p.progress.message = f"Failed: {error_msg[:200]}"
                except Exception:
                    pass
            finally:
                with self._pause_condition:
                    self._active_projects.discard(project_id)
                self._q.task_done()

    # -- progress helpers -------------------------------------------------
    def _stage_view(self, current: int, frac: float,
                    stage_started_at: float | None = None,
                    prior: list[dict] | None = None) -> list[dict]:
        view = []
        now_ts = now()
        # Carry forward elapsed times for completed stages so the UI keeps
        # showing "how long transcribe took" after we move on to detect, etc.
        prior_elapsed: dict[str, float] = {}
        for entry in (prior or ()):
            name = entry.get("name")
            el = entry.get("elapsed_seconds")
            if name and el is not None and entry.get("status") == "done":
                prior_elapsed[name] = float(el)
        for i, name in enumerate(STAGES):
            if i < current:
                status, pct = "done", 1.0
                elapsed = prior_elapsed.get(name)
            elif i == current:
                status, pct = "active", frac
                elapsed = (now_ts - stage_started_at) if stage_started_at else None
            else:
                status, pct = "pending", 0.0
                elapsed = None
            entry = {"name": name, "label": STAGE_LABELS[name],
                     "status": status, "pct": round(pct, 3)}
            if elapsed is not None:
                entry["elapsed_seconds"] = round(elapsed, 1)
            view.append(entry)
        return view

    def _advance(self, project_id: str, stage_idx: int, message: str,
                 frac: float = 0.0) -> None:
        self._wait_if_paused(project_id)
        overall = (stage_idx + max(min(frac, 1.0), 0.0)) / len(STAGES) * 100.0
        # Throttle DB writes: only persist when progress crosses a 2% boundary
        # or at least 5s have passed since the last write. Intermediate progress
        # is tracked in-memory via _stage_idx / _stage_started.
        last_pct = self._last_progress_pct.get(project_id, -1.0)
        last_ts = self._last_progress_ts.get(project_id, 0.0)
        now_ts = now()
        if (abs(overall - last_pct) < 2.0 and now_ts - last_ts < 5.0
                and stage_idx == self._stage_idx.get(project_id, -1)):
            # Update in-memory state only — skip the DB write.
            self._stage_started[project_id] = self._stage_started.get(
                project_id, now_ts)
            self._stage_idx[project_id] = stage_idx
            return
        self._last_progress_pct[project_id] = overall
        self._last_progress_ts[project_id] = now_ts

        # Track when the current stage began so the UI can show per-stage
        # elapsed time. Reset when the stage index changes.
        prev_started = self._stage_started.get(project_id)
        prev_idx = self._stage_idx.get(project_id)
        stage_started = prev_started if prev_idx == stage_idx else now()
        self._stage_started[project_id] = stage_started
        self._stage_idx[project_id] = stage_idx
        with store.mutate(project_id) as p:
            p.status = ProjectStatus.processing
            # Anchor the ETA clock on the first advance and keep it across stages.
            started = p.progress.started_at or now()
            prior_stages = p.progress.stages or []
            p.progress = JobProgress(
                stage=STAGES[stage_idx], stage_index=stage_idx,
                total_stages=len(STAGES), message=message,
                pct=round(overall, 1),
                stages=self._stage_view(stage_idx, frac, stage_started, prior_stages),
                started_at=started,
            )

    def _paused_stage_view(self, p) -> list[dict]:
        stages = [dict(s) for s in (p.progress.stages or self._stage_view(p.progress.stage_index, 0.0))]
        for stage in stages:
            if stage.get("name") == p.progress.stage:
                stage["status"] = "paused"
                break
        return stages

    def _mark_paused(self, project_id: str) -> None:
        try:
            with store.mutate(project_id) as p:
                if p.status in (ProjectStatus.ready, ProjectStatus.failed):
                    return
                p.status = ProjectStatus.paused
                if p.progress.stage == "render":
                    p.progress.message = "Paused - current encodes finish, then rendering waits"
                else:
                    p.progress.message = "Paused - waiting to resume"
                p.progress.stages = self._paused_stage_view(p)
                p.progress.updated_at = now()
        except Exception:
            pass

    def _wait_if_paused(self, project_id: str) -> None:
        announced = False
        while True:
            p = store.get(project_id)
            if p is None:
                return
            with self._pause_condition:
                requested = project_id in self._pause_requested
            paused = bool(p and p.status == ProjectStatus.paused)
            if not requested and not paused:
                return
            if not announced:
                self._mark_paused(project_id)
                announced = True
            with self._pause_condition:
                self._pause_condition.wait(timeout=0.5)

    # -- the pipeline -----------------------------------------------------
    def _process(self, project_id: str) -> None:
        project = store.get(project_id)
        if project is None or project.source is None:
            raise RuntimeError("project or source media missing")
        self._wait_if_paused(project_id)
        settings = get_settings()
        src_path = str(settings.media_dir / project.source.path)
        info = ffmpeg.probe(src_path)

        # 1. transcribe ---------------------------------------------------
        # The wav lives in the project dir (not a TemporaryDirectory) so the
        # gameplay detector can reuse it — decoding an hour-long VOD's audio
        # twice costs minutes. Deleted as soon as detection is done.
        self._advance(project_id, 0, "Extracting audio…")
        wav = ingest.project_dir(project_id) / "audio16k.wav"
        wav_path: str | None = None
        if info.has_audio:
            ffmpeg.extract_audio_wav(src_path, wav)
            wav_path = str(wav)
            transcript = transcribe_mod.transcribe(
                wav_path, language=project.settings.language,
                progress=lambda f: self._advance(project_id, 0,
                                                 f"Transcribing… {int(f*100)}%", f),
                power_mode=project.settings.power_mode.value,
            )
        else:
            # No audio track — skip ASR entirely, go straight to synthetic.
            transcript = transcribe_mod.synthetic_transcript(
                src_path, lang=project.settings.language)

        # Pin caption words to the *exact* speech with Silero VAD: clamp each
        # word to its speech region and drop words stuck in silence, so captions
        # start/stop precisely when the speaker is talking. VAD also strips
        # Whisper's classic silence hallucinations ("thank you for watching").
        # It's now a hard dep; if it's somehow absent we warn the user that
        # captions may drift rather than fail silently.
        vad_absent_warned = False
        if wav_path and transcript.provider != "synthetic":
            try:
                from ..providers import vad as vad_mod
                speech = vad_mod.speech_intervals(wav_path)
                if speech:
                    transcript.words = vad_mod.refine_words(transcript.words, speech)
                elif not vad_mod.available():
                    vad_absent_warned = True
                    log.warning("Silero VAD not installed — captions may drift")
            except Exception as e:
                log.warning("VAD refine failed: %s", e)
            # Optional LR-ASD: attribute each word to the on-screen active speaker
            # (no-op unless the LR-ASD checkout is wired via CLIPFORGE_ASD_DIR).
            try:
                from ..providers import active_speaker as asd_mod
                if asd_mod.available():
                    transcript.words = asd_mod.attribute_speakers(
                        src_path, transcript.words)
            except Exception as e:
                log.warning("active-speaker attribution failed: %s", e)

        with store.mutate(project_id) as p:
            p.transcript = transcript
            if vad_absent_warned:
                p.add_warning(
                    "Silero VAD isn't installed, so captions may drift from the "
                    "speech and include hallucinated words in quiet sections. "
                    "Install it (pip install silero-vad) and re-process for "
                    "accurate captions.", severity="warning")

        # Pre-compute speech intervals for the full timeline so each clip
        # doesn't re-scan the transcript word list.
        _precompute_speech_intervals(transcript)

        # Decide talking vs gameplay (auto-detect unless forced).
        forced = project.settings.content_type
        if forced == ContentType.auto:
            self._advance(project_id, 1, "Analyzing footage…")
            kind, _metrics = classify.detect_content_type(src_path, info, transcript)
        else:
            kind = forced.value
        synthetic_caption_warning = (
            kind == "talking" and info.has_audio
            and transcript.provider == "synthetic")
        # Gameplay with a streamer cam: find the overlay once for the whole
        # source — it drives the split/framed layouts and reaction scoring.
        cam_rect = None
        if kind == "gameplay" and project.settings.facecam_layout != "off":
            self._advance(project_id, 1, "Looking for a facecam…")
            try:
                cam_rect = facecam_mod.detect_facecam(src_path, info.duration)
            except Exception as e:
                log.warning("facecam detection failed: %s", e)
        with store.mutate(project_id) as p:
            p.content_type = kind
            p.facecam = cam_rect
            p.warnings = []
            # Placeholder captions are a real degradation — flag as error.
            if synthetic_caption_warning:
                p.add_warning(
                    "Speech recognition didn't run, so captions are placeholder "
                    "text. Install the VC++ Redistributable (and a Whisper "
                    "model), then re-process.", severity="error",
                    code="synthetic_transcript")

        # 2. detect + 3. score (branch on content type) -------------------
        plat = project.settings.platform.value
        if kind == "gameplay":
            self._advance(project_id, 1, "Finding gameplay highlights…")
            prof = project.settings.game_profile
            gweights = feedback.learned_weights(
                feedback.score_scope("gameplay", prof), gameplay_mod.audio_weights(prof))
            detected: list = []
            detect_warnings: list[str] = []
            gcs = gameplay_mod.detect_gameplay(src_path, info, project.settings,
                                               weights=gweights, wav_path=wav_path,
                                               events_out=detected,
                                               warnings_out=detect_warnings)
            if detect_warnings:
                with store.mutate(project_id) as p:
                    for msg in detect_warnings:
                        p.add_warning(msg, severity="warn", code="detector")
            if not gcs:
                raise RuntimeError("no highlights found in this footage")
            # Learn reusable AUDIO cues from the on-screen (OCR) events: snip the
            # game sound at each banner and save it, so the cheap audio matcher
            # catches that event on future videos even with OCR off.
            if info.has_audio and project.settings.cue_learning:
                ocr_evs = [e for e in detected if getattr(e, "source", "") == "ocr"]
                if ocr_evs:
                    try:
                        from .. import cue_learning
                        learned = cue_learning.save_audio_cues_from_ocr(
                            src_path, ocr_evs, prof)
                        if learned:
                            log.info("learned %d audio cue(s) from OCR: %s",
                                     len(learned), ", ".join(learned))
                    except Exception as e:
                        log.warning("cue learning failed: %s", e)
            if cam_rect is not None:
                # Re-score with the facecam reaction folded in (cue-matched
                # clips keep their exact-sound score).
                self._advance(project_id, 2, "Reading facecam reactions…")
                gweights = gameplay_mod.with_reaction(gweights)
                for gc in gcs:
                    # Exact-event clips (cue / OCR) keep their guaranteed score —
                    # the audio scorer would zero them out (their RMS feats are 0).
                    if gc.features.get("cue", 0.0) > 0.0 or gc.features.get("ocr", 0.0) > 0.0:
                        continue
                    r = facecam_mod.reaction_energy(src_path, cam_rect,
                                                    gc.start, gc.end)
                    if r is not None:
                        gc.features["reaction"] = round(r, 4)
                    gc.score, gc.factors = gameplay_mod.score_audio(
                        gc.features, gweights)
            clips = []
            real_speech = transcript.provider != "synthetic"
            for gc in gcs:
                words = ([w for w in transcript.words if w.end > gc.start and w.t < gc.end]
                         if real_speech else [])  # never carry filler text
                features = dict(gc.features)
                features["evidence_t"] = round(float(gc.peak_t), 3)
                clips.append(Clip(
                    start=gc.start, end=gc.end, title=gc.title,
                    kind="gameplay", score=gc.score, factors=gc.factors,
                    features=features,
                    # Mute captions around matched game sounds — the announcer
                    # saying "Double Kill" isn't the streamer talking.
                    caption_mute=[[round(t - 0.4, 3), round(t + 2.6, 3)]
                                  for t in gc.cue_ts],
                    transcript_excerpt=" ".join(w.text for w in words)[:400]))
            # Snap each highlight's start to a nearby hard cut (killcam wipe,
            # replay transition) so the clip opens on a fresh shot.
            self._advance(project_id, 2, "Snapping to scene cuts…")
            for clip in clips:
                try:
                    cuts = scenes_mod.scene_cuts(
                        src_path, max(clip.start - 2.0, 0.0), clip.start + 2.0)
                    ns = scenes_mod.snap(clip.start, cuts, window=1.5)
                    if ns != clip.start and clip.end - ns >= 3.0:
                        clip.start = round(ns, 3)
                except Exception as e:
                    log.warning("scene snap failed for %s: %s", clip.id, e)
            self._advance(project_id, 2, "Ranking highlights by intensity…")
            bscope = feedback.bound_scope("gameplay", project.settings.game_profile)
        else:
            # Personalised weights from prior feedback (cold-start safe).
            weights = feedback.learned_weights(
                feedback.score_scope("talking", plat), score_mod.base_weights(project.settings))
            self._advance(project_id, 1, "Scanning transcript for moments…")
            candidates = detect_mod.detect_moments(transcript.words, project.settings,
                                                   lang=transcript.language, weights=weights)
            if not candidates:
                raise RuntimeError("no clip-worthy moments found in this video")
            clips = [Clip(start=round(c.start, 3), end=round(c.end, 3), title=c.title,
                          kind="speech",
                          transcript_excerpt=" ".join(w.text for w in c.words)[:400])
                     for c in candidates]
            self._advance(project_id, 2, "Scoring clips…")
            for clip in clips:
                words = [w for w in transcript.words
                         if w.end > clip.start and w.t < clip.end]
                clip.score, clip.factors, clip.features = score_mod.score_clip(
                    words, clip.duration, project.settings,
                    lang=transcript.language, weights=weights)
                # Reward a clean, loopable ending (rewatch signal).
                clip.score, clip.factors = score_mod.apply_replay_bonus(
                    clip.score, clip.factors, words, clip.duration,
                    lang=transcript.language)
            # Optional: a local LLM (Ollama) gives a second opinion on virality
            # (re-ranks within ±12 pts, explainable) and writes sharper titles —
            # concurrent, budgeted so a slow model can't stall the pipeline.
            if llm_mod.available():
                self._advance(project_id, 2, "AI reading virality…")
                reads = llm_mod.score_virals(
                    [c.transcript_excerpt for c in clips], lang=transcript.language)
                for i, (viral, reason) in reads.items():
                    clips[i].score, clips[i].factors = score_mod.apply_viral_boost(
                        clips[i].score, clips[i].factors, viral, reason)
                    clips[i].features["llm_viral"] = round(viral, 4)
                self._advance(project_id, 2, "Writing AI titles…")
                titles = llm_mod.suggest_titles(
                    [c.transcript_excerpt for c in clips], lang=transcript.language)
                for i, t in titles.items():
                    clips[i].title = t
            # Fallback: auto-generate titles for clips that still have none.
            if llm_mod.available():
                for clip in clips:
                    if not clip.title:
                        clip.title = llm_mod.generate_title(
                            clip.transcript_excerpt, lang=transcript.language)
            # Hook/first-3s analysis: warn the user if their opener is weak.
            if clips:
                first = clips[0]
                hook_words = [w for w in transcript.words
                              if w.end > first.start and w.t < first.end]
                h = score_mod.hook_analysis(hook_words, lang=transcript.language)
                if h["verdict"] == "weak" and h["suggestion"]:
                    with store.mutate(project_id) as p:
                        p.add_warning(
                            f"Hook: {h['suggestion']}",
                            severity="warning", code="hook")
            bscope = feedback.bound_scope("talking", plat)

        # Optional speech-emotion excitement (emotion2vec): a high-arousal
        # delivery — laughter, hype, rage — is the short-form viral driver, so
        # fold it in as an explainable factor. Cheap per-clip, no-op when absent.
        if settings.has_emotion and wav_path:
            try:
                from ..providers import emotion as emo_mod
                for clip in clips:
                    a = emo_mod.excitement(wav_path, clip.start, clip.end)
                    if a is not None:
                        clip.score, clip.factors = emo_mod.apply_excitement_bonus(
                            clip.score, clip.factors, a)
                        clip.features["excitement"] = round(a, 4)
            except Exception as e:
                log.warning("emotion scoring failed: %s", e)

        # Optional audio-event detection (PANNs): hear the *sounds* that signal a
        # highlight — cheering, laughter, applause, an explosion — and fold them
        # in as an explainable factor. Zero-shot, so it needs no per-game cue.
        from ..providers import audio_events as ae_mod
        if project.settings.use_audio_events and ae_mod.available() and wav_path:
            try:
                self._advance(project_id, 2, "Listening for crowd / hype…")
                aecfg = getattr(project.settings, "game_config", None)
                ae_pos = getattr(aecfg, "audio_prompts", None) if aecfg else None
                ae_neg = getattr(aecfg, "audio_negative_prompts", None) if aecfg else None
                for clip in clips:
                    # Skip clips born from an audio event already — they carry
                    # the CLAP signal in their baseline score, so a second
                    # apply_event_bonus would double-count the same evidence.
                    if clip.features.get("audio_event", 0.0) > 0.0:
                        continue
                    res = ae_mod.event_score(
                        wav_path, clip.start, clip.end,
                        profile=project.settings.game_profile,
                        language=project.settings.language,
                        positive_prompts=ae_pos, negative_prompts=ae_neg)
                    if res is not None:
                        hype, reason = res
                        clip.score, clip.factors = ae_mod.apply_event_bonus(
                            clip.score, clip.factors, hype, reason)
                        clip.features["audio_event"] = round(hype, 4)
            except Exception as e:
                log.warning("audio-event scoring failed: %s", e)

        wav.unlink(missing_ok=True)  # audio no longer needed past detection

        # Optional vision-language second opinion (Ollama VLM): score how each
        # clip *looks* from a few keyframes — expression, action, framing — and
        # blend it like the text read (bounded ±, explainable). No-op without a
        # local vision model.
        try:
            from ..providers import vlm as vlm_mod
            if project.settings.use_vlm and vlm_mod.available():
                self._advance(project_id, 2, "AI watching the clips…")
                gcfg = getattr(project.settings, "game_config", None)
                vlm_cues = getattr(gcfg, "vlm_visual_prompts", None) if gcfg else None
                reads = _score_visual_reads(
                    src_path, clips, power_mode=project.settings.power_mode.value,
                    lang=transcript.language, cues=vlm_cues)
                for i, (viral, reason) in reads.items():
                    clips[i].score, clips[i].factors = score_mod.apply_viral_boost(
                        clips[i].score, clips[i].factors, viral,
                        f"looks {reason}" if reason else "strong visual")
                    clips[i].features["vlm_viral"] = round(viral, 4)
                clips.sort(key=lambda c: c.score, reverse=True)
        except Exception as e:
            log.warning("VLM scoring failed: %s", e)

        # Learned boundary correction — nudge toward where you actually trim.
        cs, ce = feedback.boundary_correction(bscope)
        for clip in clips:
            clip.raw_start, clip.raw_end = clip.start, clip.end
            if cs or ce:
                ns = max(0.0, clip.start + cs)
                ne = min(info.duration, clip.end + ce)
                if ne - ns >= max(project.settings.min_len * 0.5, 1.0):
                    clip.start, clip.end = round(ns, 3), round(ne, 3)

        # ASR word timestamps can drift past EOF; a span beyond the file renders
        # empty or fails, so clamp every clip to the real media duration.
        if info.duration > 0:
            for clip in clips:
                # Clamp both ends — a start past EOF would render an empty span
                # before the duration filter below could drop it.
                clip.start = round(max(0.0, min(clip.start, info.duration)), 3)
                clip.end = round(min(clip.end, info.duration), 3)
            clips = [c for c in clips if c.end - c.start >= 1.0]
            if not clips:
                raise RuntimeError("no clip-worthy moments found in this video")

        if kind == "gameplay":
            gameplay_mod.apply_evidence_caps(clips)
            gameplay_mod.apply_title_fallbacks(clips)

        clips.sort(key=lambda c: c.score, reverse=True)
        # B-roll candidate selection: find strong visual moments (scene cuts and
        # high-motion regions) that could serve as PiP cutaways during static
        # talking spans. The actual render-time composite is future work — the
        # selection runs here so the data is available on each clip as
        # ``clip.broll_overlay``.
        if kind == "talking" and settings.has_scenedetect:
            try:
                from . import broll as broll_mod
                all_cuts = scenes_mod.scene_cuts(src_path, 0.0, info.duration)
                if all_cuts:
                    cands = list(broll_mod.candidates_from_cuts(
                        all_cuts, window=1.5, clip_end=info.duration))
                    for clip in clips:
                        # Pick the B-roll window closest to the middle of the
                        # clip but at least 2s in (viewer has settled after the
                        # hook). Skip clips shorter than 5s.
                        dur = clip.end - clip.start
                        if dur < 5.0 or not cands:
                            clip.broll_overlay = None
                            continue
                        mid = clip.start + dur / 2
                        # Find candidates that actually fall within this clip.
                        valid = [c for c in cands
                                 if c.start >= clip.start and c.end <= clip.end]
                        if not valid:
                            clip.broll_overlay = None
                            continue
                        best = min(valid, key=lambda c: abs(c.start - mid))
                        clip.broll_overlay = {
                                "source_t": best.start,
                                "start_rel": round(best.start - clip.start, 2),
                                "duration": round(best.end - best.start, 2),
                            }
            except Exception as e:
                log.debug("b-roll selection skipped: %s", e)
        with store.mutate(project_id) as p:
            p.clips = clips
            if kind == "gameplay":
                p.events = gameplay_mod.accepted_events_for_clips(detected, clips)

        # 4. reframe ------------------------------------------------------
        out_w, out_h = project.settings.dims()
        # Pre-compute face tracks for the entire source in one ffmpeg pass,
        # instead of each clip decoding its own segment separately.
        if kind != "gameplay" and info.has_video and info.duration > 0:
            reframe_mod.precompute_face_tracks(src_path, info.duration)
        for i, clip in enumerate(clips):
            if kind == "gameplay":
                self._advance(project_id, 3, f"Framing clip {i+1}/{len(clips)}…",
                              i / max(len(clips), 1))
                clip.reframe = self._gameplay_reframe(
                    src_path, clip, cam_rect, project.settings,
                    vertical=out_h > out_w)
            else:
                self._advance(project_id, 3, f"Tracking speaker {i+1}/{len(clips)}…",
                              i / max(len(clips), 1))
                clip.reframe = reframe_mod.compute_reframe(
                    src_path, clip.start, clip.end, info.aspect,
                    speech=_speech_intervals(transcript, clip.start, clip.end))
        if kind == "gameplay":
            self._advance(project_id, 3, "Framing clips…", 1.0)

        # 5. caption (+ tighten + hashtags) --------------------------------
        self._advance(project_id, 4, "Building captions…")
        style_id = project.settings.default_style_id
        # A synthetic transcript means no real speech was recognized. For talking
        # content we keep it (and warn loudly); for gameplay, burning filler text
        # over kills/goals would be pure noise — leave those clips caption-free.
        skip_captions = kind == "gameplay" and transcript.provider == "synthetic"
        do_tighten = (project.settings.tighten and kind == "talking"
                      and transcript.provider != "synthetic")
        for clip in clips:
            # Speakers present in the clip (for the editor's per-speaker toggles)
            # and the current keep-set (None = all) for caption building.
            if not skip_captions and transcript.provider != "synthetic":
                clip.speakers = captionize.speakers_in(transcript, clip.start, clip.end)
            spk = set(clip.caption_speakers) if clip.caption_speakers is not None else None
            if skip_captions:
                clip.captions.style_id = style_id  # keep the empty CaptionSet
            elif do_tighten:
                segs = captionize.compute_tight_segments(transcript, clip.start, clip.end)
                if len(segs) >= 2:
                    clip.segments = [[round(a, 3), round(b, 3)] for a, b in segs]
                    clip.tightened_duration = round(sum(b - a for a, b in segs), 3)
                    clip.captions = captionize.build_tight_caption_set(
                        transcript, segs, style_id, speakers=spk)
                    # Remap speaker-tracking keyframes onto the tightened timeline.
                    for kf in clip.reframe.keyframes:
                        kf.t = round(captionize.map_to_tight(clip.start + kf.t, segs), 3)
                else:
                    clip.captions = captionize.build_caption_set(
                        transcript, clip.start, clip.end, style_id, speakers=spk)
            else:
                clip.captions = captionize.build_caption_set(
                    transcript, clip.start, clip.end, style_id,
                    exclude=[(a, b) for a, b in clip.caption_mute] or None,
                    speakers=spk)
                if kind == "gameplay":
                    # Strip stock announcer/agent lines the ASR picked up.
                    clip.captions.words = captionize.remove_phrases(
                        clip.captions.words,
                        captionize.game_noise(project.settings.game_profile))
            clip.hashtags = hashtags_mod.suggest_hashtags(
                clip.transcript_excerpt or clip.title,
                content_type=kind, platform=project.settings.platform.value,
                game=project.settings.game_profile if kind == "gameplay" else None)
        with store.mutate(project_id) as p:
            # The user may have rated a clip while this stage ran; don't wipe
            # the marker with our (older) local copies.
            prev = {c.id: c.feedback for c in p.clips}
            for clip in clips:
                clip.feedback = prev.get(clip.id, clip.feedback)
            p.clips = clips

        # Optional Demucs vocal isolation: separate the voice once and render
        # from a denoised copy (video stream copied, untouched) so every clip
        # gets studio-clean speech. Falls back to the original on any failure.
        render_src = src_path
        if project.settings.denoise and settings.has_demucs and info.has_audio:
            self._advance(project_id, 5, "Isolating voice (Demucs)…")
            try:
                from ..providers import separate as sep_mod
                dst_path = ingest.project_dir(project_id) / "source.denoised.mp4"
                dst = str(dst_path)
                cleaned = dst if dst_path.exists() else sep_mod.denoise_source(src_path, dst)
                if cleaned:
                    render_src = cleaned
            except Exception as e:
                log.warning("denoise failed: %s", e)

        # 6. render (parallel per clip) ----------------------------------
        out_w, out_h = project.settings.dims()
        self._advance(project_id, 5, "Rendering clips…")
        self._render_all(project_id, clips, render_src, info, out_w, out_h,
                         project.settings.burn_captions, project.settings.motion,
                         project.settings.power_mode.value)

        self._finish_render_progress(project_id)
        log.info("project %s complete", project_id)

    # First-person games keep the action at the crosshair — never chase motion.
    _CENTERED_PROFILES = {"valorant", "cs2", "cs", "horror"}

    def _gameplay_reframe(self, src_path: str, clip: Clip, cam_rect,
                          settings, *, vertical: bool) -> Reframe:
        """Layout + crop for one gameplay clip.

        With a facecam: stacked split (or PiP if forced). The gameplay pane is
        biased toward where the motion actually is — except in first-person
        games, where the crosshair (center) is the action.
        """
        cx = 0.5
        prof = (settings.game_profile or "generic").lower()
        if prof not in self._CENTERED_PROFILES:
            try:
                c = facecam_mod.action_center(src_path, clip.start, clip.end,
                                              cam_rect)
                if c is not None:
                    cx = round(c, 4)
            except Exception as e:
                log.warning("action centroid failed for %s: %s", clip.id, e)
        layout = LayoutType.center
        if vertical and cam_rect is not None and settings.facecam_layout != "off":
            layout = (LayoutType.framed if settings.facecam_layout == "framed"
                      else LayoutType.split)
        return Reframe(layout=layout, tracked=False, facecam=cam_rect,
                       keyframes=[ReframeKeyframe(t=0.0, cx=cx)])

    def _render_all(self, project_id: str, clips: list[Clip], src_path: str,
                    info: MediaInfo, out_w: int, out_h: int,
                    burn_captions: bool = True, motion: str = "none",
                    power_mode: str | None = None) -> None:
        settings = get_settings()
        done = 0
        total = len(clips)
        n = max(1, min(settings.render_workers_for(power_mode), total))
        clip_iter = iter(clips)
        # Load the project once for per-project settings (e.g. background_music).
        proj = store.get(project_id)
        bgm = proj.settings.background_music if proj else ""
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = {}

            def submit_next() -> bool:
                self._wait_if_paused(project_id)
                try:
                    clip = next(clip_iter)
                except StopIteration:
                    return False
                futs[ex.submit(self._render_one, project_id, clip, src_path, info,
                               out_w, out_h, burn_captions, motion, bgm)] = clip
                return True

            for _ in range(n):
                if not submit_next():
                    break
            while futs:
                completed, _pending = wait(futs, return_when=FIRST_COMPLETED)
                for fut in completed:
                    futs.pop(fut, None)
                    done += 1
                    fut.result()  # surface unexpected (non-per-clip) errors
                    self._advance(project_id, 5,
                                  f"Rendered {done}/{total} clips", done / total)
                while len(futs) < n and submit_next():
                    pass

    def _render_one(self, project_id: str, clip: Clip, src_path: str,
                    info: MediaInfo, out_w: int, out_h: int,
                    burn_captions: bool = True, motion: str = "none",
                    background_music: str = "") -> None:
        # A clip may carry its own output aspect (editor override); it wins
        # over the project default passed in.
        dims = ASPECTS.get(clip.aspect or "")
        if dims:
            out_w, out_h = dims
        settings = get_settings()
        pdir = ingest.project_dir(project_id)
        out = pdir / "clips" / f"{clip.id}.mp4"
        thumb = pdir / "clips" / f"{clip.id}.jpg"
        style = get_style(clip.captions.style_id)
        try:
            with store.mutate(project_id) as p:
                c = p.clip(clip.id)
                if c and c.status != ClipStatus.ready:
                    c.status = ClipStatus.rendering
            with self._clip_lock(project_id, clip.id):
                self._render_one_locked(project_id, clip, src_path, info,
                                        out_w, out_h, burn_captions, motion,
                                        out, thumb, style, settings,
                                        background_music)
        except Exception as e:  # one bad clip shouldn't sink the batch
            log.error("clip %s render failed: %s", clip.id, e)
            with store.mutate(project_id) as p:
                c = p.clip(clip.id)
                if c:
                    c.status = ClipStatus.failed
                    c.error = str(e)

    def _render_one_locked(self, project_id, clip, src_path, info, out_w, out_h,
                           burn_captions, motion, out, thumb, style, settings,
                           background_music: str = "") -> None:
        render_mod.render_clip(clip, src_path, info, style, out, thumb,
                               out_w=out_w, out_h=out_h,
                               burn_captions=burn_captions, motion=motion,
                               background_music=background_music)
        with store.mutate(project_id) as p:
            c = p.clip(clip.id)
            if c:
                c.status = ClipStatus.ready
                c.export_url = _media_url(out)
                c.thumb_url = _media_url(thumb)
                c.error = None

    def _finish_render_progress(self, project_id: str,
                                selected_ids: set[str] | None = None) -> None:
        with store.mutate(project_id) as p:
            ready = sum(1 for c in p.clips if c.status == ClipStatus.ready)
            failed = sum(1 for c in p.clips if c.status == ClipStatus.failed)
            if selected_ids is not None:
                selected_ready = sum(1 for c in p.clips
                                     if c.id in selected_ids and c.status == ClipStatus.ready)
                if selected_ready == 0:
                    for c in p.clips:
                        if c.id in selected_ids and c.status != ClipStatus.failed:
                            c.status = ClipStatus.failed
                            c.error = "Selected re-render did not produce an output."
                    p.status = ProjectStatus.ready if ready else ProjectStatus.failed
                    p.progress = JobProgress(
                        stage="render", stage_index=5, total_stages=len(STAGES),
                        message="Selected re-render failed for every selected clip",
                        pct=100.0, stages=self._stage_view(5, 1.0))
                    return
            if ready == 0:
                p.status = ProjectStatus.failed
                p.error = "Rendering failed for every clip."
                p.progress = JobProgress(
                    stage="render", stage_index=5, total_stages=len(STAGES),
                    message="Rendering failed for every clip",
                    pct=100.0, stages=self._stage_view(5, 1.0))
                return
            if failed:
                p.add_warning(f"{failed} clip(s) failed to render.",
                              severity="warn", code="render_failed")
            p.status = ProjectStatus.ready
            p.error = None
            p.progress = JobProgress(
                stage="done", stage_index=len(STAGES), total_stages=len(STAGES),
                message=f"Done - {ready} clips ready", pct=100.0,
                stages=self._stage_view(len(STAGES), 1.0))

    # -- single-clip re-render (editor edits) -----------------------------
    def rerender_clip(self, project_id: str, clip_id: str) -> None:
        # Runs on a bare thread spawned by the API: any exception here would
        # otherwise vanish and leave the clip stuck in "rendering" forever, so
        # everything before _render_one (which has its own catch) is guarded.
        try:
            project = store.get(project_id)
            if not project or not project.source:
                raise RuntimeError("project or source media missing")
            clip = project.clip(clip_id)
            if not clip:
                raise RuntimeError("clip not found")
            src_path = self._render_source(project_id, project)
            info = ffmpeg.probe(src_path)
            # _render_one resolves a per-clip aspect override on its own.
            out_w, out_h = project.settings.dims()
            burn = project.settings.burn_captions
            with store.mutate(project_id) as p:
                c = p.clip(clip_id)
                if c:
                    c.status = ClipStatus.rendering
        except Exception as e:
            log.error("re-render setup failed for clip %s: %s", clip_id, e)
            self._mark_clip_failed(project_id, clip_id, e)
            return
        self._render_one(project_id, clip, src_path, info, out_w, out_h, burn,
                         project.settings.motion)

    def rerender_all(self, project_id: str) -> None:
        """Re-render every clip with the current settings (e.g. after a
        format change) — transcription, detection, scoring, and captions
        stay exactly as they are."""
        try:
            project = store.get(project_id)
            if not project or not project.source:
                raise RuntimeError("project or source media missing")
            src_path = self._render_source(project_id, project)
            info = ffmpeg.probe(src_path)
            out_w, out_h = project.settings.dims()
            self._render_all(project_id, project.clips, src_path, info,
                             out_w, out_h, project.settings.burn_captions,
                             project.settings.motion,
                             project.settings.power_mode.value)
            self._finish_render_progress(project_id)
        except Exception as e:
            log.error("re-render all failed for %s: %s", project_id, e)
            try:
                with store.mutate(project_id) as p:
                    p.status = ProjectStatus.ready  # clips keep their old files
                    p.progress.message = f"Format change failed: {e}"
            except Exception:
                pass  # project deleted underneath us

    def rerender_clips(self, project_id: str, clip_ids: list[str]) -> None:
        """Re-render selected clips with current project render settings."""
        try:
            project = store.get(project_id)
            if not project or not project.source:
                raise RuntimeError("project or source media missing")
            wanted = set(clip_ids)
            clips = [c for c in project.clips if c.id in wanted]
            if not clips:
                raise RuntimeError("no selected clips found")
            src_path = self._render_source(project_id, project)
            info = ffmpeg.probe(src_path)
            out_w, out_h = project.settings.dims()
            with store.mutate(project_id) as p:
                for c in p.clips:
                    if c.id in wanted:
                        c.status = ClipStatus.rendering
                        c.error = None
            self._render_all(project_id, clips, src_path, info,
                             out_w, out_h, project.settings.burn_captions,
                             project.settings.motion,
                             project.settings.power_mode.value)
            self._finish_render_progress(project_id, selected_ids=wanted)
        except Exception as e:
            log.error("selected re-render failed for %s: %s", project_id, e)
            for cid in clip_ids:
                self._mark_clip_failed(project_id, cid, e)
            try:
                with store.mutate(project_id) as p:
                    p.status = ProjectStatus.ready
                    p.progress.message = f"Selected re-render failed: {e}"
            except Exception:
                pass  # project deleted underneath us

    def _render_source(self, project_id: str, project) -> str:
        """The video to render clips from: a Demucs-denoised copy when one was
        produced for this project, else the original source."""
        denoised = ingest.project_dir(project_id) / "source.denoised.mp4"
        if project.settings.denoise and denoised.exists():
            return str(denoised)
        return str(get_settings().media_dir / project.source.path)

    def _mark_clip_failed(self, project_id: str, clip_id: str, err: Exception) -> None:
        try:
            with store.mutate(project_id) as p:
                c = p.clip(clip_id)
                if c:
                    c.status = ClipStatus.failed
                    c.error = str(err)
        except Exception:
            pass  # project deleted underneath us — nothing left to mark

    # -- montage build (off the request path) -----------------------------
    def build_montage(self, project_id: str, montage_id: str) -> None:
        # Same bare-thread rule as rerender_clip: every failure must land on
        # the montage record, or the UI shows "Rendering…" forever.
        try:
            project = store.get(project_id)
            if not project:
                raise RuntimeError("project not found")
            mtg = project.montage(montage_id)
            if not mtg:
                raise RuntimeError("montage not found")
            settings = get_settings()
            mdir = ingest.project_dir(project_id) / "montages"
            mdir.mkdir(parents=True, exist_ok=True)
            paths = []
            for cid in mtg.clip_ids:
                c = project.clip(cid)
                if c and c.export_url:
                    paths.append(settings.media_dir / _strip_media_prefix(c.export_url))
            out = mdir / f"{montage_id}.mp4"
            thumb = mdir / f"{montage_id}.jpg"
            dur = montage_mod.build_montage_video(paths, out, thumb)
            with store.mutate(project_id) as p:
                m = p.montage(montage_id)
                if m:
                    m.status = ClipStatus.ready
                    m.export_url = _media_url(out)
                    m.thumb_url = _media_url(thumb)
                    m.duration = round(dur, 2)
                    m.error = None
        except Exception as e:
            log.error("montage %s failed: %s", montage_id, e)
            try:
                with store.mutate(project_id) as p:
                    m = p.montage(montage_id)
                    if m:
                        m.status = ClipStatus.failed
                        m.error = str(e)
            except Exception:
                pass  # project deleted underneath us


engine = Engine()
