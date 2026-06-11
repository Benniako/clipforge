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
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import feedback, store
from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import (ASPECTS, Clip, ClipStatus, ContentType, JobProgress,
                      LayoutType, ProjectStatus, Reframe, ReframeKeyframe)
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
        self._q.put(project_id)

    # -- worker loop ------------------------------------------------------
    def _loop(self) -> None:
        while True:
            project_id = self._q.get()
            try:
                self._process(project_id)
            except Exception as e:  # never let the worker die
                log.error("pipeline failed for %s: %s\n%s", project_id, e,
                          traceback.format_exc())
                try:
                    with store.mutate(project_id) as p:
                        p.status = ProjectStatus.failed
                        p.error = str(e)
                        p.progress.message = f"Failed: {e}"
                except Exception:
                    pass
            finally:
                self._q.task_done()

    # -- progress helpers -------------------------------------------------
    def _stage_view(self, current: int, frac: float) -> list[dict]:
        view = []
        for i, name in enumerate(STAGES):
            if i < current:
                status, pct = "done", 1.0
            elif i == current:
                status, pct = "active", frac
            else:
                status, pct = "pending", 0.0
            view.append({"name": name, "label": STAGE_LABELS[name],
                         "status": status, "pct": round(pct, 3)})
        return view

    def _advance(self, project_id: str, stage_idx: int, message: str,
                 frac: float = 0.0) -> None:
        overall = (stage_idx + max(min(frac, 1.0), 0.0)) / len(STAGES) * 100.0
        with store.mutate(project_id) as p:
            p.status = ProjectStatus.processing
            p.progress = JobProgress(
                stage=STAGES[stage_idx], stage_index=stage_idx,
                total_stages=len(STAGES), message=message,
                pct=round(overall, 1), stages=self._stage_view(stage_idx, frac),
            )

    # -- the pipeline -----------------------------------------------------
    def _process(self, project_id: str) -> None:
        project = store.get(project_id)
        if project is None or project.source is None:
            raise RuntimeError("project or source media missing")
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
            )
        else:
            # No audio track — skip ASR entirely, go straight to synthetic.
            transcript = transcribe_mod.synthetic_transcript(
                src_path, lang=project.settings.language)
        with store.mutate(project_id) as p:
            p.transcript = transcript

        # Decide talking vs gameplay (auto-detect unless forced).
        forced = project.settings.content_type
        if forced == ContentType.auto:
            self._advance(project_id, 1, "Analyzing footage…")
            kind, _metrics = classify.detect_content_type(src_path, info, transcript)
        else:
            kind = forced.value
        warnings: list[str] = []
        # Only a problem for talking content — gameplay legitimately has no speech.
        if kind == "talking" and info.has_audio and transcript.provider == "synthetic":
            warnings.append(
                "Speech recognition didn't run, so captions are placeholder text. "
                "Install the VC++ Redistributable (and a Whisper model), then re-process.")
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
            p.warnings = warnings

        # 2. detect + 3. score (branch on content type) -------------------
        plat = project.settings.platform.value
        if kind == "gameplay":
            self._advance(project_id, 1, "Finding gameplay highlights…")
            prof = project.settings.game_profile
            gweights = feedback.learned_weights(
                feedback.score_scope("gameplay", prof), gameplay_mod.audio_weights(prof))
            gcs = gameplay_mod.detect_gameplay(src_path, info, project.settings,
                                               weights=gweights, wav_path=wav_path)
            if not gcs:
                raise RuntimeError("no highlights found in this footage")
            if cam_rect is not None:
                # Re-score with the facecam reaction folded in (cue-matched
                # clips keep their exact-sound score).
                self._advance(project_id, 2, "Reading facecam reactions…")
                gweights = gameplay_mod.with_reaction(gweights)
                for gc in gcs:
                    if gc.features.get("cue", 0.0) > 0.0:
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
                clips.append(Clip(start=gc.start, end=gc.end, title=gc.title,
                                  kind="gameplay", score=gc.score, factors=gc.factors,
                                  features=gc.features,
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
            # Optional: sharper titles from a local LLM (Ollama) — concurrent,
            # with an overall budget so a slow model can't stall the pipeline.
            if llm_mod.available():
                self._advance(project_id, 2, "Writing AI titles…")
                titles = llm_mod.suggest_titles(
                    [c.transcript_excerpt for c in clips], lang=transcript.language)
                for i, t in titles.items():
                    clips[i].title = t
            bscope = feedback.bound_scope("talking", plat)

        wav.unlink(missing_ok=True)  # audio no longer needed past detection

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
                clip.end = round(min(clip.end, info.duration), 3)
            clips = [c for c in clips if c.end - c.start >= 1.0]
            if not clips:
                raise RuntimeError("no clip-worthy moments found in this video")

        clips.sort(key=lambda c: c.score, reverse=True)
        with store.mutate(project_id) as p:
            p.clips = clips

        # 4. reframe ------------------------------------------------------
        out_w, out_h = project.settings.dims()
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
            if skip_captions:
                clip.captions.style_id = style_id  # keep the empty CaptionSet
            elif do_tighten:
                segs = captionize.compute_tight_segments(transcript, clip.start, clip.end)
                if len(segs) >= 2:
                    clip.segments = [[round(a, 3), round(b, 3)] for a, b in segs]
                    clip.tightened_duration = round(sum(b - a for a, b in segs), 3)
                    clip.captions = captionize.build_tight_caption_set(
                        transcript, segs, style_id)
                    # Remap speaker-tracking keyframes onto the tightened timeline.
                    for kf in clip.reframe.keyframes:
                        kf.t = round(captionize.map_to_tight(clip.start + kf.t, segs), 3)
                else:
                    clip.captions = captionize.build_caption_set(
                        transcript, clip.start, clip.end, style_id)
            else:
                clip.captions = captionize.build_caption_set(
                    transcript, clip.start, clip.end, style_id)
            clip.hashtags = hashtags_mod.suggest_hashtags(
                clip.transcript_excerpt or clip.title,
                content_type=kind, platform=project.settings.platform.value)
        with store.mutate(project_id) as p:
            # The user may have rated a clip while this stage ran; don't wipe
            # the marker with our (older) local copies.
            prev = {c.id: c.feedback for c in p.clips}
            for clip in clips:
                clip.feedback = prev.get(clip.id, clip.feedback)
            p.clips = clips

        # 6. render (parallel per clip) ----------------------------------
        out_w, out_h = project.settings.dims()
        self._advance(project_id, 5, "Rendering clips…")
        self._render_all(project_id, clips, src_path, info, out_w, out_h,
                         project.settings.burn_captions, project.settings.motion)

        with store.mutate(project_id) as p:
            ready = sum(1 for c in p.clips if c.status == ClipStatus.ready)
            p.status = ProjectStatus.ready
            p.progress = JobProgress(
                stage="done", stage_index=len(STAGES), total_stages=len(STAGES),
                message=f"Done — {ready} clips ready", pct=100.0,
                stages=self._stage_view(len(STAGES), 1.0),
            )
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
                    burn_captions: bool = True, motion: str = "none") -> None:
        settings = get_settings()
        done = 0
        total = len(clips)
        n = max(1, min(settings.render_workers, total))
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = {ex.submit(self._render_one, project_id, c, src_path, info,
                              out_w, out_h, burn_captions, motion): c for c in clips}
            for fut in as_completed(futs):
                done += 1
                fut.result()  # surface unexpected (non-per-clip) errors
                self._advance(project_id, 5,
                              f"Rendered {done}/{total} clips", done / total)

    def _render_one(self, project_id: str, clip: Clip, src_path: str,
                    info: MediaInfo, out_w: int, out_h: int,
                    burn_captions: bool = True, motion: str = "none") -> None:
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
            with self._clip_lock(project_id, clip.id):
                self._render_one_locked(project_id, clip, src_path, info,
                                        out_w, out_h, burn_captions, motion,
                                        out, thumb, style, settings)
        except Exception as e:  # one bad clip shouldn't sink the batch
            log.error("clip %s render failed: %s", clip.id, e)
            with store.mutate(project_id) as p:
                c = p.clip(clip.id)
                if c:
                    c.status = ClipStatus.failed
                    c.error = str(e)

    def _render_one_locked(self, project_id, clip, src_path, info, out_w, out_h,
                           burn_captions, motion, out, thumb, style, settings) -> None:
        render_mod.render_clip(clip, src_path, info, style, out, thumb,
                               out_w=out_w, out_h=out_h,
                               burn_captions=burn_captions, motion=motion)
        rel = out.relative_to(settings.media_dir)
        trel = thumb.relative_to(settings.media_dir)
        with store.mutate(project_id) as p:
            c = p.clip(clip.id)
            if c:
                c.status = ClipStatus.ready
                c.export_url = f"/media/{rel}"
                c.thumb_url = f"/media/{trel}"
                c.error = None

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
            src_path = str(get_settings().media_dir / project.source.path)
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
            src_path = str(get_settings().media_dir / project.source.path)
            info = ffmpeg.probe(src_path)
            out_w, out_h = project.settings.dims()
            self._render_all(project_id, project.clips, src_path, info,
                             out_w, out_h, project.settings.burn_captions,
                             project.settings.motion)
            with store.mutate(project_id) as p:
                ready = sum(1 for c in p.clips if c.status == ClipStatus.ready)
                p.status = ProjectStatus.ready
                p.progress = JobProgress(
                    stage="done", stage_index=len(STAGES), total_stages=len(STAGES),
                    message=f"Done — {ready} clips ready", pct=100.0,
                    stages=self._stage_view(len(STAGES), 1.0))
        except Exception as e:
            log.error("re-render all failed for %s: %s", project_id, e)
            try:
                with store.mutate(project_id) as p:
                    p.status = ProjectStatus.ready  # clips keep their old files
                    p.progress.message = f"Format change failed: {e}"
            except Exception:
                pass  # project deleted underneath us

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
                    paths.append(settings.media_dir / c.export_url.removeprefix("/media/"))
            out = mdir / f"{montage_id}.mp4"
            thumb = mdir / f"{montage_id}.jpg"
            dur = montage_mod.build_montage_video(paths, out, thumb)
            rel = out.relative_to(settings.media_dir)
            trel = thumb.relative_to(settings.media_dir)
            with store.mutate(project_id) as p:
                m = p.montage(montage_id)
                if m:
                    m.status = ClipStatus.ready
                    m.export_url = f"/media/{rel}"
                    m.thumb_url = f"/media/{trel}"
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
