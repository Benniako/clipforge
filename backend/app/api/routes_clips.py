"""Clip editing: trim, restyle captions, fix caption text, override the crop.

Edits mutate the stored clip, then trigger a background re-render of *only* that
clip (PRD: edits re-render what changed, never the whole batch, and never touch
the original source). The endpoint returns immediately with the clip marked
``rendering`` so the editor can show progress.
"""
from __future__ import annotations

import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from .. import feedback, store
from ..config import get_settings
from ..models import (ASPECTS, CaptionWord, Clip, ClipStatus, LayoutType,
                      Montage, Project, Rect, Reframe, ReframeKeyframe)
from ..pipeline import captionize
from ..pipeline.captions import build_srt
from ..pipeline import montage as montage_mod
from ..pipeline import reframe as reframe_mod
from ..pipeline.orchestrator import engine
from ..providers import score as score_mod
from ..styles import all_styles, get_style

router = APIRouter(prefix="/api", tags=["clips"])


def _scopes(project, clip) -> tuple[str, str]:
    """(score_scope, boundary_scope) for this clip's content type.

    Talking learns per platform; gameplay learns per game profile.
    """
    ct = "gameplay" if clip.kind == "gameplay" else "talking"
    key = project.settings.game_profile if ct == "gameplay" else project.settings.platform.value
    return feedback.score_scope(ct, key), feedback.bound_scope(ct, key)


class ClipEdit(BaseModel):
    title: str | None = None
    start: float | None = None
    end: float | None = None
    style_id: str | None = None
    caption_words: list[CaptionWord] | None = None  # full manual replacement
    # Speakers to keep in captions (diarized ids); null resets to all speakers.
    # Distinguished from "not sent" via the request's model_fields_set.
    caption_speakers: list[int] | None = None
    reframe_cx: float | None = None                 # static manual crop centre [0,1]
    layout: str | None = None                       # "center"|"split"|"framed"
    facecam: Rect | None = None                     # facecam region override
    # Per-clip output aspect: an ASPECTS key, or "" to return to the
    # project default.
    aspect: str | None = None


@router.get("/styles")
def list_styles():
    return [s.model_dump() for s in all_styles()]


@router.patch("/projects/{project_id}/clips/{clip_id}", response_model=Clip)
def edit_clip(project_id: str, clip_id: str, edit: ClipEdit) -> Clip:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")

    src_dur = project.source.duration if project.source else clip.end
    span_changed = False

    if edit.title is not None:
        clip.title = edit.title.strip()[:120]

    if edit.start is not None or edit.end is not None:
        new_start = clip.start if edit.start is None else max(0.0, edit.start)
        new_end = clip.end if edit.end is None else min(src_dur, edit.end)
        if new_end - new_start < 1.0:
            raise HTTPException(400, "clip must be at least 1 second")
        span_changed = new_start != clip.start or new_end != clip.end
        clip.start, clip.end = round(new_start, 3), round(new_end, 3)
        if span_changed:
            # Jump-cut segments were computed for the old span; render plain.
            clip.segments = []
            clip.tightened_duration = None

    if edit.style_id is not None:
        clip.captions.style_id = get_style(edit.style_id).id

    # Per-speaker caption toggle. "caption_speakers" not in the request leaves
    # the clip's keep-set untouched; null resets it to "all speakers".
    speakers_sent = "caption_speakers" in edit.model_fields_set
    if speakers_sent:
        clip.caption_speakers = edit.caption_speakers
    spk = set(clip.caption_speakers) if clip.caption_speakers is not None else None

    def _rebuild_captions() -> None:
        """Re-derive the caption set for the current span honouring the kind,
        any tightening, cue exclusions and the per-speaker keep-set."""
        tr = project.transcript
        sid = clip.captions.style_id
        if clip.kind == "gameplay":
            if tr.provider == "synthetic":
                clip.captions.words = []
                return
            clip.captions = captionize.build_caption_set(
                tr, clip.start, clip.end, sid,
                exclude=[(a, b) for a, b in clip.caption_mute] or None, speakers=spk)
            clip.captions.words = captionize.remove_phrases(
                clip.captions.words, captionize.game_noise(project.settings.game_profile))
        elif clip.segments:
            clip.captions = captionize.build_tight_caption_set(
                tr, [(a, b) for a, b in clip.segments], sid, speakers=spk)
        else:
            clip.captions = captionize.build_caption_set(
                tr, clip.start, clip.end, sid, speakers=spk)

    # Re-derive captions/score/reframe for a new span, unless the user supplied
    # explicit overrides for those pieces. Gameplay clips keep their audio-based
    # score/features (the speech scorer would corrupt them) and never get filler
    # captions from a synthetic transcript.
    if span_changed and project.transcript and edit.caption_words is None:
        _rebuild_captions()
        if clip.kind != "gameplay" and project.transcript.provider != "synthetic":
            clip.speakers = captionize.speakers_in(
                project.transcript, clip.start, clip.end)
        if clip.kind != "gameplay":
            words = [w for w in project.transcript.words
                     if w.end > clip.start and w.t < clip.end]
            clip.score, clip.factors, clip.features = score_mod.score_clip(
                words, clip.duration, project.settings,
                lang=project.transcript.language)
    elif speakers_sent and project.transcript and edit.caption_words is None:
        # Only the speaker keep-set changed — re-filter captions on the same span.
        _rebuild_captions()

    if edit.caption_words is not None:
        clip.captions.words = edit.caption_words

    if edit.reframe_cx is not None:
        cx = max(0.0, min(1.0, edit.reframe_cx))
        # A manual crop centre keeps the clip's facecam layout intact.
        clip.reframe = Reframe(layout=clip.reframe.layout,
                               keyframes=[ReframeKeyframe(t=0.0, cx=cx)],
                               tracked=False, overridden=True,
                               cx_overridden=True,
                               facecam=clip.reframe.facecam)
    elif span_changed and project.source:
        if clip.kind == "gameplay":
            # Gameplay stays action-centered; face tracking is for talking clips.
            clip.reframe = Reframe(layout=clip.reframe.layout, tracked=False,
                                   keyframes=[ReframeKeyframe(t=0.0, cx=0.5)],
                                   facecam=clip.reframe.facecam)
        else:
            from ..pipeline.orchestrator import _speech_intervals
            clip.reframe = reframe_mod.compute_reframe(
                str(get_settings().media_dir / project.source.path),
                clip.start, clip.end,
                (project.source.width / project.source.height)
                if project.source.height else 1.78,
                speech=_speech_intervals(project.transcript, clip.start, clip.end))

    if edit.facecam is not None:
        clip.reframe.facecam = edit.facecam.clamped()
        clip.reframe.overridden = True

    if edit.aspect is not None:
        clip.aspect = edit.aspect if edit.aspect in ASPECTS else None

    if edit.layout is not None:
        try:
            lay = LayoutType(edit.layout)
        except ValueError:
            raise HTTPException(400, f"unknown layout '{edit.layout}'")
        if lay in (LayoutType.split, LayoutType.framed):
            cam = clip.reframe.facecam or project.facecam
            if cam is None:
                raise HTTPException(
                    400, "no facecam region known — set one in the editor first")
            clip.reframe.facecam = cam
        clip.reframe.layout = lay
        clip.reframe.overridden = True

    # Learn from the edit: a trim tells us where the moment really was, and
    # refining a clip implies you're keeping it.
    score_sc, bound_sc = _scopes(project, clip)
    if span_changed and clip.raw_start is not None and clip.raw_end is not None:
        feedback.record_trim(bound_sc, clip.start - clip.raw_start,
                             clip.end - clip.raw_end)
    if clip.features:
        feedback.record_rating(clip_id, score_sc, 1.0, clip.features,
                               source="trim", weight=0.5)

    clip.status = ClipStatus.rendering
    clip.error = None
    with store.mutate(project_id) as p:
        existing = p.clip(clip_id)
        if existing is None:  # deleted while we were editing the detached copy
            raise HTTPException(404, "clip not found")
        # A render may have finished since we read the project; keep its output
        # URLs so the editor preview doesn't blank out before the re-render.
        clip.export_url = existing.export_url or clip.export_url
        clip.thumb_url = existing.thumb_url or clip.thumb_url
        p.clips[p.clips.index(existing)] = clip

    # Re-render just this clip off the request path.
    threading.Thread(target=engine.rerender_clip, args=(project_id, clip_id),
                     daemon=True).start()
    return clip


@router.post("/projects/{project_id}/clips/{clip_id}/rerender", response_model=Clip)
def rerender(project_id: str, clip_id: str) -> Clip:
    project = store.get(project_id)
    if not project or not project.clip(clip_id):
        raise HTTPException(404, "clip not found")
    with store.mutate(project_id) as p:
        c = p.clip(clip_id)
        if c is None:  # deleted between the check above and taking the lock
            raise HTTPException(404, "clip not found")
        c.status = ClipStatus.rendering
    threading.Thread(target=engine.rerender_clip, args=(project_id, clip_id),
                     daemon=True).start()
    return store.get(project_id).clip(clip_id)


class BatchRerenderBody(BaseModel):
    clip_ids: list[str]


@router.post("/projects/{project_id}/clips/rerender", response_model=Project)
def rerender_selected(project_id: str, body: BatchRerenderBody) -> Project:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    ids = [cid for cid in body.clip_ids if project.clip(cid)]
    if not ids:
        raise HTTPException(400, "choose at least one clip to re-render")
    with store.mutate(project_id) as p:
        for cid in ids:
            c = p.clip(cid)
            if c:
                c.status = ClipStatus.rendering
                c.error = None
    threading.Thread(target=engine.rerender_clips, args=(project_id, ids),
                     daemon=True).start()
    return store.get(project_id)


class RatingBody(BaseModel):
    rating: str   # "up" | "down" | "none"


@router.post("/projects/{project_id}/clips/{clip_id}/feedback", response_model=Clip)
def rate_clip(project_id: str, clip_id: str, body: RatingBody) -> Clip:
    """Thumbs up/down — teaches the local scorer your taste."""
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    score_sc, _ = _scopes(project, clip)
    state: str | None = None
    if body.rating == "up":
        feedback.record_rating(clip_id, score_sc, 1.0, clip.features, weight=1.0)
        state = "up"
    elif body.rating == "down":
        feedback.record_rating(clip_id, score_sc, 0.0, clip.features, weight=1.0)
        state = "down"
    else:
        feedback.remove_rating(clip_id, "explicit")
    with store.mutate(project_id) as p:
        c = p.clip(clip_id)
        if c:
            c.feedback = state
    return store.get(project_id).clip(clip_id)


@router.get("/learning")
def learning_overview() -> dict:
    return feedback.overview()


@router.post("/learning/reset")
def learning_reset(body: dict | None = None) -> dict:
    feedback.reset((body or {}).get("scope"))
    return {"ok": True}


class MontageCreate(BaseModel):
    clip_ids: list[str]            # in the order they should appear
    title: str | None = None


@router.post("/projects/{project_id}/montage", response_model=Montage)
def create_montage(project_id: str, body: MontageCreate) -> Montage:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    ordered = [c for c in (project.clip(cid) for cid in body.clip_ids)
               if c and c.export_url]
    if len(ordered) < 2:
        raise HTTPException(400, "select at least 2 rendered clips for a montage")
    score, factors = montage_mod.score_montage(ordered)
    mtg = Montage(title=(body.title or f"Montage of {len(ordered)} clips")[:120],
                  clip_ids=[c.id for c in ordered], score=score, factors=factors,
                  status=ClipStatus.rendering)
    with store.mutate(project_id) as p:
        p.montages.insert(0, mtg)
    threading.Thread(target=engine.build_montage, args=(project_id, mtg.id),
                     daemon=True).start()
    return mtg


@router.get("/projects/{project_id}/montages/{montage_id}/download")
def download_montage(project_id: str, montage_id: str):
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    mtg = project.montage(montage_id)
    if not mtg or not mtg.export_url:
        raise HTTPException(409, "montage is not rendered yet")
    path = get_settings().media_dir / mtg.export_url.removeprefix("/media/")
    if not path.exists():
        raise HTTPException(404, "montage file missing")
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in mtg.title).strip()
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{(safe or montage_id)[:60]}.mp4")


@router.get("/projects/{project_id}/clips/{clip_id}/download")
def download_clip(project_id: str, clip_id: str):
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip or not clip.export_url:
        raise HTTPException(409, "clip is not rendered yet")
    path = get_settings().media_dir / clip.export_url.removeprefix("/media/")
    if not path.exists():
        raise HTTPException(404, "clip file missing")
    # Downloading a clip is an implicit "keep" — a weak positive signal.
    if clip.features:
        score_sc, _ = _scopes(project, clip)
        feedback.record_rating(clip_id, score_sc, 1.0, clip.features,
                               source="download", weight=0.5)
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in clip.title).strip()
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{(safe or clip_id)[:60]}.mp4")


@router.get("/projects/{project_id}/clips/{clip_id}/captions.srt")
def download_srt(project_id: str, clip_id: str):
    """Caption sidecar (.srt) — import into Premiere/Resolve to restyle."""
    project = store.get(project_id)
    clip = project.clip(clip_id) if project else None
    if not clip:
        raise HTTPException(404, "clip not found")
    if not clip.captions.words:
        raise HTTPException(409, "this clip has no captions")
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in clip.title).strip()
    return PlainTextResponse(
        build_srt(clip.captions), media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{(safe or clip_id)[:60]}.srt"'})
