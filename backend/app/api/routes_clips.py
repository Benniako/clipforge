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
from pydantic import BaseModel, Field

from .. import feedback, store
from ..config import get_settings
from ..models import (ASPECTS, CaptionWord, Clip, ClipStatus, LayoutType,
                      Montage, Project, Rect, Reframe, ReframeKeyframe,
                      StyleTemplate)
from ..pipeline import captionize
from ..pipeline.captions import build_srt
from ..pipeline import montage as montage_mod
from ..pipeline import reframe as reframe_mod
from ..pipeline.orchestrator import engine
from ..providers import score as score_mod
from ..services import audio_waveform, suggestions_copilot
from .. import user_styles

router = APIRouter(prefix="/api", tags=["clips"])


def _scopes(project, clip) -> tuple[str, str]:
    """(score_scope, boundary_scope) for this clip's content type.

    Talking learns per platform; gameplay learns per game profile.
    """
    ct = "gameplay" if clip.kind == "gameplay" else "talking"
    key = project.settings.game_profile if ct == "gameplay" else project.settings.platform.value
    return feedback.score_scope(ct, key), feedback.bound_scope(ct, key)


def _filter_caption_words_by_speakers(words: list[CaptionWord],
                                      speakers: set[int] | None) -> list[CaptionWord]:
    """Apply the clip's speaker keep-set to manual caption word replacements."""
    if speakers is None:
        return words
    return [w for w in words if (w.speaker or 0) in speakers]


def _caption_export_set(clip: Clip):
    preview = max(float(getattr(clip, "loop_preview_seconds", 0.0) or 0.0), 0.0)
    duration = float(clip.tightened_duration or clip.duration)
    if preview <= 0.0:
        return clip.captions
    return captionize.with_loop_preview(clip.captions, preview, duration)


class ClipEdit(BaseModel):
    title: str | None = None
    description: str | None = None
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
    # Seconds of rendered tail to prepend for a loopability preview. 0 disables.
    loop_preview_seconds: float | None = None
    # Per-clip output aspect: an ASPECTS key, or "" to return to the
    # project default.
    aspect: str | None = None


@router.get("/styles")
def list_styles():
    return [s.model_dump() for s in user_styles.all_styles()]


class StyleCreate(BaseModel):
    id: str
    name: str
    font: str = "DejaVu Sans"
    font_size: int = 92
    primary: str = "FFFFFF"
    highlight: str = "F5C518"
    outline: str = "000000"
    outline_w: int = 6
    y_frac: float = 0.78
    uppercase: bool = True
    emphasis: bool = True
    emoji: bool = False


@router.post("/styles", status_code=201)
def create_style(style: StyleCreate) -> StyleTemplate:
    """Create a new custom style template (brand template)."""
    st = StyleTemplate(**style.model_dump())
    user_styles.create_style(st)
    return st


@router.put("/styles/{style_id}")
def update_style(style_id: str, style: StyleCreate) -> StyleTemplate:
    """Update an existing custom style template."""
    if style_id != style.id:
        raise HTTPException(400, "style id in path must match body")
    updates = style.model_dump(exclude={"id"})
    updated = user_styles.update_style(style_id, updates)
    if updated is None:
        raise HTTPException(404, f"style '{style_id}' not found")
    return updated


@router.delete("/styles/{style_id}")
def delete_style(style_id: str) -> dict:
    """Delete a custom style template."""
    ok = user_styles.delete_style(style_id)
    if not ok:
        raise HTTPException(404, f"style '{style_id}' not found")
    return {"ok": True}


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

    if edit.description is not None:
        clip.description = edit.description.strip()[:600]

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
        clip.captions.style_id = user_styles.get_style(edit.style_id).id

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
    elif speakers_sent and edit.caption_words is None:
        # Only the speaker keep-set changed: prefer a clean transcript rebuild,
        # but still honour the toggle for clips that only have manual words.
        if project.transcript:
            _rebuild_captions()
        else:
            clip.captions.words = _filter_caption_words_by_speakers(
                clip.captions.words, spk)

    if edit.caption_words is not None:
        clip.captions.words = _filter_caption_words_by_speakers(
            edit.caption_words, spk)

    if "reframe_cx" in edit.model_fields_set:
        if edit.reframe_cx is not None:
            cx = max(0.0, min(1.0, edit.reframe_cx))
            # A manual crop centre keeps the clip's facecam layout intact.
            clip.reframe = Reframe(layout=clip.reframe.layout,
                                   keyframes=[ReframeKeyframe(t=0.0, cx=cx)],
                                   tracked=False, overridden=True,
                                   cx_overridden=True,
                                   facecam=clip.reframe.facecam)
        else:
            # Explicit null = reset to automatic tracking.
            clip.reframe = Reframe(layout=clip.reframe.layout, tracked=False,
                                   keyframes=[], overridden=False,
                                   cx_overridden=False,
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

    if edit.loop_preview_seconds is not None:
        seconds = float(edit.loop_preview_seconds)
        clip.loop_preview_seconds = 0.0 if seconds <= 0 else round(
            max(3.0, min(5.0, seconds)), 1)

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
    path = get_settings().media_dir / (mtg.export_url[len("/media/"):] if mtg.export_url and mtg.export_url.startswith("/media/") else mtg.export_url)
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
    path = get_settings().media_dir / (clip.export_url[7:] if clip.export_url.startswith("/media/") else clip.export_url)
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
    fname = f"{(safe or clip_id)[:60]}.srt"
    return PlainTextResponse(
        build_srt(_caption_export_set(clip)), media_type="application/x-subrip",
        headers={"Content-Disposition": f"attachment; filename={fname}"})


@router.get("/projects/{project_id}/clips/{clip_id}/publish-content")
def publish_content(project_id: str, clip_id: str, platform: str = "generic"):
    """AI-generated publish-ready content for this clip (titles, description,
    hashtags). Generated on demand so the user can iterate."""
    from ..providers import hashtags as hashtags_mod
    from ..providers import llm as llm_mod

    project = store.get(project_id)
    clip = project.clip(clip_id) if project else None
    if not clip:
        raise HTTPException(404, "clip not found")
    transcript = getattr(project, "transcript", None)
    excerpt = clip.transcript_excerpt or (
        " ".join(w.text for w in transcript.words
                 if w.end > clip.start and w.t < clip.end)[:600]
        if transcript else "")
    lang = getattr(transcript, "language", "en") or "en"
    game = getattr(project, "settings", None) and getattr(project.settings, "game_profile", None)

    titles = llm_mod.suggest_title_variants(excerpt, lang=lang, platform=platform)
    desc = llm_mod.suggest_description(excerpt, lang=lang, platform=platform,
                                       title=titles[0] if titles else None)
    ai_tags = llm_mod.suggest_hashtags_llm(excerpt, lang=lang, platform=platform, game=game)
    if ai_tags is None:
        ai_tags = hashtags_mod.suggest_hashtags(
            excerpt, content_type=getattr(clip, "kind", "talking"),
            platform=platform, game=game)

    return {
        "titles": titles,
        "description": desc or "",
        "hashtags": ai_tags,
        "excerpt": excerpt[:200],
    }


# --------------------------------------------------------------------------- #
# Interactive Studio Endpoints (Waveform, Storyboard, Suggestions, Batch)
# --------------------------------------------------------------------------- #

@router.get("/projects/{project_id}/clips/{clip_id}/waveform")
def get_clip_waveform(project_id: str, clip_id: str) -> dict:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return audio_waveform.extract_clip_waveform(project, clip)


@router.get("/projects/{project_id}/clips/{clip_id}/storyboard")
def get_clip_storyboard(project_id: str, clip_id: str) -> dict:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")

    settings = get_settings()
    dur = max(0.1, round(clip.duration, 3))
    cols = 5
    rows = 2
    total_frames = cols * rows
    interval_s = round(dur / total_frames, 2)
    tile_w = 160
    tile_h = 90

    # Destination sprite path
    rel_sprite = f"clips/{clip_id}_storyboard.jpg"
    dest_path = settings.media_dir / project_id / rel_sprite
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sprite_url = f"/media/{project_id}/{rel_sprite}"

    if not dest_path.exists():
        src_file = None
        is_clip_export = False
        if clip.export_url and clip.export_url.startswith("/media/"):
            cand = settings.media_dir / clip.export_url[len("/media/"):]
            if cand.exists():
                src_file = cand
                is_clip_export = True
        if not src_file and project.source and project.source.path:
            cand = settings.media_dir / project.source.path
            if cand.exists():
                src_file = cand

        if src_file and getattr(settings, "ffmpeg", None):
            try:
                import subprocess
                ss_args = [] if is_clip_export else ["-ss", str(clip.start), "-to", str(clip.end)]
                cmd = [
                    settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    *ss_args,
                    "-i", str(src_file),
                    "-vf", f"fps=1/{max(0.1, interval_s)},scale={tile_w}:{tile_h},tile={cols}x{rows}",
                    "-frames:v", "1",
                    str(dest_path)
                ]
                subprocess.run(cmd, check=True, timeout=10, capture_output=True)
            except Exception:
                pass

    if not dest_path.exists():
        sprite_url = clip.thumb_url or f"/media/{project_id}/{rel_sprite}"

    return {
        "clip_id": clip_id,
        "sprite_url": sprite_url,
        "cols": cols,
        "rows": rows,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "interval_s": interval_s,
        "total_frames": total_frames,
        "duration_s": dur,
    }


@router.get("/projects/{project_id}/clips/{clip_id}/suggestions")
def get_clip_suggestions(project_id: str, clip_id: str) -> dict:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    suggestions = suggestions_copilot.generate_clip_suggestions(project, clip)
    return {
        "clip_id": clip_id,
        "suggestions": suggestions,
    }


class SuggestionApplyBody(BaseModel):
    patch: dict | None = None


@router.post("/projects/{project_id}/clips/{clip_id}/suggestions/{suggestion_id}/apply")
def apply_clip_suggestion(project_id: str, clip_id: str, suggestion_id: str,
                          body: SuggestionApplyBody | None = None) -> dict:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")
    clip = project.clip(clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")

    suggestions = suggestions_copilot.generate_clip_suggestions(project, clip)
    matching = next((s for s in suggestions if s["id"] == suggestion_id), None)

    patch = {}
    if matching:
        patch = dict(matching.get("patch", {}))
    if body and body.patch:
        patch.update(body.patch)

    if not matching and not patch:
        raise HTTPException(404, f"suggestion '{suggestion_id}' not found")

    updated_fields: list[str] = []
    src_dur = project.source.duration if project.source else clip.end
    span_changed = False

    # 1. Start boundary trim
    if "start_t" in patch or "start" in patch:
        new_start = float(patch.get("start_t", patch.get("start")))
        new_start = max(0.0, min(clip.end - 1.0, new_start))
        if new_start != clip.start:
            clip.start = round(new_start, 3)
            span_changed = True
            updated_fields.append("start_t")

    # 2. End boundary trim
    if "end_t" in patch or "end" in patch:
        new_end = float(patch.get("end_t", patch.get("end")))
        new_end = max(clip.start + 1.0, min(src_dur, new_end))
        if new_end != clip.end:
            clip.end = round(new_end, 3)
            span_changed = True
            updated_fields.append("end_t")

    # 3. Layout reframe
    if "layout" in patch:
        try:
            lay = LayoutType(patch["layout"])
            if lay in (LayoutType.split, LayoutType.framed):
                cam = clip.reframe.facecam or project.facecam
                if cam:
                    clip.reframe.facecam = cam
            clip.reframe.layout = lay
            clip.reframe.overridden = True
            updated_fields.append("layout")
        except ValueError:
            raise HTTPException(400, f"unknown layout '{patch['layout']}'")

    # 4. Teaser loop
    if "teaser_loop" in patch:
        clip.teaser_loop = bool(patch["teaser_loop"])
        updated_fields.append("teaser_loop")

    # 5. Dead air tighten
    if "tighten" in patch:
        if "segments" in patch and patch["segments"]:
            clip.segments = patch["segments"]
        elif project.transcript:
            segs = captionize.compute_tight_segments(project.transcript, clip.start, clip.end)
            clip.segments = [[round(a, 3), round(b, 3)] for a, b in segs]
        updated_fields.append("tighten")

    # 6. Manual cx override
    if "reframe_cx" in patch:
        cx = max(0.0, min(1.0, float(patch["reframe_cx"])))
        clip.reframe = Reframe(layout=clip.reframe.layout,
                               keyframes=[ReframeKeyframe(t=0.0, cx=cx)],
                               tracked=False, overridden=True,
                               cx_overridden=True,
                               facecam=clip.reframe.facecam)
        updated_fields.append("reframe_cx")

    # If span changed, re-derive captions
    if span_changed and project.transcript:
        spk = set(clip.caption_speakers) if clip.caption_speakers is not None else None
        sid = clip.captions.style_id
        if clip.kind == "gameplay":
            if project.transcript.provider != "synthetic":
                clip.captions = captionize.build_caption_set(
                    project.transcript, clip.start, clip.end, sid,
                    exclude=[(a, b) for a, b in clip.caption_mute] or None, speakers=spk)
        elif clip.segments:
            clip.captions = captionize.build_tight_caption_set(
                project.transcript, [(a, b) for a, b in clip.segments], sid, speakers=spk)
        else:
            clip.captions = captionize.build_caption_set(
                project.transcript, clip.start, clip.end, sid, speakers=spk)

    clip.status = ClipStatus.rendering
    clip.error = None

    with store.mutate(project_id) as p:
        existing = p.clip(clip_id)
        if existing is None:
            raise HTTPException(404, "clip not found")
        clip.export_url = existing.export_url or clip.export_url
        clip.thumb_url = existing.thumb_url or clip.thumb_url
        p.clips[p.clips.index(existing)] = clip

    threading.Thread(target=engine.rerender_clip, args=(project_id, clip_id),
                     daemon=True).start()

    return {
        "success": True,
        "clip": clip.model_dump(),
        "updated_fields": updated_fields,
    }


class BatchClipOperation(BaseModel):
    clip_ids: list[str]
    action: str  # "rate" | "export" | "tag" | "delete" | "sequence"
    payload: dict = Field(default_factory=dict)


@router.post("/projects/{project_id}/clips/batch")
def batch_clip_operations(project_id: str, body: BatchClipOperation) -> dict:
    project = store.get(project_id)
    if not project:
        raise HTTPException(404, "project not found")

    if not body.clip_ids:
        raise HTTPException(400, "clip_ids list cannot be empty")

    valid_actions = {"rate", "export", "tag", "delete", "sequence"}
    if body.action not in valid_actions:
        raise HTTPException(400, f"unsupported batch action '{body.action}'. Must be one of {valid_actions}")

    target_ids = set(body.clip_ids)
    affected_ids: list[str] = [c.id for c in project.clips if c.id in target_ids]

    if body.action == "rate":
        rating_raw = body.payload.get("rating")
        if rating_raw is None:
            raise HTTPException(400, "payload must contain 'rating' (e.g. 1-5 or 'up'/'down')")
        with store.mutate(project_id) as p:
            for cid in affected_ids:
                c = p.clip(cid)
                if c:
                    if isinstance(rating_raw, (int, float)):
                        val = int(rating_raw)
                        c.score = min(100, max(0, val * 20))
                        c.feedback = "up" if val >= 4 else ("down" if val <= 2 else None)
                    elif str(rating_raw).lower() in ("up", "down"):
                        c.feedback = str(rating_raw).lower()
                    else:
                        c.feedback = None

    elif body.action == "tag":
        tags = body.payload.get("tags") or body.payload.get("hashtags") or []
        if not isinstance(tags, list):
            raise HTTPException(400, "'tags' in payload must be a list of strings")
        clean_tags = [str(t).strip() for t in tags if str(t).strip()]
        with store.mutate(project_id) as p:
            for cid in affected_ids:
                c = p.clip(cid)
                if c:
                    for t in clean_tags:
                        if t not in c.hashtags:
                            c.hashtags.append(t)

    elif body.action == "delete":
        with store.mutate(project_id) as p:
            p.clips = [c for c in p.clips if c.id not in target_ids]

    elif body.action == "sequence":
        if "order" in body.payload:
            order_list = body.payload["order"]
            if not isinstance(order_list, list) or not all(isinstance(x, str) for x in order_list):
                raise HTTPException(400, "'order' in payload must be a list of clip ID strings")
        else:
            order_list = body.clip_ids

        seen: set[str] = set()
        deduped_order: list[str] = []
        for cid in order_list:
            if cid not in seen:
                seen.add(cid)
                deduped_order.append(cid)

        with store.mutate(project_id) as p:
            id_to_clip = {c.id: c for c in p.clips}
            ordered_clips = [id_to_clip[cid] for cid in deduped_order if cid in id_to_clip]
            remaining = [c for c in p.clips if c.id not in seen]
            p.clips = ordered_clips + remaining

    elif body.action == "export":
        with store.mutate(project_id) as p:
            for cid in affected_ids:
                c = p.clip(cid)
                if c:
                    c.status = ClipStatus.rendering
                    c.error = None
        threading.Thread(target=engine.rerender_clips, args=(project_id, affected_ids),
                         daemon=True).start()

    updated_project = store.get(project_id)
    return {
        "success": True,
        "action": body.action,
        "affected_clip_ids": affected_ids,
        "project": updated_project.model_dump(),
    }

