"""Render stage — produce the final 9:16 captioned MP4 for one clip.

ffmpeg does the heavy lifting: cut the source span, crop to a speaker-tracked
9:16 window (a time-varying `x` when the speaker moves, a static crop when they
don't — needless motion looks cheap), scale to the 1080×1920 canvas, burn the
ASS captions with libass, and encode H.264/AAC with faststart for instant web
playback.

The whole filtergraph is written to a script file and passed via
``-filter_script:v`` so we never fight shell/filtergraph escaping rules.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("clipforge.render")

_X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-profile:v", "high"]

from ..config import get_settings
from ..media import ffmpeg
from ..media.ffmpeg import MediaInfo
from ..models import Clip, LayoutType, Rect, StyleTemplate
from . import captions as captions_mod
# Zoom is a time-varying punch-in on emphasis words and scene cuts. Pure module,
# tested; build_zoom_filter returns None when no spikes exist (filter skipped).
from . import zoom as zoom_mod

# Share of the canvas height the facecam strip takes in the split layout, and
# the PiP width fraction in the framed layout — the proportions gaming TikToks
# actually use.
SPLIT_CAM_FRAC = 0.30
PIP_WIDTH_FRAC = 0.42


def _even(x: float) -> int:
    return int(x) // 2 * 2


def build_crop(clip: Clip, src_w: int, src_h: int,
               out_w: int = 1080, out_h: int = 1920) -> tuple[int, int, str]:
    """Return (crop_w, crop_h, x_arg) for the target aspect out_w:out_h.

    x_arg is a number (static crop) or a `t` expression (panning). The crop
    window is the largest region of the source matching the output aspect.
    """
    a = out_w / out_h
    if src_w / src_h >= a:        # source wider than target -> full height
        ch = _even(src_h)
        cw = min(_even(ch * a), _even(src_w))
    else:                         # source narrower/taller -> full width
        cw = _even(src_w)
        ch = min(_even(cw / a), _even(src_h))
    max_x = max(src_w - cw, 0)

    kfs = clip.reframe.keyframes or []
    cxs = [k.cx for k in kfs] or [0.5]
    # Static crop when the subject barely moves — avoids cheap-looking drift.
    if (max(cxs) - min(cxs)) < 0.04:
        cx = sorted(cxs)[len(cxs) // 2]  # median
        x = int(round(min(max(cx * src_w - cw / 2, 0), max_x)))
        return cw, ch, str(x)

    return cw, ch, _crop_expr(kfs, src_w, cw, max_x)


def _crop_expr(kfs, src_w: int, cw: int, max_x: int) -> str:
    """Piecewise-linear x(t) from centre-x keyframes, clamped to the frame.

    Wrapped in single quotes by the caller's context (the ass/crop options use
    `:` separators); we keep commas here and rely on the filter_script file +
    quoting to pass them through verbatim.
    """
    pts = [(round(k.t, 3), k.cx) for k in kfs]
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))

    expr = repr(pts[-1][1])  # value for t beyond the last keyframe
    for i in range(len(pts) - 2, -1, -1):
        t_i, c_i = pts[i]
        t_j, c_j = pts[i + 1]
        dt = (t_j - t_i) or 1e-6
        seg = f"({c_i}+({c_j - c_i})*(t-{t_i})/{dt})"
        expr = f"if(lt(t,{t_j}),{seg},{expr})"

    # cx fraction -> pixel x, clamped to [0, max_x]
    return f"max(0,min({max_x},({expr})*{src_w}-{cw / 2}))"


def rect_crop(rect: Rect, src_w: int, src_h: int,
              aspect: float | None = None) -> tuple[int, int, int, int]:
    """Pixel crop (w, h, x, y) covering ``rect``, optionally grown to ``aspect``
    (w/h) around its centre, clamped inside the frame."""
    w = rect.w * src_w
    h = rect.h * src_h
    if aspect:
        if w / max(h, 1e-6) < aspect:
            w = h * aspect
        else:
            h = w / aspect
        if w > src_w:
            w, h = src_w, src_w / aspect
        if h > src_h:
            h, w = src_h, src_h * aspect
    cw, ch = max(_even(w), 2), max(_even(h), 2)
    cx = (rect.x + rect.w / 2) * src_w
    cy = (rect.y + rect.h / 2) * src_h
    x = int(round(min(max(cx - cw / 2, 0), src_w - cw)))
    y = int(round(min(max(cy - ch / 2, 0), src_h - ch)))
    return cw, ch, x, y


def game_pane_crop(cx: float, cam: Rect | None, src_w: int, src_h: int,
                   aspect: float) -> tuple[int, int, int, int]:
    """Largest crop of the given aspect centred near ``cx``, shifted off the
    facecam when that's possible — the streamer shouldn't appear twice."""
    if src_w / src_h >= aspect:
        ch = _even(src_h)
        cw = min(_even(ch * aspect), _even(src_w))
    else:
        cw = _even(src_w)
        ch = min(_even(cw / aspect), _even(src_h))
    max_x = max(src_w - cw, 0)
    x = min(max(cx * src_w - cw / 2, 0), max_x)
    if cam is not None and max_x > 0:
        cam_x0, cam_x1 = cam.x * src_w, (cam.x + cam.w) * src_w
        if cam_x0 < x + cw and cam_x1 > x:        # overlaps — try to dodge
            options = []
            if cam_x1 <= max_x:
                options.append((abs(cam_x1 - x), cam_x1))      # right of cam
            if cam_x0 - cw >= 0:
                options.append((abs(cam_x0 - cw - x), cam_x0 - cw))  # left of cam
            if options:
                x = min(options)[1]
    return cw, ch, int(round(x)), _even((src_h - ch) / 2)


def _composed_graph(clip: Clip, cam: Rect, info: MediaInfo,
                    out_w: int, out_h: int, ass_part: str | None) -> list[str]:
    """Filtergraph lines for the facecam layouts.

    split  — cam strip scaled across the top, gameplay pane below (vstack).
    framed — gameplay full-bleed with the cam as a bordered PiP up top.
    Both use static crops: a pinned cam and a steady game window read better
    than a panning gameplay crop.
    """
    cxs = sorted(k.cx for k in clip.reframe.keyframes) or [0.5]
    cx = cxs[len(cxs) // 2]
    lines = ["[0:v]setpts=PTS-STARTPTS,split=2[cam0][game0];"]
    if clip.reframe.layout == LayoutType.split:
        top_h = _even(out_h * SPLIT_CAM_FRAC)
        bot_h = out_h - top_h
        ccw, cch, ccx, ccy = rect_crop(cam, info.width, info.height, out_w / top_h)
        gcw, gch, gx, gy = game_pane_crop(cx, cam, info.width, info.height,
                                          out_w / bot_h)
        tail = ["vstack=inputs=2", "setsar=1"] + ([ass_part] if ass_part else [])
        lines += [
            f"[cam0]crop=w={ccw}:h={cch}:x={ccx}:y={ccy},"
            f"scale={out_w}:{top_h}:flags=lanczos[cam1];",
            f"[game0]crop=w={gcw}:h={gch}:x={gx}:y={gy},"
            f"scale={out_w}:{bot_h}:flags=lanczos[game1];",
            f"[cam1][game1]{','.join(tail)}[vo]",
        ]
    else:  # framed: full-bleed gameplay + PiP cam
        gcw, gch, gx, gy = game_pane_crop(cx, cam, info.width, info.height,
                                          out_w / out_h)
        ccw, cch, ccx, ccy = rect_crop(cam, info.width, info.height)
        pip_w = _even(out_w * PIP_WIDTH_FRAC)
        pip_h = max(_even(pip_w * cch / max(ccw, 1)), 2)
        margin = _even(out_h * 0.04)
        post = f",{ass_part}" if ass_part else ""
        lines += [
            f"[game0]crop=w={gcw}:h={gch}:x={gx}:y={gy},"
            f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[bg1];",
            f"[cam0]crop=w={ccw}:h={cch}:x={ccx}:y={ccy},"
            f"scale={pip_w}:{pip_h}:flags=lanczos,"
            f"pad=w=iw+8:h=ih+8:x=4:y=4:color=white[cam1];",
            f"[bg1][cam1]overlay=x=(W-w)/2:y={margin}{post}[vo]",
        ]
    return lines




def render_clip(clip: Clip, src_path: str, info: MediaInfo, style: StyleTemplate,
                out_path: Path, thumb_path: Path, *, out_w: int, out_h: int,
                burn_captions: bool = True, motion: str = "none",
                background_music: str = "") -> None:
    """Render ``clip`` from ``src_path`` into ``out_path`` (+ a thumbnail).

    With ``burn_captions=False`` the clip is reframed/encoded but left clean — for
    re-editing in a desktop NLE where you'd add your own captions. When the clip
    carries jump-cut ``segments``, they're trimmed and concatenated; ``motion``
    "push" adds a slow push-in.

    ``background_music`` is a path to an audio file mixed at low volume behind
    the clip audio. Empty string (default) omits the background track.
    """
    cw, ch, x_arg = build_crop(clip, info.width, info.height, out_w, out_h)
    static = x_arg.lstrip("-").isdigit()
    x_field = x_arg if static else f"'{x_arg}'"
    # Absolute, because ffmpeg runs with cwd set to the temp dir below.
    src_abs = os.path.abspath(str(src_path))
    out_abs = os.path.abspath(str(out_path))
    fps = min(info.fps or 30, 60)
    segments = [(a, b) for a, b in (clip.segments or []) if b > a]
    tightened = len(segments) >= 2
    eff_dur = sum(b - a for a, b in segments) if tightened else clip.duration
    # Facecam layouts (gameplay): cam + gameplay composed on a vertical canvas.
    cam = clip.reframe.facecam
    composed = (clip.reframe.layout in (LayoutType.split, LayoutType.framed)
                and cam is not None and out_h > out_w and not tightened)

    with tempfile.TemporaryDirectory() as tmp:
        # Reference the .ass by a BARE filename and run ffmpeg with cwd=tmp, so
        # the filtergraph never contains a Windows path (drive-letter ':' and
        # '\' are filtergraph metacharacters and would break parsing).
        ass_part = None
        if burn_captions and clip.captions.words:
            captions_mod.write_ass(clip.captions, style, out_w, out_h, Path(tmp) / "cap.ass")
            ass_part = "ass=f=cap.ass"

        parts = [f"crop=w={cw}:h={ch}:x={x_field}:y=(ih-{ch})/2",
                 f"scale={out_w}:{out_h}:flags=lanczos", "setsar=1"]
        if motion == "push" and eff_dur > 0 and not composed:
            # Slow push-in to ~1.06x over the clip — subtle, edited feel.
            frames = max(int(eff_dur * fps), 1)
            parts.append(
                f"zoompan=z='min(1+{0.06 / frames:.8f}*on,1.06)'"
                f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d=1:s={out_w}x{out_h}:fps={fps:.3f}")
        # Auto zoom (AI Boost): brief punch-in on emphasis words, gated by the
        # clip's settings (default on for most presets). build_zoom_filter
        # returns None when no spikes exist, so the existing scale+setsar chain
        # is unchanged.
        if motion != "push" and not composed and not tightened:
            # Generate zoom spikes from caption emphasis. The emphasis marks
            # are set by caption_fx.annotate during build_ass; we re-annotate
            # here (pure function, no-op when emphasis is off on the style).
            from .caption_fx import annotate as annotate_words
            ann_words = annotate_words(
                clip.captions.words, lang=clip.captions.lang or "en",
                emphasis=style.emphasis, emoji=style.emoji,
                max_words_per_line=clip.captions.max_words_per_line)
            zoom_spikes = zoom_mod.spikes_from_emphasis(ann_words)
            if zoom_spikes:
                zoom_filter = zoom_mod.build_zoom_filter(
                    zoom_spikes, out_w, out_h)
                if zoom_filter:
                    # Replace scale + setsar with the dynamic zoom filter.
                    # The zoom filter itself already handles the final
                    # dimensions (s=out_wxout_h), so setsar isn't needed.
                    parts[1] = zoom_filter
                    parts[2] = ""  # setsar consumed by zoom filter
        # Remove empty parts (e.g. the setsar slot when zoom replaced it).
        parts = [p for p in parts if p]
        if ass_part:
            parts.append(ass_part)
        vchain = ",".join(parts)

        if composed:
            lines = _composed_graph(clip, cam, info, out_w, out_h, ass_part)
            (Path(tmp) / "graph.txt").write_text("\n".join(lines), encoding="utf-8")
            base = ["-ss", f"{clip.start:.3f}", "-i", src_abs,
                    "-t", f"{clip.duration:.3f}",
                    "-filter_complex_script", "graph.txt", "-map", "[vo]"]
            audio = (["-map", "0:a?", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                      "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
                     if info.has_audio else ["-an"])
        elif tightened:
            # Jump cuts: trim each speech segment, concat, then the normal chain.
            # Seek the input to the clip span (instead of decoding the whole
            # source for every clip) and rebase the trim times onto the seeked
            # timeline, where the first frame lands at t=0.
            seek = segments[0][0]
            span = segments[-1][1] - seek
            lines = []
            for i, (a, b) in enumerate(segments):
                a, b = max(a - seek, 0.0), b - seek
                lines.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},"
                             f"setpts=PTS-STARTPTS[v{i}];")
                if info.has_audio:
                    lines.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},"
                                 f"asetpts=PTS-STARTPTS[a{i}];")
            n = len(segments)
            if info.has_audio:
                pairs = "".join(f"[v{i}][a{i}]" for i in range(n))
                lines.append(f"{pairs}concat=n={n}:v=1:a=1[vc][ac];")
                lines.append("[ac]loudnorm=I=-16:TP=-1.5:LRA=11[ao];")
            else:
                pairs = "".join(f"[v{i}]" for i in range(n))
                lines.append(f"{pairs}concat=n={n}:v=1:a=0[vc];")
            lines.append(f"[vc]{vchain}[vo]")
            (Path(tmp) / "graph.txt").write_text("\n".join(lines), encoding="utf-8")
            # -t as an *input* option: read only the span; the output duration
            # is the (shorter) sum of the concatenated segments.
            base = ["-ss", f"{seek:.3f}", "-t", f"{span:.3f}", "-i", src_abs,
                    "-filter_complex_script", "graph.txt", "-map", "[vo]"]
            audio = (["-map", "[ao]", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
                     if info.has_audio else ["-an"])
        else:
            # B-roll PiP overlay: when the clip carries a broll_overlay, the
            # talking head gets a picture-in-picture showing the strong visual
            # moment during the designated window. We switch from a simple
            # one-input -filter_script:v to a two-input -filter_complex_script.
            broll = None
            broll_dur = 0.0
            if hasattr(clip, "broll_overlay") and clip.broll_overlay:
                broll = clip.broll_overlay
                broll_dur = min(float(broll.get("duration", 0)),
                                clip.duration - float(broll.get("start_rel", 0)))
            if broll and broll_dur > 0.300:
                ov_start = float(broll.get("start_rel", 0))
                ov_end = ov_start + broll_dur
                pip_w = _even(out_w * 0.30)
                pip_h = _even(pip_w * out_h / out_w)
                lines = [
                    f"[0:v]setpts=PTS-STARTPTS,{vchain}[bg];",
                    f"[1:v]setpts=PTS-STARTPTS,"
                    f"scale={pip_w}:{pip_h}:flags=lanczos[br];",
                    f"[bg][br]overlay=x=W-w-20:y=20"
                    f":enable='between(t,{ov_start:.2f},{ov_end:.2f})'[vo]",
                ]
                (Path(tmp) / "graph.txt").write_text("\n".join(lines), encoding="utf-8")
                base = ["-ss", f"{clip.start:.3f}", "-i", src_abs,
                        "-ss", f"{float(broll['source_t']):.3f}", "-i", src_abs,
                        "-t", f"{clip.duration:.3f}",
                        "-filter_complex_script", "graph.txt", "-map", "[vo]"]
            else:
                (Path(tmp) / "graph.txt").write_text("setpts=PTS-STARTPTS," + vchain,
                                                     encoding="utf-8")
                base = ["-ss", f"{clip.start:.3f}", "-i", src_abs,
                        "-t", f"{clip.duration:.3f}", "-filter_script:v", "graph.txt"]
            if info.has_audio:
                # Normalise loudness so clips sound consistent across a batch.
                audio = ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                         "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
            else:
                audio = ["-an"]

        # Background music overlay: mix a second audio track at low volume.
        bgm = background_music.strip()
        if bgm and os.path.exists(bgm) and info.has_audio and audio != ["-an"]:
            bgm_abs = os.path.abspath(bgm)
            # Insert the music file as an additional input, then use amix.
            base.insert(-1, "-i")
            base.insert(-1, bgm_abs)
            # Replace audio args with filtered complex featuring amix.
            audio = [
                "-filter_complex",
                "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,volume=1.0[voice];"
                "[1:a]volume=0.15[bgm];"
                "[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[ao]",
                "-map", "[ao]", "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            ]
        tail = ["-r", f"{fps:.3f}", "-movflags", "+faststart", out_abs]

        s = get_settings()
        encoders = [s.video_encoder_args()]
        if s.use_nvenc:                    # GPU path can fail at runtime -> CPU fallback
            encoders.append(_X264)
        last: Exception | None = None
        for enc in encoders:
            try:
                ffmpeg.run([*base, *enc, *audio, *tail], timeout=900, cwd=tmp)
                last = None
                break
            except ffmpeg.FFmpegError as e:
                last = e
                log.warning("encode failed with %s; trying fallback", enc[1] if len(enc) > 1 else enc)
        if last is not None:
            raise last

    # Thumbnail with AI title overlay: extracts a frame from the rendered clip,
    # adds a bottom gradient, then overlays the clip's AI-generated title in bold
    # white text. Gives a "YouTube thumbnail" look without any external API.
    at = min(max(eff_dur * 0.35, 0.5), max(eff_dur - 0.1, 0.0))
    _make_thumbnail(out_path, thumb_path, at=at, width=540, duration=eff_dur,
                    title=getattr(clip, "title", None))

def _make_thumbnail(out_path: Path, thumb_path: Path, *, at: float,
                    width: int, title: str | None = None,
                    duration: float | None = None) -> None:
    """Extract a thumbnail frame and optionally overlay the clip title.

    When ``duration`` is known, samples frames across the clip and uses face
    detection to pick the frame with the strongest face presence. Falls back
    to the centre frame when no face is found, or to the ``at`` parameter
    when ``duration`` is not provided.
    """
    best_at = _pick_thumbnail_at(out_path, duration or 0.0, at, width)
    ffmpeg.make_thumbnail(out_path, thumb_path, at=best_at, width=width)
    if not title:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        img = Image.open(thumb_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)
        # Semi-transparent gradient at bottom for text legibility.
        grad_h = max(h // 3, 30)
        grad = np.zeros((grad_h, w, 4), dtype=np.uint8)
        for y in range(grad_h):
            alpha = int(180 * (1.0 - min(y / grad_h, 1.0)))
            grad[y, :, 3] = alpha
        gradient = Image.fromarray(grad, mode="RGBA")
        img = Image.alpha_composite(img.convert("RGBA"), gradient).convert("RGB")
        draw = ImageDraw.Draw(img)
        font_size = max(int(w * 0.065), 20)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        # Word-wrap title to thumbnail width (minus 24px padding).
        words = (title or "").split()
        lines, line = [], ""
        max_w = w - 32
        for wd in words:
            test = f"{line} {wd}" if line else wd
            bw = draw.textbbox((0, 0), test, font=font)[2]
            if bw > max_w and line:
                lines.append(line)
                line = wd
            else:
                line = test
        if line:
            lines.append(line)
        line_h = int(font_size * 1.35)
        total_h = len(lines) * line_h
        start_y = h - 20 - total_h
        for i, l in enumerate(lines):
            bw = draw.textbbox((0, 0), l, font=font)[2]
            tx = (w - bw) / 2
            draw.text((tx, start_y + i * line_h), l, fill="white",
                      font=font, stroke_width=3, stroke_color="black")
        img.save(thumb_path, quality=90)
    except Exception as e:
        log.debug("title overlay on thumbnail failed: %s", e)


def _pick_thumbnail_at(out_path: Path, duration: float, default_at: float,
                       width: int) -> float:
    """Sample frames across the clip and pick the one with the most face presence.

    Uses OpenCV + face detection (YOLOv8-face / YuNet / Haar cascade). Returns
    the center frame when face detection is unavailable or no faces are found.
    """
    if duration <= 0.5:
        return default_at

    try:
        import cv2
        import tempfile
        from ..media.faces import detect_faces
    except Exception:
        return default_at

    n_samples = 7
    timestamps = [duration * (i + 1) / (n_samples + 1) for i in range(n_samples)]
    best_score = -1.0
    best_t = duration * 0.5  # fallback: center frame

    for t in timestamps:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            ffmpeg.make_thumbnail(out_path, tmp_path, at=t, width=width)
            img = cv2.imread(str(tmp_path))
            tmp_path.unlink(missing_ok=True)
            if img is None:
                continue
            faces = detect_faces(img, min_size_frac=0.03)
            if faces:
                # Score = number of faces * average face area fraction
                h, w = img.shape[:2]
                total_area = sum(fw * fh for _, _, fw, fh in faces)
                score = len(faces) * (total_area / (w * h))
                if score > best_score:
                    best_score = score
                    best_t = t
        except Exception:
            continue

    return best_t
