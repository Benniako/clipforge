"""Domain models — the object vocabulary ClipForge reasons about.

These mirror PRD §6 (Core Objects): Project, Transcript, Clip, CaptionSet,
Reframe, StyleTemplate. They are plain Pydantic models so the same shapes are
used for storage (serialised to JSON), for the pipeline, and on the wire.
"""
from __future__ import annotations

import time
import uuid
import os
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class ProjectStatus(str, Enum):
    created = "created"          # source accepted, not yet queued
    queued = "queued"
    processing = "processing"
    paused = "paused"
    ready = "ready"             # clips available
    failed = "failed"


class ClipStatus(str, Enum):
    pending = "pending"         # detected, not yet rendered
    rendering = "rendering"
    ready = "ready"
    failed = "failed"


class Platform(str, Enum):
    tiktok = "tiktok"
    reels = "reels"
    shorts = "shorts"
    generic = "generic"


class PowerMode(str, Enum):
    balanced = "balanced"      # good default: fast without monopolising the PC
    max_gpu = "max_gpu"        # saturate local GPU/CPU for batch creation
    quality = "quality"        # slower reads with more visual context


def _default_power_mode() -> PowerMode:
    try:
        return PowerMode(os.environ.get("CLIPFORGE_DEFAULT_POWER_MODE", "balanced"))
    except ValueError:
        return PowerMode.balanced


class LayoutType(str, Enum):
    fill = "fill"               # full-bleed, speaker-tracked
    center = "center"           # static center crop
    split = "split"             # facecam strip on top, gameplay below (vstack)
    framed = "framed"           # facecam PiP overlaid on full-bleed gameplay


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #
class Word(BaseModel):
    t: float                    # start (s, source timeline)
    d: float                    # duration (s)
    text: str
    speaker: int | None = None

    @property
    def end(self) -> float:
        return self.t + self.d


class Transcript(BaseModel):
    words: list[Word] = Field(default_factory=list)
    language: str = "de"
    speakers: int = 1
    provider: str = "synthetic"
    # Absolute source-time speech spans from VAD when available. These let
    # captions clear on real voice activity even if ASR word boxes are loose.
    speech: list[list[float]] = Field(default_factory=list)

    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
class CaptionWord(BaseModel):
    t: float                    # start relative to the CLIP (s)
    d: float
    text: str
    # Which diarized speaker said this word (0-based). None when diarization
    # didn't run (single speaker). Lets the editor toggle a speaker's lines
    # in/out of the burned captions.
    speaker: int | None = None
    # A "power word" (hook/emotion/payoff/number) — rendered in the highlight
    # colour + slightly larger for the whole line, not only while spoken. This
    # is the keyword-emphasis look that reads as professionally edited.
    emphasis: bool = False
    # An optional emoji appended after the word (auto-picked from its meaning),
    # capped to a couple per line so captions stay tasteful, not spammy.
    emoji: str | None = None


class CaptionSet(BaseModel):
    id: str = Field(default_factory=lambda: _id("cap"))
    words: list[CaptionWord] = Field(default_factory=list)
    # Clip-relative speech spans, copied from Transcript.speech for this clip.
    # Renderers use these only to shorten overlong caption display.
    speech: list[list[float]] = Field(default_factory=list)
    style_id: str = "bold-pop"
    # Words per on-screen line — keeps lines phone-readable. libass wraps any
    # line that is still too wide for the safe area (WrapStyle 0).
    max_words_per_line: int = 3
    # Spoken language of these words ("en"/"de"), used to pick the right power-word
    # lexicon for keyword emphasis + emoji at render time.
    lang: str = "en"


# --------------------------------------------------------------------------- #
# Detected events (audio cues + on-screen OCR) — saved per project
# --------------------------------------------------------------------------- #
class Notice(BaseModel):
    """A non-fatal issue surfaced to the UI, with a severity so the front-end
    can style it (and an optional code for targeted handling). Plain strings
    coming from storage or older code are coerced to ``severity="warn"``."""
    message: str
    severity: str = "warn"      # "info" | "warn" | "error"
    code: str | None = None


class DetectedEvent(BaseModel):
    """A pinpointed moment found in the source: a matched audio cue or a piece
    of viral on-screen text (OCR). Persisted on the project so the user can see
    exactly what ClipForge keyed off, and so OCR hits can be promoted to cues."""
    t: float                    # source timeline (s)
    source: str                 # "cue" | "ocr" | "audio"
    label: str                  # canonical event name, e.g. "kill", "victory"
    detail: str = ""            # raw text / cue file matched
    confidence: float = 0.0     # 0..1


class OcrRead(BaseModel):
    """One non-empty OCR observation retained for scan debugging."""
    t: float
    roi: str
    text: str = ""
    confidence: float = 0.0
    matched: list[str] = Field(default_factory=list)


class OcrReport(BaseModel):
    """Bounded OCR scan telemetry surfaced in the UI.

    Detector candidates can be discarded later when clips are scored. This report
    keeps enough raw evidence to explain why a configured visual cue did or did
    not fire without storing every crop from long videos.
    """
    enabled: bool = False
    engine: str = ""
    status: str = "skipped"     # "skipped" | "unavailable" | "ran" | "failed"
    frames_sampled: int = 0
    crops_read: int = 0
    cache_hits: int = 0
    texts_found: int = 0
    matches: int = 0
    warnings: list[str] = Field(default_factory=list)
    reads: list[OcrRead] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reframe (16:9 -> 9:16 crop path)
# --------------------------------------------------------------------------- #
class Rect(BaseModel):
    """A region of the source frame, all values fractions [0,1] of its size."""
    x: float
    y: float
    w: float
    h: float

    def clamped(self) -> "Rect":
        w = min(max(self.w, 0.02), 1.0)
        h = min(max(self.h, 0.02), 1.0)
        return Rect(x=min(max(self.x, 0.0), 1.0 - w),
                    y=min(max(self.y, 0.0), 1.0 - h), w=w, h=h)


class GameProfileConfig(BaseModel):
    """Optional per-project cue tuning for gameplay detection.

    The built-in detectors stay automatic, but this lets a project carry custom
    OCR boxes, visual phrases, and CLAP prompt hints without hard-coding them
    into a global game profile.
    """
    detection_mode: str = "zero_shot"  # "zero_shot" | "manual" | "hybrid"
    visual_rois: list[Rect] = Field(default_factory=list)
    visual_text_cues: list[str] = Field(default_factory=list)
    reference_audio_files: list[str] = Field(default_factory=list)
    vlm_visual_prompts: list[str] = Field(default_factory=lambda: [
        "victory screen",
        "defeat screen",
        "kill feed",
        "skull icon",
    ])
    audio_prompts: list[str] = Field(default_factory=list)
    audio_negative_prompts: list[str] = Field(default_factory=lambda: [
        "mouse clicking",
        "UI menu navigation",
        "keyboard typing",
        "quiet lobby music",
        "loading screen ambience",
    ])


class ReframeKeyframe(BaseModel):
    t: float                    # time relative to the CLIP (s)
    cx: float                   # crop-window centre X as a fraction [0,1] of src width


class Reframe(BaseModel):
    layout: LayoutType = LayoutType.fill
    # crop window aspect is fixed 9:16; we only need the horizontal path.
    keyframes: list[ReframeKeyframe] = Field(default_factory=list)
    tracked: bool = False       # True if a real subject track drove the path
    overridden: bool = False    # True once the user edits ANY reframe aspect
    # True only when the user set a manual crop centre — layout/facecam edits
    # also flip `overridden`, so the editor needs this to tell them apart.
    cx_overridden: bool = False
    # Facecam region (source-frame fractions) for split/framed layouts.
    facecam: Rect | None = None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class ScoreFactor(BaseModel):
    label: str                  # human-readable, e.g. "Strong hook in first 3s"
    weight: float               # contribution to the headline score (points)
    detail: str = ""


# --------------------------------------------------------------------------- #
# Clip
# --------------------------------------------------------------------------- #
class Clip(BaseModel):
    id: str = Field(default_factory=lambda: _id("clip"))
    start: float                # source timeline (s)
    end: float
    title: str = ""
    description: str = ""       # AI-generated or user-written post description
    kind: str = "speech"       # "speech" | "gameplay" — how it was detected
    score: int = 0             # 0-100
    factors: list[ScoreFactor] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)  # signal values (for learning)
    hashtags: list[str] = Field(default_factory=list)
    transcript_excerpt: str = ""
    captions: CaptionSet = Field(default_factory=CaptionSet)
    reframe: Reframe = Field(default_factory=Reframe)
    # Per-clip output aspect (an ASPECTS key); None = the project default.
    aspect: str | None = None
    status: ClipStatus = ClipStatus.pending
    # Raw detector boundaries before any learned correction — lets the learner
    # measure how far the user actually trims from the detector's output.
    raw_start: float | None = None
    raw_end: float | None = None
    # Jump-cut segments (absolute source times) when silence-tightening applied;
    # empty = render the plain start..end span.
    segments: list[list[float]] = Field(default_factory=list)
    # Absolute source spans whose words are kept OUT of captions — in-game
    # announcer/agent lines located by audio-cue matches.
    caption_mute: list[list[float]] = Field(default_factory=list)
    # Speakers (0-based diarization ids) whose words are burned into captions.
    # None = show every speaker (the default). Set via the editor's per-speaker
    # toggles; words from omitted speakers are dropped from the caption set.
    caption_speakers: list[int] | None = None
    # All diarized speakers present in this clip's span — drives the editor's
    # toggle chips (kept even when a speaker is muted out of the captions).
    speakers: list[int] = Field(default_factory=list)
    tightened_duration: float | None = None
    feedback: str | None = None   # "up" | "down" | None (current user rating)
    export_url: str | None = None
    thumb_url: str | None = None
    error: str | None = None
    # B-roll overlay: a PiP cutaway from a strong visual moment in the source,
    # displayed during a static talking span. ``start_rel`` is offset from the
    # clip's start; the overlay lasts ``duration`` seconds.
    broll_overlay: dict | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


class Montage(BaseModel):
    """Several clips stitched into one video, scored on its own virality."""

    id: str = Field(default_factory=lambda: _id("mtg"))
    title: str = "Montage"
    clip_ids: list[str] = Field(default_factory=list)
    score: int = 0
    factors: list[ScoreFactor] = Field(default_factory=list)
    duration: float = 0.0
    status: ClipStatus = ClipStatus.pending
    export_url: str | None = None
    thumb_url: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Style templates (reusable caption looks)
# --------------------------------------------------------------------------- #
class StyleTemplate(BaseModel):
    id: str
    name: str
    font: str = "DejaVu Sans"
    font_size: int = 92
    primary: str = "FFFFFF"     # base text colour (RRGGBB)
    highlight: str = "F5C518"   # active-word colour
    outline: str = "000000"
    outline_w: int = 6
    # vertical anchor as a fraction of height (0=top .. 1=bottom); ~0.78 sits
    # in the safe zone above platform UI.
    y_frac: float = 0.78
    uppercase: bool = True
    # Keyword emphasis: colour + enlarge "power words" across the whole line
    # (Submagic/Hormozi look). Off keeps only the spoken word highlighted.
    emphasis: bool = True
    # Auto-emoji: append a tasteful emoji to matched power words (max ~2/line).
    emoji: bool = False


# --------------------------------------------------------------------------- #
# Project
# --------------------------------------------------------------------------- #
class ContentType(str, Enum):
    auto = "auto"          # detect talking vs gameplay automatically
    talking = "talking"    # podcast/interview/talk -> transcript-driven moments
    gameplay = "gameplay"  # games -> audio-energy/event-driven highlights


# Output aspect ratios. 9:16/4:5/1:1 vertical/square for socials; 16:9 horizontal
# for YouTube or for re-editing in a desktop NLE (Premiere/Resolve).
ASPECTS: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


class AiBoostSettings(BaseModel):
    """Per-project AI Boost — the viral-effect toggles that control the
    production-value passes (caption emphasis/emoji, speaker colours, auto-zoom,
    B-roll, hook check). Each maps to a toggle in the upload panel's AI Boost
    group so the user sees the whole "make it look pro" surface in one place."""
    emphasis: bool = True       # Keyword colour+scale across the line
    emoji: bool = True           # Tasteful auto-emoji on power words
    speakerColors: bool = True  # Per-speaker caption colour (podcast look)
    autoZoom: bool = True       # Punch-in zoom on emphasis words/cuts
    broll: bool = False         # Smart cutaway B-roll (opt-in; changes the cut)
    hookCheck: bool = True      # First-3s hook strength warning


class ImportSettings(BaseModel):
    platform: Platform = Platform.generic
    power_mode: PowerMode = Field(default_factory=_default_power_mode)
    min_len: float = 15.0
    max_len: float = 60.0
    target_clips: int = 10
    default_style_id: str = "bold-pop"
    # Spoken-language hint for transcription: "de" (default — German-first), "en",
    # or "auto". Detection/scoring pick their lexicon from the transcript language.
    language: str = "de"
    # What kind of footage this is — drives which detector runs.
    content_type: ContentType = ContentType.auto
    # Output aspect ratio key (see ASPECTS).
    aspect: str = "9:16"
    # Burn captions into the video. Turn off to get clean clips for editing in a
    # desktop NLE (Premiere/Resolve) where you'd add your own captions.
    burn_captions: bool = True
    # Game profile tunes gameplay highlight detection. "auto"/"generic" work for
    # any game; the named ones bias the audio/scene signals per genre.
    game_profile: str = "auto"
    # Remove dead air inside talking clips (social-style jump cuts).
    tighten: bool = False
    # Isolate the voice from background music/game audio (Demucs) so captions and
    # speech sound studio-clean. Optional power-up; no-op without Demucs installed.
    denoise: bool = False
    # Subtle camera motion: "none" or "push" (slow push-in across the clip).
    motion: str = "none"
    # Gameplay facecam handling: "auto" (stacked layout when a facecam is
    # found), "split" / "framed" to force a layout, "off" for plain crop.
    facecam_layout: str = "auto"
    # Visible detector switches. They default on so existing projects keep the
    # previous best-effort behaviour, while the UI can now make them explicit.
    use_ocr: bool = True
    use_vlm: bool = True
    use_cues: bool = True
    use_audio_events: bool = True
    cue_learning: bool = True
    # AI Boost — per-project production-value toggles that drive the
    # caption_fx emphasis/emoji, speaker colours, auto-zoom, B-roll cutaways,
    # and hook-check passes. On by default: they ship the "professionally
    # edited" Submagic/OpusClip look at import time. Set per-project in the
    # upload panel's AI Boost group.
    ai_boost: AiBoostSettings = Field(default_factory=AiBoostSettings)
    # Let ClipForge pick a platform/content tuned range instead of the manual
    # length preset.
    auto_length: bool = False
    # Gameplay event padding: seconds kept before/after a detected cue, OCR hit,
    # or audio-event window. None keeps the profile default.
    lead_seconds: float | None = None
    tail_seconds: float | None = None
    # Project-local cue configuration for custom CLAP prompts and OCR ROIs.
    game_config: GameProfileConfig = Field(default_factory=GameProfileConfig)
    # Optional background music track overlaid on the clip audio.
    # Path to an audio file on the local filesystem, or empty string for no music.
    # When set, ffmpeg mixes the music at low volume during render.
    background_music: str = ""

    def dims(self) -> tuple[int, int]:
        return ASPECTS.get(self.aspect, ASPECTS["9:16"])


class JobProgress(BaseModel):
    stage: str = "queued"
    stage_index: int = 0
    total_stages: int = 6
    message: str = ""
    pct: float = 0.0            # overall 0-100
    stages: list[dict] = Field(default_factory=list)  # per-stage status timeline
    started_at: float | None = None   # when real processing began (for ETA)
    updated_at: float = Field(default_factory=now)


class SourceMedia(BaseModel):
    filename: str
    path: str                   # relative to media dir
    url: str | None = None      # original import URL, if any
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    size_bytes: int = 0


class Project(BaseModel):
    id: str = Field(default_factory=lambda: _id("proj"))
    name: str = "Untitled"
    status: ProjectStatus = ProjectStatus.created
    settings: ImportSettings = Field(default_factory=ImportSettings)
    source: SourceMedia | None = None
    transcript: Transcript | None = None
    clips: list[Clip] = Field(default_factory=list)
    montages: list[Montage] = Field(default_factory=list)
    progress: JobProgress = Field(default_factory=JobProgress)
    content_type: str | None = None       # detected/used: "talking" | "gameplay"
    facecam: Rect | None = None           # detected streamer cam region, if any
    # Accepted audio-cue/OCR/audio events tied to final clips, sorted by time.
    # Raw detector candidates stay internal so the UI does not overstate proof.
    events: list[DetectedEvent] = Field(default_factory=list)
    ocr_report: OcrReport = Field(default_factory=OcrReport)
    warnings: list[Notice] = Field(default_factory=list)  # non-fatal issues for the UI
    error: str | None = None
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)

    @field_validator("warnings", mode="before")
    @classmethod
    def _coerce_warnings(cls, v):
        """Accept legacy/raw strings (from stored JSON or string-appending code)
        and lift them to Notice objects so the field has one shape."""
        if not isinstance(v, list):
            return v
        out = []
        for item in v:
            if isinstance(item, str):
                out.append({"message": item, "severity": "warn"})
            else:
                out.append(item)
        return out

    def add_warning(self, message: str, *, severity: str = "warn",
                    code: str | None = None) -> None:
        """Append a notice, de-duplicating by message."""
        if any(n.message == message for n in self.warnings):
            return
        self.warnings.append(Notice(message=message, severity=severity, code=code))

    def clip(self, clip_id: str) -> Clip | None:
        return next((c for c in self.clips if c.id == clip_id), None)

    def montage(self, montage_id: str) -> "Montage | None":
        return next((m for m in self.montages if m.id == montage_id), None)


# --------------------------------------------------------------------------- #
# API request/response helpers
# --------------------------------------------------------------------------- #
class ProjectSummary(BaseModel):
    """Lightweight projection for list views (no transcript / heavy fields)."""

    id: str
    name: str
    status: ProjectStatus
    clip_count: int
    ready_clips: int
    duration: float
    progress: JobProgress
    created_at: float
    updated_at: float

    @classmethod
    def of(cls, p: Project) -> "ProjectSummary":
        return cls(
            id=p.id, name=p.name, status=p.status,
            clip_count=len(p.clips),
            ready_clips=sum(1 for c in p.clips if c.status == ClipStatus.ready),
            duration=p.source.duration if p.source else 0.0,
            progress=p.progress, created_at=p.created_at, updated_at=p.updated_at,
        )
