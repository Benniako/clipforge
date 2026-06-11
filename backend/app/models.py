"""Domain models — the object vocabulary ClipForge reasons about.

These mirror PRD §6 (Core Objects): Project, Transcript, Clip, CaptionSet,
Reframe, StyleTemplate. They are plain Pydantic models so the same shapes are
used for storage (serialised to JSON), for the pipeline, and on the wire.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


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
    language: str = "en"
    speakers: int = 1
    provider: str = "synthetic"

    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# --------------------------------------------------------------------------- #
# Captions
# --------------------------------------------------------------------------- #
class CaptionWord(BaseModel):
    t: float                    # start relative to the CLIP (s)
    d: float
    text: str


class CaptionSet(BaseModel):
    id: str = Field(default_factory=lambda: _id("cap"))
    words: list[CaptionWord] = Field(default_factory=list)
    style_id: str = "bold-pop"
    # Words per on-screen line — keeps lines phone-readable. libass wraps any
    # line that is still too wide for the safe area (WrapStyle 0).
    max_words_per_line: int = 3


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
    tightened_duration: float | None = None
    feedback: str | None = None   # "up" | "down" | None (current user rating)
    export_url: str | None = None
    thumb_url: str | None = None
    error: str | None = None

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


class ImportSettings(BaseModel):
    platform: Platform = Platform.generic
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
    # Subtle camera motion: "none" or "push" (slow push-in across the clip).
    motion: str = "none"
    # Gameplay facecam handling: "auto" (stacked layout when a facecam is
    # found), "split" / "framed" to force a layout, "off" for plain crop.
    facecam_layout: str = "auto"

    def dims(self) -> tuple[int, int]:
        return ASPECTS.get(self.aspect, ASPECTS["9:16"])


class JobProgress(BaseModel):
    stage: str = "queued"
    stage_index: int = 0
    total_stages: int = 6
    message: str = ""
    pct: float = 0.0            # overall 0-100
    stages: list[dict] = Field(default_factory=list)  # per-stage status timeline
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
    warnings: list[str] = Field(default_factory=list)  # non-fatal issues for the UI
    error: str | None = None
    created_at: float = Field(default_factory=now)
    updated_at: float = Field(default_factory=now)

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
