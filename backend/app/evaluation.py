"""Offline evaluation: score detected events against hand-checked timestamps.

Detection quality cannot be judged by eye on a four-hour recording, and a
tweak that fixes one goal can silently lose three others. This module turns
"does it work?" into numbers: a *label file* lists the moments a human has
confirmed in a video, and the functions here compare any detector's output
against it (precision, recall, timing error, and the exact timestamps that
were missed or invented, so somebody can go and look).

Label file (JSON, version 1)::

    {
      "version": 1,
      "video": "match-day.mp4",              absolute, or relative to this file's folder
      "game_profile": "eafc",
      "status": "draft",                     "draft" until a human has checked it, then "verified"
      "duration": 13978.75,                  optional
      "labelled_ranges": [[0, 3600]],        optional; see below
      "events": [
        {"t": 159.2, "label": "goal", "team": "a", "score": "2-0", "minute": 11, "note": ""}
      ]
    }

Only ``t`` and ``label`` are required per event. When ``labelled_ranges`` is
present, only those time ranges were checked by a human: what a detector
reports outside them is neither right nor wrong and is left out of the numbers.

Everything here is plain Python (standard library only), so it can run in CI
and from the CLI without the web server, the database or any video tooling.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

LABEL_VERSION = 1
STATUSES = ("draft", "verified")
DEFAULT_TOLERANCE_S = 6.0
# A draft note is added below this tracker confidence. Clean reads sit at
# 0.85-0.95; goals seen in a single sample or inferred from a two-goal jump
# come out at 0.6 or less.
LOW_CONFIDENCE = 0.8
MAX_LISTED = 15

# Slack for float comparisons against the tolerance: label times are decimal
# (159.2 + 6 is not exactly 165.2 in binary), and an event "exactly at the
# tolerance" must count as matched.
_EPS = 1e-9
# Largest alignment table (truth x predictions inside one cluster of nearby
# events) solved exactly; beyond it a linear sweep keeps the match count
# maximal and gives up only on choosing the closest partners.
_DP_CELL_LIMIT = 2_000_000

_TOP_KEYS = ("version", "video", "game_profile", "status", "duration", "labelled_ranges", "events")
_REQUIRED_KEYS = ("version", "video", "game_profile", "status", "events")
_EVENT_KEYS = ("t", "label", "team", "score", "minute", "note")
_SCORE_RE = re.compile(r"^\d{1,3}-\d{1,3}$")


# --------------------------------------------------------------------------- #
# Label files
# --------------------------------------------------------------------------- #
@dataclass
class LabelEvent:
    """One hand-checked (or drafted) moment on the source timeline."""
    t: float                         # seconds from the start of the video
    label: str                       # e.g. "goal"
    team: str | None = None          # "a" = first team on the scoreboard, "b" = second
    score: str | None = None         # score after the event, "A-B"
    minute: int | None = None        # match minute shown on the clock
    note: str = ""


@dataclass
class LabelSet:
    video: str                       # absolute, or relative to the label file's folder
    game_profile: str = "auto"
    status: str = "draft"
    events: list[LabelEvent] = field(default_factory=list)
    duration: float | None = None
    # None = the whole video was checked. Otherwise only these [t0, t1] spans.
    labelled_ranges: list[tuple[float, float]] | None = None
    version: int = LABEL_VERSION

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unknown_keys(obj: dict, allowed: tuple[str, ...], where: str) -> None:
    for key in obj:
        if key not in allowed:
            raise ValueError(f"{where}unknown key {key!r} (allowed: {', '.join(allowed)})")


def labels_from_dict(data, *, source: str = "labels") -> LabelSet:
    """Validate a parsed label file and return it as a :class:`LabelSet`.

    Strict on purpose: a typo in a hand-edited file (``labeled_ranges``, a
    negative time, an event outside the checked ranges) would otherwise shift
    the numbers without anybody noticing. Every problem raises ``ValueError``
    naming the field — and the event index — at fault. ``source`` prefixes the
    message (the file name, when there is one).
    """
    def fail(msg: str):
        raise ValueError(f"{source}: {msg}")

    if not isinstance(data, dict):
        fail("the label file must be a JSON object")
    try:
        _unknown_keys(data, _TOP_KEYS, "")
    except ValueError as e:
        fail(str(e))
    for key in _REQUIRED_KEYS:
        if key not in data:
            fail(f"missing key {key!r}")

    version = data["version"]
    if not _is_int(version) or version != LABEL_VERSION:
        fail(f"version must be {LABEL_VERSION}, got {version!r}")
    video = data["video"]
    if not isinstance(video, str) or not video.strip():
        fail(f"video must be a non-empty path string, got {video!r}")
    profile = data["game_profile"]
    if not isinstance(profile, str) or not profile.strip():
        fail(f"game_profile must be a non-empty string, got {profile!r}")
    status = data["status"]
    if status not in STATUSES:
        fail(f"status must be one of {', '.join(STATUSES)}, got {status!r}")

    duration = data.get("duration")
    if duration is not None:
        if not _is_number(duration) or duration <= 0:
            fail(f"duration must be a positive number, got {duration!r}")
        duration = float(duration)

    ranges: list[tuple[float, float]] | None = None
    raw_ranges = data.get("labelled_ranges")
    if raw_ranges is not None:
        if not isinstance(raw_ranges, list):
            fail(f"labelled_ranges must be a list of [start, end] pairs, got {raw_ranges!r}")
        ranges = []
        for k, item in enumerate(raw_ranges):
            where = f"labelled_ranges[{k}]"
            if (not isinstance(item, (list, tuple)) or len(item) != 2
                    or not all(_is_number(v) for v in item)):
                fail(f"{where} must be a [start, end] pair of numbers, got {item!r}")
            t0, t1 = float(item[0]), float(item[1])
            if t0 < 0:
                fail(f"{where} starts before 0 ({t0:g})")
            if t1 <= t0:
                fail(f"{where} must end after it starts, got [{t0:g}, {t1:g}]")
            if duration is not None and t1 > duration + _EPS:
                fail(f"{where} ends at {t1:g}, beyond duration {duration:g}")
            if ranges and t0 < ranges[-1][1]:
                fail(f"{where} is unsorted or overlaps the range before it "
                     f"([{t0:g}, {t1:g}] after [{ranges[-1][0]:g}, {ranges[-1][1]:g}])")
            ranges.append((t0, t1))

    raw_events = data["events"]
    if not isinstance(raw_events, list):
        fail(f"events must be a list, got {type(raw_events).__name__}")
    events: list[LabelEvent] = []
    for k, item in enumerate(raw_events):
        where = f"events[{k}]"
        if not isinstance(item, dict):
            fail(f"{where} must be an object, got {item!r}")
        try:
            _unknown_keys(item, _EVENT_KEYS, f"{where}: ")
        except ValueError as e:
            fail(str(e))
        for key in ("t", "label"):
            if key not in item:
                fail(f"{where} is missing key {key!r}")
        t = item["t"]
        if not _is_number(t):
            fail(f"{where}.t must be a number of seconds, got {t!r}")
        if t < 0:
            fail(f"{where}.t must not be negative, got {t!r}")
        t = float(t)
        if duration is not None and t > duration + _EPS:
            fail(f"{where}.t is {t:g}, beyond duration {duration:g}")
        if ranges is not None and not _in_ranges(t, ranges):
            fail(f"{where}.t is {t:g}, outside labelled_ranges")
        label = item["label"]
        if not isinstance(label, str) or not label.strip():
            fail(f"{where}.label must be a non-empty string, got {label!r}")
        team = item.get("team")
        if team is not None and team not in ("a", "b"):
            fail(f"{where}.team must be \"a\" or \"b\", got {team!r}")
        score = item.get("score")
        if score == "":
            score = None
        if score is not None and (not isinstance(score, str) or not _SCORE_RE.match(score)):
            fail(f"{where}.score must look like \"2-1\", got {score!r}")
        minute = item.get("minute")
        if minute is not None and (not _is_int(minute) or minute < 0):
            fail(f"{where}.minute must be a whole number >= 0, got {minute!r}")
        note = item.get("note")
        if note is None:
            note = ""
        if not isinstance(note, str):
            fail(f"{where}.note must be a string, got {note!r}")
        events.append(LabelEvent(t=t, label=label.strip(), team=team, score=score,
                                 minute=minute, note=note))
    events.sort(key=lambda e: e.t)
    return LabelSet(video=video, game_profile=profile, status=status, events=events,
                    duration=duration, labelled_ranges=ranges, version=LABEL_VERSION)


def load_labels(path) -> LabelSet:
    """Read and validate a label file.

    Raises ``ValueError`` for anything wrong with its content (see
    :func:`labels_from_dict`) and ``OSError`` when the file cannot be read.
    Events come back sorted by time.
    """
    p = Path(path)
    raw = p.read_bytes()
    try:
        # utf-8-sig: Windows editors like to prepend a byte-order mark.
        data = json.loads(raw.decode("utf-8-sig"))
    except ValueError as e:
        raise ValueError(f"{p.name}: not a valid UTF-8 JSON file ({e})") from e
    return labels_from_dict(data, source=p.name)


def _num(value: float) -> float:
    return round(float(value), 3)


def labels_to_dict(labels: LabelSet) -> dict:
    """The label set as the plain dict that is written to disk.

    Events are sorted by time; optional fields that are unset are left out.
    """
    out: dict = {"version": labels.version, "video": labels.video,
                 "game_profile": labels.game_profile, "status": labels.status}
    if labels.duration is not None:
        out["duration"] = _num(labels.duration)
    if labels.labelled_ranges is not None:
        out["labelled_ranges"] = [[_num(a), _num(b)] for a, b in labels.labelled_ranges]
    events = []
    for ev in sorted(labels.events, key=lambda e: e.t):
        row: dict = {"t": _num(ev.t), "label": ev.label}
        if ev.team is not None:
            row["team"] = ev.team
        if ev.score:
            row["score"] = ev.score
        if ev.minute is not None:
            row["minute"] = ev.minute
        if ev.note:
            row["note"] = ev.note
        events.append(row)
    out["events"] = events
    return out


def dumps_labels(labels: LabelSet) -> str:
    """The label file text: 2-space indent, one event per line.

    One line per event keeps a hundred goals readable and makes a corrected
    timestamp a one-line diff. Raises ``ValueError`` when the set would not
    load back (the same checks as :func:`load_labels`).
    """
    data = labels_to_dict(labels)
    labels_from_dict(data)
    dump = lambda v: json.dumps(v, ensure_ascii=False)  # noqa: E731
    lines = [f"  {dump(key)}: {dump(data[key])}" for key in _TOP_KEYS
             if key in data and key != "events"]
    if data["events"]:
        rows = ",\n".join("    " + dump(row) for row in data["events"])
        lines.append(f"  \"events\": [\n{rows}\n  ]")
    else:
        lines.append("  \"events\": []")
    return "{\n" + ",\n".join(lines) + "\n}\n"


def save_labels(labels: LabelSet, path) -> Path:
    """Write a label file (UTF-8, ``\\n`` line ends, trailing newline).

    The output is stable — the same set always produces the same bytes — and
    the file is replaced in one step, so an interrupted write cannot leave a
    half-written file where hand-checked labels used to be.
    """
    p = Path(path)
    text = dumps_labels(labels)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, p)
    return p


def resolve_video(labels: LabelSet, label_path) -> Path:
    """Absolute path of the video a label file refers to.

    A relative ``video`` is taken relative to the folder holding the label
    file, so a label file can travel together with its recording.
    """
    video = Path(labels.video).expanduser()
    if not video.is_absolute():
        video = Path(os.path.abspath(label_path)).parent / video
    return Path(os.path.abspath(video))


def video_reference(video_path, label_path) -> str:
    """How a label file at ``label_path`` should refer to ``video_path``.

    Relative (with forward slashes) when the video sits in the label file's
    folder or below it, absolute otherwise.
    """
    video = Path(video_path).resolve()
    folder = Path(label_path).resolve().parent
    try:
        return video.relative_to(folder).as_posix()
    except ValueError:
        return video.as_posix()


def draft_from_hud(result, *, video: str, game_profile: str, duration: float | None) -> LabelSet:
    """Draft labels from a scoreboard-tracker result (``HudResult``).

    One event per tracked goal, time rounded to 0.1 s. The draft is a starting
    point for a human to check, never ground truth: its status is ``"draft"``
    and the notes point at the goals most worth a second look.
    """
    last_t = None
    if duration is not None and duration > 0:
        last_t = math.floor(float(duration) * 10.0) / 10.0
    events: list[LabelEvent] = []
    for ev in getattr(result, "events", None) or []:
        t = round(max(float(ev.t), 0.0), 1)
        if last_t is not None:
            t = min(t, last_t)
        notes: list[str] = []
        confidence = float(getattr(ev, "confidence", 1.0))
        if confidence < LOW_CONFIDENCE:
            notes.append(f"low confidence ({confidence:.2f})")
        if getattr(ev, "inferred", False):
            notes.append("inferred: one of two goals between two scoreboard readings, time is a guess")
        elif getattr(ev, "hidden", False):
            notes.append("scoreboard hidden around the goal, time is approximate")
        after = getattr(ev, "score_after", None)
        minute = getattr(ev, "minute", None)
        events.append(LabelEvent(
            t=t, label=getattr(ev, "label", "goal") or "goal", team=getattr(ev, "team", None),
            score=f"{int(after[0])}-{int(after[1])}" if after else None,
            minute=int(minute) if minute is not None else None,
            note="; ".join(notes)))
    events.sort(key=lambda e: e.t)
    return LabelSet(video=video, game_profile=game_profile, status="draft", events=events,
                    duration=_num(duration) if duration else None)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _in_ranges(t: float, ranges: Sequence[tuple[float, float]]) -> bool:
    return any(a - _EPS <= t <= b + _EPS for a, b in ranges)


def _sweep(a: list[float], b: list[float], limit: float) -> list[tuple[int, int]]:
    """Maximum one-to-one matching of two sorted lists, earliest partners first.

    The earliest unmatched point on either side can only pair with the
    earliest points on the other; if those are out of reach, nothing later is
    in reach either. So pairing greedily in time order never costs a match.
    """
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= limit:
            out.append((i, j))
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def _align(a: list[float], b: list[float], limit: float) -> list[tuple[int, int]]:
    """Maximum matching of two sorted lists with the smallest total distance.

    On a line a best matching never needs to cross (swapping the partners of
    two crossed pairs keeps both within the limit and does not add distance),
    so an order-preserving alignment table finds it exactly.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    if n * m > _DP_CELL_LIMIT:
        return _sweep(a, b, limit)
    # best[i][j] = (pairs, -total distance) using a[:i] and b[:j]
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, above = best[i], best[i - 1]
        for j in range(1, m + 1):
            cand = above[j] if above[j] >= row[j - 1] else row[j - 1]
            d = abs(a[i - 1] - b[j - 1])
            if d <= limit:
                paired = (above[j - 1][0] + 1, above[j - 1][1] - d)
                if paired >= cand:
                    cand = paired
            row[j] = cand
    out: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        d = abs(a[i - 1] - b[j - 1])
        if d <= limit and best[i][j] == (best[i - 1][j - 1][0] + 1, best[i - 1][j - 1][1] - d):
            out.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def match_events(truth_times: Sequence[float], predicted_times: Sequence[float],
                 tolerance: float) -> list[tuple[int, int]]:
    """Pair truth events with predictions, one-to-one, ``|dt| <= tolerance``.

    Returns ``(truth_index, predicted_index)`` pairs (indices into the inputs,
    which need not be sorted), ordered by truth time. The number of pairs is
    the maximum possible; among all such matchings the one with the smallest
    total ``|dt|`` is returned, so a prediction right on a goal is preferred to
    an earlier duplicate that merely falls inside the tolerance.

    The timeline is first cut wherever two neighbouring points (of either
    kind) are further apart than the tolerance — nothing can match across such
    a gap — which leaves a handful of points per cluster to align exactly.
    """
    tol = float(tolerance)
    if not tol >= 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")
    limit = tol + _EPS
    points = sorted([(float(t), 0, i) for i, t in enumerate(truth_times)]
                    + [(float(t), 1, j) for j, t in enumerate(predicted_times)])
    pairs: list[tuple[int, int]] = []
    group: tuple[list[tuple[float, int]], list[tuple[float, int]]] = ([], [])
    last: float | None = None

    def flush() -> None:
        truths, preds = group
        if truths and preds:
            for i, j in _align([t for t, _ in truths], [t for t, _ in preds], limit):
                pairs.append((truths[i][1], preds[j][1]))
        truths.clear()
        preds.clear()

    for t, kind, idx in points:
        if last is not None and t - last > limit:
            flush()
        group[kind].append((t, idx))
        last = t
    flush()
    return pairs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class LabelMetrics:
    """Detection quality for one label (or for all labels together)."""
    label: str
    truth: int                                # labelled events scored
    predicted: int                            # predictions scored (tp + fp)
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_abs_error: float | None              # seconds; None without matched pairs
    median_abs_error: float | None
    bias: float | None                        # mean of predicted - truth; > 0 = detected late
    missed: list[LabelEvent] = field(default_factory=list)    # truth with no prediction
    extra: list[LabelEvent] = field(default_factory=list)     # predictions with no truth
    pairs: list[tuple[float, float]] = field(default_factory=list)   # (truth t, predicted t)


@dataclass
class EventMetrics:
    tolerance: float
    overall: LabelMetrics
    per_label: dict[str, LabelMetrics]
    ignored_predictions: int = 0              # unmatched and outside the labelled ranges
    ignored_truth: int = 0                    # truth events outside the labelled ranges
    labelled_ranges: list[tuple[float, float]] | None = None


@dataclass
class ClipMetrics:
    clips: int                                # clips scored
    hits: int                                 # ... that contain at least one truth event
    clip_hit_rate: float
    events: int                               # truth events scored
    events_covered: int                       # ... that fall inside at least one clip
    event_coverage: float
    empty_clips: list[tuple[float, float]] = field(default_factory=list)
    uncovered: list[LabelEvent] = field(default_factory=list)
    ignored_clips: int = 0                    # entirely outside the labelled ranges


def _as_event(item) -> LabelEvent:
    if isinstance(item, LabelEvent):
        return item
    if isinstance(item, (tuple, list)):
        return LabelEvent(t=float(item[0]), label=str(item[1]))
    return LabelEvent(t=float(item.t), label=str(item.label))


def _ratio(hit: int, total: int) -> float:
    """``hit / total``, and 1.0 when there was nothing to get wrong."""
    return hit / total if total else 1.0


def _label_metrics(label: str, pairs: list[tuple[float, float]], missed: list[LabelEvent],
                   extra: list[LabelEvent]) -> LabelMetrics:
    tp, fp, fn = len(pairs), len(extra), len(missed)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    errors = [p - t for t, p in pairs]
    return LabelMetrics(
        label=label, truth=tp + fn, predicted=tp + fp, tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        mean_abs_error=sum(abs(e) for e in errors) / len(errors) if errors else None,
        median_abs_error=median(abs(e) for e in errors) if errors else None,
        bias=sum(errors) / len(errors) if errors else None,
        missed=sorted(missed, key=lambda e: e.t), extra=sorted(extra, key=lambda e: e.t),
        pairs=sorted(pairs))


def score_events(truth: Iterable, predicted: Iterable, *, tolerance: float = DEFAULT_TOLERANCE_S,
                 labelled_ranges: Sequence[tuple[float, float]] | None = None,
                 labels: Iterable[str] | None = None) -> EventMetrics:
    """Compare predicted events with the truth, label by label.

    ``truth`` holds :class:`LabelEvent` (or ``(t, label)``), ``predicted``
    holds ``(t, label)``. A prediction is right when it pairs with a truth
    event of the same label no more than ``tolerance`` seconds away (see
    :func:`match_events`); each event pairs at most once. ``labels`` restricts
    the comparison to those labels on both sides.

    With ``labelled_ranges``, only those spans were checked by a human. A
    prediction outside them that matches no truth event is ignored — neither
    right nor wrong — instead of being counted as a false positive. (One that
    does match, a second outside a range holding the goal it found, still
    counts as found.) Truth events outside the ranges are ignored as well.

    The empty cases are defined so that "nothing to get wrong" scores 1.0:

    * no truth and no predictions: precision = recall = 1.0 (f1 1.0)
    * truth but no predictions:    precision 1.0, recall 0.0 (f1 0.0)
    * predictions but no truth:    precision 0.0, recall 1.0 (f1 0.0)

    Timing figures are over the matched pairs: mean and median ``|dt|`` and
    the signed bias (mean of predicted minus truth; positive means the
    detector reports late). They are ``None`` when nothing matched.
    """
    tol = float(tolerance)
    ranges = [(float(a), float(b)) for a, b in labelled_ranges] if labelled_ranges is not None else None
    wanted = None if labels is None else list(dict.fromkeys(labels))
    truth_ev = [_as_event(e) for e in truth]
    pred_ev = [_as_event(e) for e in predicted]
    if wanted is not None:
        truth_ev = [e for e in truth_ev if e.label in wanted]
        pred_ev = [e for e in pred_ev if e.label in wanted]
    ignored_truth = 0
    if ranges is not None:
        kept = [e for e in truth_ev if _in_ranges(e.t, ranges)]
        ignored_truth = len(truth_ev) - len(kept)
        truth_ev = kept

    names = sorted({e.label for e in truth_ev} | {e.label for e in pred_ev} | set(wanted or ()))
    per_label: dict[str, LabelMetrics] = {}
    ignored = 0
    for name in names:
        ts = [e for e in truth_ev if e.label == name]
        ps = [e for e in pred_ev if e.label == name]
        matched = match_events([e.t for e in ts], [e.t for e in ps], tol)
        hit_t = {i for i, _ in matched}
        hit_p = {j for _, j in matched}
        extra = []
        for j, e in enumerate(ps):
            if j in hit_p:
                continue
            if ranges is not None and not _in_ranges(e.t, ranges):
                ignored += 1
            else:
                extra.append(e)
        per_label[name] = _label_metrics(
            name, [(ts[i].t, ps[j].t) for i, j in matched],
            [e for i, e in enumerate(ts) if i not in hit_t], extra)

    overall = _label_metrics(
        "all", [p for m in per_label.values() for p in m.pairs],
        [e for m in per_label.values() for e in m.missed],
        [e for m in per_label.values() for e in m.extra])
    return EventMetrics(tolerance=tol, overall=overall, per_label=per_label,
                        ignored_predictions=ignored, ignored_truth=ignored_truth,
                        labelled_ranges=ranges)


def score_clips(truth: Iterable, clips: Iterable[tuple[float, float]], *,
                labelled_ranges: Sequence[tuple[float, float]] | None = None,
                labels: Iterable[str] | None = None) -> ClipMetrics:
    """How well final clips, given as ``(start, end)``, line up with the truth.

    * ``clip_hit_rate`` — share of clips containing at least one truth event
      (how many clips are about something real);
    * ``event_coverage`` — share of truth events inside at least one clip
      (how much of what happened made it into a clip).

    ``labels`` limits which truth events count. With ``labelled_ranges``,
    clips lying entirely outside the checked spans are ignored, as are truth
    events outside them. As in :func:`score_events`, an empty side scores 1.0:
    no clips gives a hit rate of 1.0, no truth events a coverage of 1.0.
    """
    ranges = [(float(a), float(b)) for a, b in labelled_ranges] if labelled_ranges is not None else None
    wanted = None if labels is None else set(labels)
    events = [_as_event(e) for e in truth]
    if wanted is not None:
        events = [e for e in events if e.label in wanted]
    if ranges is not None:
        events = [e for e in events if _in_ranges(e.t, ranges)]
    spans = [(min(float(a), float(b)), max(float(a), float(b))) for a, b in clips]
    ignored = 0
    if ranges is not None:
        kept = [(a, b) for a, b in spans if any(a <= r1 and b >= r0 for r0, r1 in ranges)]
        ignored = len(spans) - len(kept)
        spans = kept
    spans.sort()
    empty = [(a, b) for a, b in spans if not any(a - _EPS <= e.t <= b + _EPS for e in events)]
    uncovered = [e for e in events if not _in_ranges(e.t, spans)]
    hits, covered = len(spans) - len(empty), len(events) - len(uncovered)
    return ClipMetrics(clips=len(spans), hits=hits, clip_hit_rate=_ratio(hits, len(spans)),
                       events=len(events), events_covered=covered,
                       event_coverage=_ratio(covered, len(events)), empty_clips=empty,
                       uncovered=sorted(uncovered, key=lambda e: e.t), ignored_clips=ignored)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_hms(t: float) -> str:
    """Seconds as ``H:MM:SS`` (whole seconds, rounded down like a player's clock)."""
    s = max(int(float(t)), 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _describe(ev: LabelEvent) -> str:
    parts = [format_hms(ev.t), ev.label]
    if ev.score:
        parts.append(ev.score)
    if ev.minute is not None:
        parts.append(f"{ev.minute}'")
    if ev.note:
        parts.append(f"({ev.note})")
    return "  ".join(parts)


def _listing(title: str, rows: list[str], limit: int) -> list[str]:
    if not rows:
        return []
    out = [title] + ["  " + r for r in rows[:limit]]
    if len(rows) > limit:
        out.append(f"  ... and {len(rows) - limit} more")
    return out


def format_report(metrics: EventMetrics, clips: ClipMetrics | None = None, *,
                  title: str | None = None, warnings: Sequence[str] = (),
                  max_listed: int = MAX_LISTED) -> str:
    """A compact plain-text report.

    Per-label table, a timing line, then up to ``max_listed`` missed and
    ``max_listed`` extra timestamps as ``H:MM:SS`` — the places to look at in
    the video — and, when ``clips`` is given, the clip figures.
    """
    lines: list[str] = []
    if title:
        lines.append(title)
    lines += [f"WARNING: {w}" for w in warnings]
    lines.append(f"tolerance: +/-{metrics.tolerance:g} s")

    rows = list(metrics.per_label.values())
    if len(rows) != 1:
        rows.append(metrics.overall)
    width = max([len("label")] + [len(m.label) for m in rows])
    lines.append("")
    lines.append(f"{'label':<{width}}  {'truth':>5}  {'pred':>5}  {'tp':>5}  {'fp':>5}  {'fn':>5}"
                 f"  {'precision':>9}  {'recall':>6}  {'f1':>5}")
    for m in rows:
        lines.append(f"{m.label:<{width}}  {m.truth:>5}  {m.predicted:>5}  {m.tp:>5}  {m.fp:>5}"
                     f"  {m.fn:>5}  {m.precision:>9.3f}  {m.recall:>6.3f}  {m.f1:>5.3f}")
    lines.append("")
    o = metrics.overall
    if o.tp:
        lines.append(f"timing: {o.tp} matched, mean |dt| {o.mean_abs_error:.2f} s, "
                     f"median |dt| {o.median_abs_error:.2f} s, bias {o.bias:+.2f} s "
                     f"(positive = detected late)")
    else:
        lines.append("timing: no matched events")
    if metrics.ignored_predictions:
        lines.append(f"ignored: {metrics.ignored_predictions} unmatched predictions "
                     f"outside the labelled ranges")
    if metrics.ignored_truth:
        lines.append(f"ignored: {metrics.ignored_truth} labelled events outside the labelled ranges")
    lines += _listing(f"missed ({o.fn} labelled, not detected):",
                      [_describe(e) for e in o.missed], max_listed)
    lines += _listing(f"extra ({o.fp} detected, not labelled):",
                      [_describe(e) for e in o.extra], max_listed)
    if clips is not None:
        lines.append("")
        lines.append(f"clips: {clips.clips} scored, {clips.hits} contain a labelled event "
                     f"(hit rate {clips.clip_hit_rate:.3f}); {clips.events_covered} of "
                     f"{clips.events} labelled events are inside a clip "
                     f"(coverage {clips.event_coverage:.3f})")
        if clips.ignored_clips:
            lines.append(f"ignored: {clips.ignored_clips} clips outside the labelled ranges")
        lines += _listing(f"empty clips ({len(clips.empty_clips)}):",
                          [f"{format_hms(a)} - {format_hms(b)}" for a, b in clips.empty_clips],
                          max_listed)
    return "\n".join(lines)


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _event_json(ev: LabelEvent) -> dict:
    out: dict = {"t": _num(ev.t), "time": format_hms(ev.t), "label": ev.label}
    if ev.score:
        out["score"] = ev.score
    if ev.minute is not None:
        out["minute"] = ev.minute
    if ev.note:
        out["note"] = ev.note
    return out


def _label_json(m: LabelMetrics) -> dict:
    return {
        "truth": m.truth, "predicted": m.predicted, "tp": m.tp, "fp": m.fp, "fn": m.fn,
        "precision": _r(m.precision), "recall": _r(m.recall), "f1": _r(m.f1),
        "mean_abs_error": _r(m.mean_abs_error, 3), "median_abs_error": _r(m.median_abs_error, 3),
        "bias": _r(m.bias, 3),
        "missed": [_event_json(e) for e in m.missed],
        "extra": [_event_json(e) for e in m.extra],
    }


def to_json(metrics: EventMetrics, clips: ClipMetrics | None = None, *,
            title: str | None = None, warnings: Sequence[str] = ()) -> dict:
    """The content of :func:`format_report` as a plain, JSON-serialisable dict.

    Unlike the text report, the missed and extra lists are complete.
    """
    out: dict = {
        "title": title,
        "warnings": list(warnings),
        "tolerance": metrics.tolerance,
        "overall": _label_json(metrics.overall),
        "labels": {name: _label_json(m) for name, m in metrics.per_label.items()},
        "ignored_predictions": metrics.ignored_predictions,
        "ignored_truth": metrics.ignored_truth,
        "labelled_ranges": ([[_num(a), _num(b)] for a, b in metrics.labelled_ranges]
                            if metrics.labelled_ranges is not None else None),
        "clips": None,
    }
    if clips is not None:
        out["clips"] = {
            "clips": clips.clips, "hits": clips.hits, "clip_hit_rate": _r(clips.clip_hit_rate),
            "events": clips.events, "events_covered": clips.events_covered,
            "event_coverage": _r(clips.event_coverage),
            "ignored_clips": clips.ignored_clips,
            "empty_clips": [{"start": _num(a), "end": _num(b),
                             "time": f"{format_hms(a)} - {format_hms(b)}"}
                            for a, b in clips.empty_clips],
            "uncovered": [_event_json(e) for e in clips.uncovered],
        }
    return out
