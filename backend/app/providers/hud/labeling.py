"""Give every glyph cluster its digit.

Two independent sources of evidence are combined:

1. **OCR votes.** A handful of members of each cluster are read by the OCR
   engine. Agreeing, confident reads fix ("anchor") the label.
2. **The rule of the game.** Within a match a score digit only ever changes
   to its successor. So if the cluster that follows the "2" cluster is not yet
   known, it is a "3" — no OCR needed. This fills the digits an engine cannot
   read (Tesseract returns nothing for a bold 0, 4, 5 or 6) and lets the
   tracker work with no OCR engine at all, anchored on the digit every match
   starts from.

Pure Python apart from the image handed to the reader.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Callable

from .glyphs import Cluster, ocr_image
from .readers import clean_digits

log = logging.getLogger("clipforge.hud.labeling")

VOTES_PER_CLUSTER = 5
MIN_MEMBERS = 2          # a cluster seen once cannot hold a score; skip it
ANCHOR_WEIGHT = 1.2      # summed confidence needed to fix a label from OCR alone
ANCHOR_RATIO = 3.0       # ... and how far it must lead the runner-up
EDGE_CAP = 50            # one transition seen many times counts at most this much
LONG_GAP_WEIGHT = 0.25   # a change across a scoreboard absence: goal or rematch reset


def cluster_runs(seq: list[int], times: list[float], *, min_n: int = 2
                 ) -> list[tuple[int, float, float, int]]:
    """Stable stretches of one cluster id in a per-sample sequence.

    ``seq`` holds a cluster id per sample, or -1 when the cell was unreadable.
    Returns ``(cluster, t_first, t_last, n)`` for stretches of at least
    ``min_n`` samples, with unreadable samples ignored and neighbours of the
    same cluster merged once a shorter blip between them has been dropped.
    """
    raw: list[list] = []
    for cid, t in zip(seq, times):
        if cid < 0:
            continue
        if raw and raw[-1][0] == cid:
            raw[-1][2] = t
            raw[-1][3] += 1
        else:
            raw.append([cid, t, t, 1])
    out: list[list] = []
    for r in raw:
        if r[3] < min_n:
            continue
        if out and out[-1][0] == r[0]:
            out[-1][2] = r[2]
            out[-1][3] += r[3]
        else:
            out.append(list(r))
    return [(r[0], r[1], r[2], r[3]) for r in out]


def transition_edges(seqs: dict[str, list[int]], times: list[float], *,
                     max_goal_gap_s: float = 90.0,
                     direct_gap_s: float | None = None) -> dict[tuple[int, int], float]:
    """How often cluster X was followed by cluster Y in one cell (weighted).

    A change seen from one sample to the next, with the scoreboard on screen
    throughout, is a goal for certain and counts 1. A change across a longer
    absence may be a goal behind a celebration or the reset of a quick
    rematch, so it only counts ``LONG_GAP_WEIGHT``.

    Pairs seen in both directions about equally are two looks of the same
    digit (the HUD fading in and out), not a score change, and are dropped; a
    score never goes back, so a real transition is essentially one-way.
    """
    if direct_gap_s is None:
        gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        direct_gap_s = max(3.0 * (gaps[len(gaps) // 2] if gaps else 1.0), 12.0)
    counts: dict[tuple[int, int], float] = defaultdict(float)
    for seq in seqs.values():
        runs = cluster_runs(seq, times)
        for r1, r2 in zip(runs, runs[1:]):
            gap = r2[1] - r1[2]
            if r1[0] != r2[0] and gap <= max_goal_gap_s:
                counts[(r1[0], r2[0])] += 1.0 if gap <= direct_gap_s else LONG_GAP_WEIGHT
    edges: dict[tuple[int, int], float] = {}
    for (x, y), n in counts.items():
        if n >= 3 * counts.get((y, x), 0.0):
            edges[(x, y)] = n
    return edges


def collect_votes(clusters: list[Cluster], bitmap_of: Callable[[tuple[int, str]], object],
                  reader, *, votes: int = VOTES_PER_CLUSTER) -> int:
    """Read a few members of each cluster; returns the number of OCR calls."""
    calls = 0
    if reader is None:
        return calls
    for c in clusters:
        if c.n < MIN_MEMBERS:
            continue
        k = min(votes, len(c.members))
        picks = [c.members[int(round(j * (len(c.members) - 1) / max(k - 1, 1)))] for j in range(k)]
        for ref in dict.fromkeys(picks):
            bm = bitmap_of(ref)
            if bm is None:
                continue
            try:
                text, conf = reader.read_digits(ocr_image(bm))
            except Exception as e:            # an OCR hiccup must not sink the scan
                log.debug("hud: digit read failed: %s", e)
                text, conf = "", 0.0
            calls += 1
            value = clean_digits(text)
            if value is None:
                c.empty_votes += 1
            else:
                c.votes[value] = c.votes.get(value, 0.0) + max(float(conf), 0.05)
    return calls


def _anchor(c: Cluster) -> None:
    if not c.votes:
        return
    ranked = sorted(c.votes.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ranked[0]
    second_w = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_w >= ANCHOR_WEIGHT and top_w >= ANCHOR_RATIO * second_w:
        c.label, c.anchor, c.source = top, True, "ocr"


def label_clusters(clusters: list[Cluster], seqs: dict[str, list[int]], times: list[float],
                   bitmap_of: Callable[[tuple[int, str]], object], reader, *,
                   max_goal_gap_s: float = 90.0) -> dict:
    """Label ``clusters`` in place; returns diagnostics."""
    calls = collect_votes(clusters, bitmap_of, reader)
    for c in clusters:
        _anchor(c)
    edges = transition_edges(seqs, times, max_goal_gap_s=max_goal_gap_s)

    anchored_by_rule = False
    if not any(c.label is not None for c in clusters):
        # No OCR at all: every match starts from the same digit, and (apart
        # from the odd quick rematch) nothing ever turns into it. The cluster
        # that is left far more often than it is entered is the zero.
        balance: Counter = Counter()
        for (x, y), n in edges.items():
            balance[x] += n
            balance[y] -= n
        cands = [cid for cid, bal in balance.items() if bal > 0 and clusters[cid].n >= MIN_MEMBERS]
        if cands:
            zero = max(cands, key=lambda cid: (balance[cid], clusters[cid].n))
            clusters[zero].label, clusters[zero].anchor, clusters[zero].source = 0, True, "rule"
            anchored_by_rule = True

    for _ in range(12):
        changed = False
        for cid, c in enumerate(clusters):
            if c.anchor or c.n < MIN_MEMBERS:
                continue
            tally: dict[int, float] = defaultdict(float)
            for value, w in c.votes.items():
                tally[value] += w
            for (x, y), n in edges.items():
                w = float(min(n, EDGE_CAP))
                if y == cid and clusters[x].label is not None:
                    tally[clusters[x].label + 1] += w
                if x == cid and clusters[y].label is not None and clusters[y].label >= 1:
                    tally[clusters[y].label - 1] += w
            if not tally:
                continue
            best = max(sorted(tally), key=lambda v: tally[v])
            if best != c.label:
                c.label = best
                c.source = "mixed" if c.votes else "rule"
                changed = True
        if not changed:
            break

    # A transition that is neither "+1" nor a reset to zero (a quick rematch)
    # contradicts the labels; a healthy scan has none.
    violations = sum(n for (x, y), n in edges.items()
                     if clusters[x].label is not None and clusters[y].label is not None
                     and clusters[y].label not in (clusters[x].label + 1, 0))
    return {
        "ocr_calls": calls,
        "edges": {f"{x}->{y}": round(n, 2) for (x, y), n in sorted(edges.items())},
        "violations": int(round(violations)),
        "anchored_by_rule": anchored_by_rule,
        "labels": [{"id": i, "n": c.n, "label": c.label, "source": c.source,
                    "votes": {str(k): round(v, 2) for k, v in sorted(c.votes.items())},
                    "empty": c.empty_votes}
                   for i, c in enumerate(clusters) if c.n >= MIN_MEMBERS],
    }
