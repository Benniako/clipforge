"""Suggested hashtags per clip (PRD §4 Could-have).

Talking clips: pull the most salient content words from the clip's own words and
mix in a couple of format/platform tags. Gameplay clips: gaming-oriented tags,
with the game guessed from any commentary if it's named.
"""
from __future__ import annotations

import re
from collections import Counter

_STOP = set((
    "the a an and or but so to of in on for with that this it is are was were be "
    "you your my our we they he she his her them i me as at by from up out if then "
    "than just like really very can will would could should have has had do does did "
    "not no yes about into over under more most some any all one two get got go "
    "und oder aber das die der ein eine ich du wir sie es ist sind war auf für mit "
    "von zu im in den dem ja nein nicht auch noch schon sehr mal was wie wenn dann"
).split())

_PLATFORM = {
    "tiktok": ["#tiktok", "#fyp"],
    "reels": ["#reels", "#instagram"],
    "shorts": ["#shorts", "#youtubeshorts"],
    "generic": ["#shorts"],
}

_GAMES = {
    "valorant": "#valorant", "valo": "#valorant", "fifa": "#eafc", "eafc": "#eafc",
    "fc": "#eafc", "fortnite": "#fortnite", "warzone": "#warzone", "cs": "#cs2",
    "counter": "#cs2", "league": "#leagueoflegends", "apex": "#apexlegends",
    "minecraft": "#minecraft", "overwatch": "#overwatch",
}

_WORD = re.compile(r"[a-zA-ZäöüÄÖÜß]{4,}")


def _slug(word: str) -> str:
    return "#" + re.sub(r"[^a-z0-9]", "", word.lower())


# Known game profiles -> their primary tag (feeds bucket clips by game, so
# the game-specific tag should lead — generic #gaming reaches no one).
_PROFILE_TAGS = {
    "valorant": "#valorant", "cs2": "#cs2", "cs": "#cs2", "eafc": "#eafc",
    "fifa": "#eafc", "rocketleague": "#rocketleague", "horror": "#horrorgaming",
}


def suggest_hashtags(text: str, *, content_type: str, platform: str,
                     limit: int = 7, game: str | None = None) -> list[str]:
    tags: list[str] = []
    toks = [w.lower() for w in _WORD.findall(text or "")]

    if content_type == "gameplay":
        g = _PROFILE_TAGS.get((game or "").lower().replace(" ", ""))
        if g:
            tags += [g, g + "clips"]
        tags += ["#gaming", "#gameplay", "#highlights"]
        for t in toks:
            if t in _GAMES and _GAMES[t] not in tags:
                tags.append(_GAMES[t])
    else:
        # top content keywords from the clip's words
        freq = Counter(t for t in toks if t not in _STOP)
        for word, _ in freq.most_common(4):
            tag = _slug(word)
            if tag not in tags and len(tag) > 2:
                tags.append(tag)

    for p in _PLATFORM.get(platform, _PLATFORM["generic"]):
        if p not in tags:
            tags.append(p)

    # de-dupe, cap
    seen: list[str] = []
    for t in tags:
        if t not in seen:
            seen.append(t)
    return seen[:limit]
