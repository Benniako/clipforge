"""Scoreboard (HUD) tracking: read a game's score off the screen.

    from app.providers.hud import track_scoreboard
    result = track_scoreboard(path, media_info, settings)
    for goal in result.events: ...

See :mod:`.detect` for how it works.
"""
from .detect import HudEvent, HudResult, MatchInfo, goal_importance, track_scoreboard
from .layouts import LAYOUTS, Cell, HudLayout, layouts_for

__all__ = ["HudEvent", "HudResult", "MatchInfo", "goal_importance", "track_scoreboard",
           "LAYOUTS", "Cell", "HudLayout", "layouts_for"]
