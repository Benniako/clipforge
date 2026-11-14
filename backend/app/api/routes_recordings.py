"""The user's recordings, for importing by path instead of uploading.

Same-origin only like the rest of the API (CORS admits the dev server alone),
so other websites cannot read the list.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from .. import footage
from ..config import get_settings

router = APIRouter(prefix="/api", tags=["import"])


@router.get("/recordings")
async def list_recordings() -> list[dict]:
    """Recent videos in Videos, OBS's recording folders and CLIPFORGE_FOOTAGE_DIRS."""
    def scan() -> list[dict]:
        dirs = footage.footage_dirs(get_settings().footage_dirs)
        return [r.to_dict() for r in footage.list_recordings(dirs)]

    return await run_in_threadpool(scan)
