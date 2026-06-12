"""Cue-pack management endpoints — add/remove reference game sounds from the UI.

Lets the user paste a sound URL (e.g. a MyInstants link) or upload a file and
have it installed as a matching cue — no command line. All local.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .. import game_packs

router = APIRouter(prefix="/api/cues", tags=["cues"])


@router.get("")
def list_cues() -> dict:
    return game_packs.pack_status()


@router.post("/{game}/{event}")
async def add_cue(game: str, event: str,
                  url: str | None = Form(None),
                  file: UploadFile | None = File(None)) -> dict:
    """Install a cue for <game>/<event> from a URL or an uploaded file."""
    if not url and not file:
        raise HTTPException(400, "provide a sound url or file")
    try:
        if file is not None:
            suffix = Path(file.filename or "cue").suffix or ".bin"
            # Close the fd mkstemp opens, or Windows refuses the unlink below
            # while the handle is held ([WinError 32]).
            fd, tmp_name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                with open(tmp, "wb") as out:
                    while chunk := await file.read(1 << 20):
                        out.write(chunk)
                await run_in_threadpool(game_packs.install_cue, game, event, str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
        else:
            await run_in_threadpool(game_packs.install_cue_from_url, game, event, url)
    except Exception as e:
        raise HTTPException(400, f"could not install cue: {e}")
    return game_packs.pack_status()


@router.delete("/{game}/{event}")
def delete_cue(game: str, event: str) -> dict:
    game_packs.remove_cue(game, event)
    return game_packs.pack_status()
