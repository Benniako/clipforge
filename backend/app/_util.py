"""Internal utilities shared across pipeline and providers.

Extracted from duplicated patterns:
- ``http_download`` — 3 call sites (ingest, game_packs, faces) with the same
  ``User-Agent: ClipForge/0.1`` + ``urlopen(timeout=N)`` + chunked write loop.
- ``run_subprocess`` — 5 call sites that each hand-roll the same
  ``subprocess.run(capture_output=True, text=True, timeout=…)`` + error logging.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

log = logging.getLogger("clipforge.util")

_DEFAULT_UA = "ClipForge/0.1"


def http_download(url: str, dst: str | Path, *,
                  timeout: float = 60.0,
                  user_agent: str = _DEFAULT_UA,
                  cap_bytes: int | None = None) -> Path:
    """Stream a URL to a local file, respecting a byte cap.

    Returns the destination path. Raises ``ValueError`` if ``cap_bytes`` is
    exceeded (callers use this for upload-size enforcement).
    """
    dst = Path(dst)
    # Support both (no existing file) and (initially empty temp file) callers.
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dst, "wb") as f:
        if cap_bytes is None:
            shutil.copyfileobj(resp, f)
        else:
            size = 0
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > cap_bytes:
                    raise ValueError("download exceeds the size limit")
                f.write(chunk)
    return dst


def run_subprocess(cmd: list[str], *,
                   timeout: float = 60.0,
                   check: bool = True,
                   cwd: str | None = None,
                   env: dict[str, str] | None = None,
                   capture_output: bool = True,
                   log_label: str | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess with uniform error handling.

    All the "run a binary with timeout/capture/error-logging" sites that
    duplicated this pattern converge on this one helper.
    """
    label = log_label or cmd[0] if cmd else "<subprocess>"
    try:
        proc = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log.warning("%s timed out after %.0fs", label, timeout)
        raise
    except Exception as e:
        log.warning("%s failed: %s", label, e)
        raise

    if check and proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-8:])
        log.warning("%s exited %d:\n%s", label, proc.returncode, tail)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)

    return proc