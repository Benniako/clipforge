"""ClipForge API application.

Wires the routers, serves generated media (with HTTP range support so clips
seek/scrub in the browser), reports environment capabilities, and starts the
background pipeline worker. If a built frontend is present at ``frontend/dist``
it is served at ``/`` so the whole product runs from one process.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown paths (client-side routes),
    while returning proper 404s for missing file assets (JS, CSS, images) so the
    browser doesn't choke on index.html served with a wrong MIME type.
    """

    _ASSET_EXTS = frozenset({".js", ".css", ".png", ".jpg", ".jpeg",
                             ".gif", ".svg", ".ico", ".woff", ".woff2",
                             ".ttf", ".webp", ".mp4", ".webm"})

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                # Missing JS/CSS/images should 404, not return index.html.
                if any(path.lower().endswith(ext) for ext in self._ASSET_EXTS):
                    raise
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404:
            if any(path.lower().endswith(ext) for ext in self._ASSET_EXTS):
                return response
            return await super().get_response("index.html", scope)
        return response

from . import feedback, store
from .config import get_settings
from .api import routes_clips, routes_cues, routes_projects
from .pipeline.orchestrator import engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("clipforge")

__version__ = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    feedback.init_db()
    # Clean up stale temp files from crashed pipelines (files older than 1 hour).
    _data = get_settings().data_dir
    for _p in _data.rglob("*.tmp"):
        try:
            if _p.stat().st_mtime < time.time() - 3600:
                _p.unlink(missing_ok=True)
        except Exception:
            pass
    for _p in _data.rglob("*_tmp.mp4"):
        try:
            if _p.stat().st_mtime < time.time() - 3600:
                _p.unlink(missing_ok=True)
        except Exception:
            pass
    engine.start()
    # Don't block the lifespan startup on resuming N stranded projects —
    # /api/ready would otherwise not answer until every store.mutate runs.
    # The worker pool is already up; resume_incomplete just feeds the queue.
    import threading
    threading.Thread(target=engine.resume_incomplete,
                     name="clipforge-resume", daemon=True).start()
    # Start the watch-folder poller if CLIPFORGE_WATCH_DIR is configured.
    from .pipeline.watcher import create_watcher
    watcher = create_watcher()
    if watcher:
        watcher.start()
    s = get_settings()
    logging.getLogger("clipforge").info("capabilities: %s", s.capability_report())
    yield
    # Graceful shutdown: let active ffmpeg encodes finish, then close DB
    # connections so WAL is checkpointed cleanly.
    log.info("shutting down — waiting for active renders…")
    from .pipeline.orchestrator import engine
    engine.wait_for_renders(timeout=60.0)
    store.close_all()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ClipForge API", version=__version__, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        from .providers import audio_events, llm, vlm
        caps = settings.capability_report()
        caps.update(audio_events.capability_flags())
        # Keep first-page health cheap. Settings already collected Ollama tags
        # at startup; avoid fresh /api/tags probes just to paint the nav strip.
        llm_model = llm.model_from_tags(settings.ollama_models)
        vlm_model = vlm.model_from_tags(settings.ollama_models)
        caps["llm"] = bool(llm_model)
        caps["llm_model"] = llm_model
        caps["vlm"] = bool(vlm_model)
        caps["vlm_model"] = vlm_model
        return {
            "ok": True,
            "version": __version__,
            "capabilities": caps,
            "output": {"width": settings.out_width, "height": settings.out_height},
        }

    @app.get("/api/ready", tags=["meta"])
    def ready() -> dict:
        """Fast readiness probe for launch scripts.

        /api/health intentionally checks optional AI backends, which can take a
        few seconds when Ollama or audio detectors are waking up. Startup scripts
        only need to know that FastAPI finished booting and can serve the SPA.
        """
        return {"ok": True, "version": __version__}

    @app.get("/api/capabilities", tags=["meta"])
    def capabilities() -> dict:
        """Structured inventory of what ClipForge detected installed.

        Returns two views:
        - ``flat``: the legacy boolean/string map (back-compat for /api/health
          consumers), now extended with deno, ollama, torchaudio, and per-OCR-engine flags.
        - ``detail``: a grouped, human-readable breakdown with an ``impact`` line
          per item explaining what each capability unlocks (or what degrades
          when it's absent). Used by the UI's diagnostics panel.
        """
        s = settings
        return {"flat": s.capability_report(), "detail": s.capability_detail()}

    app.include_router(routes_projects.router)
    app.include_router(routes_clips.router)
    app.include_router(routes_cues.router)

    # Generated media (sources, clips, thumbnails). StaticFiles serves Range
    # requests, so <video> scrubbing works out of the box.
    app.mount("/media", StaticFiles(directory=str(settings.media_dir)), name="media")

    # Optionally serve a built SPA so `uvicorn app.main:app` runs the whole thing.
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        # Check whether the built SPA may be stale relative to the source,
        # and auto-build if so — the developer doesn't have to remember.
        src = dist.parent / "src"
        if src.is_dir():
            dist_mtime = max((p.stat().st_mtime for p in dist.rglob("*") if p.is_file()), default=0)
            src_mtime = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0)
            if src_mtime > dist_mtime:
                import subprocess
                print("[INFO] Frontend source is newer than dist/. Auto-building…")
                ret = subprocess.run(
                    ["npx", "vite", "build"],
                    cwd=str(dist.parent), capture_output=True, text=True, timeout=120,
                )
                if ret.returncode == 0:
                    print("[OK] Frontend rebuilt.")
                else:
                    print(f"[WARN] Auto-build failed (exit {ret.returncode}). "
                          "Stale frontend may be served. Run `npx vite build` manually.")
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="spa")

    return app


app = create_app()
