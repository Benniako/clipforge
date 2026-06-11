"""ClipForge API application.

Wires the routers, serves generated media (with HTTP range support so clips
seek/scrub in the browser), reports environment capabilities, and starts the
background pipeline worker. If a built frontend is present at ``frontend/dist``
it is served at ``/`` so the whole product runs from one process.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException


class SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for unknown paths (client-side routes).

    Without this, refreshing the browser on e.g. /p/<project-id> returns a 404
    in the single-server setup. /api and /media are mounted first, so they are
    never swallowed by the fallback.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response

from . import feedback, store
from .config import get_settings
from .api import routes_clips, routes_cues, routes_projects
from .pipeline.orchestrator import engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

__version__ = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    feedback.init_db()
    engine.start()
    s = get_settings()
    logging.getLogger("clipforge").info("capabilities: %s", s.capability_report())
    yield


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
        from .providers import llm
        caps = settings.capability_report()
        caps["llm"] = llm.available()
        return {
            "ok": True,
            "version": __version__,
            "capabilities": caps,
            "output": {"width": settings.out_width, "height": settings.out_height},
        }

    app.include_router(routes_projects.router)
    app.include_router(routes_clips.router)
    app.include_router(routes_cues.router)

    # Generated media (sources, clips, thumbnails). StaticFiles serves Range
    # requests, so <video> scrubbing works out of the box.
    app.mount("/media", StaticFiles(directory=str(settings.media_dir)), name="media")

    # Optionally serve a built SPA so `uvicorn app.main:app` runs the whole thing.
    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", SPAStaticFiles(directory=str(dist), html=True), name="spa")

    return app


app = create_app()
