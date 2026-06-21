"""Persistence for projects.

A project is a nested document (clips, transcript, captions, reframe paths), so
we store it as one JSON blob per row in SQLite. SQLite gives us durability and
atomic writes for free; the JSON column keeps the rich object graph intact
without an ORM. A new connection is opened per operation (WAL mode) which keeps
the store safe to use from the API threadpool and the background worker alike.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

from .config import get_settings
from .models import Project, ProjectSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_created ON projects(created_at);
"""

# Serialise read-modify-write cycles on a single project. v1 is single-user, so
# one process-wide lock is simpler than per-id locks and plenty fast.
_write_lock = threading.RLock()


def init_db() -> None:
    with _connect() as con:
        con.executescript(_SCHEMA)


@contextmanager
def _connect():
    con = sqlite3.connect(get_settings().db_path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def save(project: Project) -> Project:
    from .models import now

    project.updated_at = now()
    with _write_lock, _connect() as con:
        con.execute(
            "INSERT INTO projects (id, status, created_at, updated_at, data) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at, data=excluded.data",
            (project.id, project.status.value, project.created_at,
             project.updated_at, project.model_dump_json()),
        )
    return project


def get(project_id: str) -> Project | None:
    with _connect() as con:
        row = con.execute(
            "SELECT data FROM projects WHERE id=?", (project_id,)
        ).fetchone()
    return Project.model_validate_json(row[0]) if row else None


def list_summaries(limit: int = 100) -> list[ProjectSummary]:
    with _connect() as con:
        rows = con.execute(
            "SELECT data FROM projects ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [ProjectSummary.of(Project.model_validate_json(r[0])) for r in rows]


def delete(project_id: str) -> bool:
    with _write_lock, _connect() as con:
        cur = con.execute("DELETE FROM projects WHERE id=?", (project_id,))
        deleted = cur.rowcount > 0  # read before the connection closes
    return deleted


@contextmanager
def mutate(project_id: str):
    """Load → yield → save a project atomically under the write lock.

    Usage::

        with mutate(pid) as p:
            p.status = ProjectStatus.ready
    """
    with _write_lock:
        project = get(project_id)
        if project is None:
            raise KeyError(project_id)
        yield project
        save(project)
