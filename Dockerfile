# ClipForge — multi-stage Docker image
#
# Stage 1: Build the React frontend
# Stage 2: Python runtime with the built frontend + backend
#
# Usage:
#   docker build -t clipforge .
#   docker run -p 8000:8000 -v clipforge-data:/app/backend/data clipforge
#
# With NVIDIA GPU acceleration:
#   docker compose up  (see docker-compose.yml)

# ---------------------------------------------------------------
# Stage 1 — Frontend build (Node 22)
# ---------------------------------------------------------------
FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend

# Dependency caching layer
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy source and build
COPY frontend/ .
RUN npm run build

# ---------------------------------------------------------------
# Stage 2 — Python runtime
# ---------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Install runtime system deps:
#   - libass (caption rendering via libass in ffmpeg)
#   - ffmpeg (media engine — may also be supplied by static-ffmpeg wheels)
#   - build-essential (for pip wheels that compile native code)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libass-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pyproject.toml is optional; COPY succeeds even if the source is missing
# by checking with shell in a RUN step instead.
COPY backend/ ./backend/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Install Python deps
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir httpx ruff pillow 2>/dev/null || true

# Default env — can be overridden at runtime
ENV CLIPFORGE_DATA_DIR=/app/backend/data \
    CLIPFORGE_MAX_UPLOAD_MB=0 \
    UVICORN_PORT=8000

# Volume for persistent data (SQLite DB + uploaded media + rendered clips)
VOLUME ["/app/backend/data"]

EXPOSE 8000

# Health check (lightweight — just checks the server is running)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/ready')" || exit 1

# Default: run the API server (frontend is served by the app itself via SPAStaticFiles)
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
