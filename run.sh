#!/usr/bin/env bash
# Start ClipForge (single server: API + built UI) on http://localhost:8000
cd "$(dirname "$0")"
if [ -f .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
if [ ! -x "$VPY" ] && [ ! -f "$VPY" ]; then echo "Run ./setup.sh first."; exit 1; fi
# Tuned for this machine (Ryzen 5 7600 + RTX 5060 Ti 16GB); override by exporting first.
export CLIPFORGE_RENDER_WORKERS="${CLIPFORGE_RENDER_WORKERS:-4}"
export CLIPFORGE_WHISPER_MODEL="${CLIPFORGE_WHISPER_MODEL:-large-v3-turbo}"
echo "ClipForge running at http://localhost:8000  (Ctrl+C to stop)"
exec "$VPY" -m uvicorn app.main:app --app-dir backend --port 8000
