#!/usr/bin/env bash
# Start ClipForge (single server: API + built UI) on http://localhost:8000
cd "$(dirname "$0")"
if [ -f .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
if [ ! -x "$VPY" ] && [ ! -f "$VPY" ]; then echo "Run ./setup.sh first."; exit 1; fi
# Tuned for this machine (Ryzen 5 7600 + RTX 5060 Ti 16GB); override by exporting first.
export CLIPFORGE_RENDER_WORKERS="${CLIPFORGE_RENDER_WORKERS:-6}"
export CLIPFORGE_WHISPER_MODEL="${CLIPFORGE_WHISPER_MODEL:-large-v3-turbo}"
export CLIPFORGE_WHISPER_BATCH="${CLIPFORGE_WHISPER_BATCH:-48}"
export CLIPFORGE_DEFAULT_POWER_MODE="${CLIPFORGE_DEFAULT_POWER_MODE:-max_gpu}"
export CLIPFORGE_OLLAMA_MODEL="qwen3:8b"
# HF_TOKEN: set this in your environment or .env file to enable speaker diarization.
# Suppress HF symlink warning (symlinks need Windows Developer Mode).
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
# Add deno + tesseract to PATH so the capability detector finds them.
DENO_PATH="/c/Users/benni/AppData/Local"
TESS_PATH="/c/Program Files/Tesseract-OCR"
if [ -d "$DENO_PATH" ] && [[ ":$PATH:" != *":$DENO_PATH:"* ]]; then export PATH="$DENO_PATH:$PATH"; fi
if [ -d "$TESS_PATH" ] && [[ ":$PATH:" != *":$TESS_PATH:"* ]]; then export PATH="$TESS_PATH:$PATH"; fi
if command -v ollama >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    (ollama serve >/dev/null 2>&1 &)
    # Retry loop — Ollama may take >2s on a slow machine or first launch.
    for i in $(seq 1 10); do
      if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi
fi
echo "ClipForge running at http://localhost:8000  (Ctrl+C to stop)"
exec "$VPY" -m uvicorn app.main:app --app-dir backend --port 8000
