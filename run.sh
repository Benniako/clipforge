#!/usr/bin/env bash
# Start ClipForge (single server: API + built UI) on http://localhost:8000
cd "$(dirname "$0")"
if [ -f .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
if [ ! -x "$VPY" ] && [ ! -f "$VPY" ]; then echo "Run ./setup.sh first."; exit 1; fi
# Tuned for this machine (Ryzen 5 7600 + RTX 5060 Ti 16GB); override by exporting first.
export CLIPFORGE_RENDER_WORKERS="${CLIPFORGE_RENDER_WORKERS:-4}"
export CLIPFORGE_WHISPER_MODEL="${CLIPFORGE_WHISPER_MODEL:-large-v3-turbo}"
export CLIPFORGE_WHISPER_BATCH="${CLIPFORGE_WHISPER_BATCH:-24}"
export CLIPFORGE_DEFAULT_POWER_MODE="${CLIPFORGE_DEFAULT_POWER_MODE:-max_gpu}"
export CLIPFORGE_OLLAMA_MODEL="qwen3:8b"
export HF_TOKEN="${HF_TOKEN:-HF_TOKEN_REVOKED}"
export CLIPFORGE_ASD_DIR="${CLIPFORGE_ASD_DIR:-C:\Users\benni\Documents\Codex\2026-06-15\use-github-or-my-uploaded-code\work\clipforge\backend\data\models\LR-ASD}"
# Add deno + tesseract to PATH so the capability detector finds them.
export PATH="$PATH:/c/Users/benni/AppData/Local:/c/Program Files/Tesseract-OCR"
if command -v ollama >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    (ollama serve >/dev/null 2>&1 &)
    sleep 2
  fi
fi
echo "ClipForge running at http://localhost:8000  (Ctrl+C to stop)"
exec "$VPY" -m uvicorn app.main:app --app-dir backend --port 8000
