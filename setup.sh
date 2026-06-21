#!/usr/bin/env bash
# ClipForge one-time setup for macOS / Linux (and Git Bash on Windows).
set -e
cd "$(dirname "$0")"
echo "=== ClipForge setup ==="

# 1. Find a supported Python (3.10-3.12; the ML libs don't support 3.13/3.14 yet)
PY=""
for v in python3.12 python3.11 python3.10; do
  if command -v "$v" >/dev/null 2>&1; then PY="$v"; break; fi
done
if [ -z "$PY" ]; then
  echo "[X] Python 3.10-3.12 not found. Install Python 3.12, then re-run."
  exit 1
fi
echo "[OK] Python: $PY"

command -v npm >/dev/null 2>&1 || { echo "[X] Node.js not found - install the LTS from https://nodejs.org"; exit 1; }

# 2. venv + backend deps (use the venv's python directly; no activation needed)
"$PY" -m venv .venv
if [ -f .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r backend/requirements.txt

# 2b. NVIDIA GPU runtime (auto-detected): Whisper-on-GPU needs cuBLAS/cuDNN;
#     the pip wheels provide them without a system CUDA install.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected - installing CUDA runtime libraries for GPU transcription..."
  "$VPY" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 \
    || echo "[!] CUDA libraries failed to install - transcription will run on CPU."
fi

# 2c. Optional AI power-ups (VAD captions, OCR, scene detect, emotion, YOLO
#     reframe, whisperX). Best-effort: a failed/conflicting wheel is logged and
#     skipped, never aborting setup (the core pipeline runs without them).
echo "Installing optional AI power-ups (large download; failures are skipped)..."
while IFS= read -r pkg; do
  case "$pkg" in ''|\#*) continue;; esac          # skip blanks/comments
  echo "  -> $pkg"
  "$VPY" -m pip install "$pkg" || echo "  [..] skipped $pkg (install failed/conflict)"
done < backend/requirements-extras.txt

# 3. (optional) YuNet face model — much better facecam/face detection than the
#    Haar fallback. Skipped silently when offline; everything still works.
MODEL=backend/data/models/face_detection_yunet_2023mar.onnx
if [ ! -f "$MODEL" ] && command -v curl >/dev/null 2>&1; then
  mkdir -p backend/data/models
  curl -fsSL --max-time 30 -o "$MODEL" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
    && echo "[OK] YuNet face model" \
    || { rm -f "$MODEL"; echo "[..] YuNet model skipped (offline?) - using Haar fallback"; }
fi

# 4. Optional Ollama models (best hardware-fit defaults when Ollama exists)
if command -v ollama >/dev/null 2>&1; then
  echo "Setting up local Ollama models (optional but recommended)..."
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    (ollama serve >/dev/null 2>&1 &)
    sleep 3
  fi
  VRAM_MB=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')
    VRAM_MB=${VRAM_MB:-0}
  fi
  RAM_GB=0
  if command -v python3 >/dev/null 2>&1; then
    RAM_GB=$(python3 - <<'PY'
import os
try:
    pages = os.sysconf("SC_PHYS_PAGES")
    size = os.sysconf("SC_PAGE_SIZE")
    print(round(pages * size / (1024**3)))
except Exception:
    print(0)
PY
)
  fi
  if [ "$VRAM_MB" -ge 20000 ] && [ "$RAM_GB" -ge 48 ]; then
    VISION=qwen2.5vl:32b; VFALL=qwen2.5vl:7b; TEXT=qwen3:32b; TFALL=qwen3:14b
  elif [ "$VRAM_MB" -ge 10000 ] && [ "$RAM_GB" -ge 24 ]; then
    VISION=qwen2.5vl:7b; VFALL=qwen2.5vl:3b; TEXT=qwen3:14b; TFALL=qwen3:8b
  elif [ "$VRAM_MB" -ge 6000 ] && [ "$RAM_GB" -ge 16 ]; then
    VISION=qwen2.5vl:7b; VFALL=qwen2.5vl:3b; TEXT=qwen3:8b; TFALL=qwen3:4b
  else
    VISION=qwen2.5vl:3b; VFALL=llava:7b; TEXT=qwen3:4b; TFALL=llama3.2:3b
  fi
  echo "Best default models: vision=$VISION, text=$TEXT"
  ollama pull "$VISION" || ollama pull "$VFALL" || true
  ollama pull "$TEXT" || ollama pull "$TFALL" || true
else
  echo "[..] Ollama not found. Install it from https://ollama.com for AI titles/vision."
fi

# 4b. User-supplied Valorant cue pack. Best-effort so setup still completes
#     if the soundboard site is temporarily unavailable.
echo "Installing Valorant reference cues (optional)..."
"$VPY" scripts/install_valorant_cues.py \
  || echo "[..] Valorant cues skipped - rerun scripts/install_valorant_cues.py later."

# 5. build the web UI
( cd frontend && npm install && npm run build )

echo
echo "=== Setup complete. Launching ClipForge... ==="
# Launch the app right away (you asked for run at the end). Skip with NO_RUN=1.
if [ -z "${NO_RUN:-}" ] && [ -x ./run.sh ]; then
  exec ./run.sh
fi
