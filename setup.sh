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

# 4. build the web UI
( cd frontend && npm install && npm run build )

echo
echo "=== Setup complete. Start it with: ./run.sh ==="
