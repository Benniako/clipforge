@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo(
echo ==========================================
echo    ClipForge - one-time setup
echo ==========================================
echo(

REM --- 1. Find a supported Python (3.10-3.12; NOT 3.13/3.14) -------------
set "PYL="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYL (
        py -%%V -c "import sys" >nul 2>&1 && set "PYL=py -%%V"
    )
)
if not defined PYL (
    echo [X] Python 3.10-3.12 was not found.
    echo     ClipForge's AI libraries don't support Python 3.13/3.14 yet.
    echo       1^) Install Python 3.12: https://www.python.org/downloads/release/python-3120/
    echo          ^(tick "Add python.exe to PATH" on the first screen^)
    echo       2^) Run setup.bat again.
    echo(
    pause
    exit /b 1
)
echo [OK] Python: !PYL!

REM --- 2. Node.js (for the web UI) --------------------------------------
where npm >nul 2>&1
if errorlevel 1 (
    echo [X] Node.js was not found.
    echo     Install the LTS version from https://nodejs.org , then run setup.bat again.
    pause
    exit /b 1
)
echo [OK] Node.js found

REM --- 3. VC++ runtime (Whisper / ctranslate2 needs it) ----------------
if not exist "%SystemRoot%\System32\vcruntime140.dll" (
    echo [!] Microsoft VC++ Redistributable looks missing - transcription may fail.
    echo     If captions come out as placeholder text, install:
    echo       https://aka.ms/vs/17/release/vc_redist.x64.exe
)

REM --- 4. Virtual env + backend dependencies ---------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating an isolated Python environment ^(.venv^)...
    !PYL! -m venv .venv
)
set "VPY=.venv\Scripts\python.exe"
echo Installing Python packages - this takes a few minutes...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r backend\requirements.txt
if errorlevel 1 ( echo [X] Backend install failed. & pause & exit /b 1 )

REM --- 4b. NVIDIA GPU runtime (auto-detected) ---------------------------
REM Whisper-on-GPU needs cuBLAS/cuDNN; the pip wheels provide them without a
REM system CUDA install. Skipped entirely on machines without an NVIDIA GPU.
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo NVIDIA GPU detected - installing CUDA runtime libraries for GPU transcription...
    "%VPY%" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    if errorlevel 1 echo [!] CUDA libraries failed to install - transcription will run on CPU.
)

REM --- 4d. Optional AI power-ups (VAD/OCR/scene/emotion/YOLO/whisperX) --
REM Best-effort: each line installs on its own; a failed or conflicting wheel
REM is skipped (the core pipeline runs without them). Large download.
echo Installing optional AI power-ups ^(large download; failures are skipped^)...
for /f "usebackq eol=# tokens=*" %%P in ("backend\requirements-extras.txt") do (
    echo   -^> %%P
    "%VPY%" -m pip install %%P || echo   [..] skipped %%P ^(install failed/conflict^)
)

REM --- 4e. Hugging Face token for WhisperX diarization ------------------
REM Token is private, so setup guides you through creating one, validates
REM pyannote access, then stores HF_TOKEN in your Windows user environment.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_hf_token.ps1" -PythonExe "%VPY%"

REM --- 4e2. PANNs audio-event checkpoint --------------------------------
REM panns-inference assumes wget exists; on Windows we download its checkpoint
REM with PowerShell so audio-event detection is actually usable.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_panns.ps1" -PythonExe "%VPY%" || echo [..] PANNs checkpoint skipped - CLAP/other detectors still run.

REM --- 4f. LR-ASD active-speaker checkout -------------------------------
REM Optional: clones the local active-speaker model adapter and records its path.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_active_speaker.ps1" -PythonExe "%VPY%"

REM --- 4c. YuNet face model (optional, better facecam detection) --------
REM Skipped silently when offline; the Haar fallback still works.
if not exist "backend\data\models\face_detection_yunet_2023mar.onnx" (
    mkdir backend\data\models 2>nul
    curl -fsSL --max-time 30 -o "backend\data\models\face_detection_yunet_2023mar.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" >nul 2>&1 && (
        echo [OK] YuNet face model
    ) || (
        del /q "backend\data\models\face_detection_yunet_2023mar.onnx" 2>nul
        echo [..] YuNet model skipped - using Haar fallback
    )
)

REM --- 4g. Local Ollama models (optional, best hardware-fit defaults) ----
REM Installs Ollama with winget when possible, starts it, and pulls the most
REM powerful text + vision models that fit the detected GPU/RAM. Failures are
REM skipped; ClipForge still runs with heuristic titles/scores.
echo Setting up local AI models ^(Ollama / Qwen; optional but recommended^)...
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_ollama_models.ps1"

REM --- 4h. User-supplied Valorant cue pack ------------------------------
REM Installs the local reference sounds from scripts\install_valorant_cues.py
REM into backend\data\game_cues\valorant. Failures are skipped so setup still
REM finishes if the soundboard site is unavailable.
echo Installing Valorant reference cues ^(optional^)...
"%VPY%" "%~dp0scripts\install_valorant_cues.py" || echo [..] Valorant cues skipped - you can rerun scripts\install_valorant_cues.py later.

REM --- 5. Build the web UI --------------------------------------------
echo Building the web interface...
pushd frontend
call npm install
if errorlevel 1 ( echo [X] npm install failed. & popd & pause & exit /b 1 )
call npm run build
if errorlevel 1 ( echo [X] npm run build failed. & popd & pause & exit /b 1 )
popd

echo(
echo ==========================================
echo    Setup complete!  Launching ClipForge...
echo ==========================================
REM You asked for run.bat at the end — launch it now.
call "%~dp0run.bat"
