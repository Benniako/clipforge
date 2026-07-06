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

REM --- 2b. Deno JS runtime (yt-dlp YouTube player parsing) --------------
REM Installed locally into .tools\deno so setup does not require admin rights.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_deno.ps1" || echo [..] Deno setup skipped - YouTube imports may be capped at 360p.

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

REM --- 4b. PyTorch + NVIDIA runtime (auto-detected) ----------------------
REM Current faster-whisper/CTranslate2 builds use CUDA 12 + cuDNN 9. This
REM helper installs matching PyTorch CUDA wheels and pip-provided NVIDIA DLLs,
REM with CPU fallbacks if any optional acceleration package fails.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_python_accel.ps1" -PythonExe "%VPY%" || echo [..] Python acceleration setup had warnings - CPU fallbacks remain available.

REM --- 4d. Optional AI power-ups (VAD/OCR/scene/emotion/YOLO/whisperX) --
REM Best-effort: each line installs on its own; a failed or conflicting wheel
REM is skipped (the core pipeline runs without them). Large download.
echo Installing optional AI power-ups ^(large download; failures are skipped^)...
for /f "usebackq eol=# tokens=*" %%P in ("backend\requirements-extras.txt") do (
    echo   -^> %%P
    "%VPY%" -m pip install %%P || echo   [..] skipped %%P ^(install failed/conflict^)
)

REM Optional packages such as ultralytics may install opencv-python. OpenCV's
REM Python wheels all share the cv2 namespace, so normalize back to the single
REM headless OpenCV 5 wheel that ClipForge expects.
echo Normalizing OpenCV 5 runtime...
"%VPY%" -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless >nul 2>&1
"%VPY%" -m pip install --force-reinstall --no-deps "opencv-python-headless>=5.0,<6" || echo [!] OpenCV 5 normalization failed - face tracking may be unavailable.

REM Optional packages can pull CPU PyTorch wheels from PyPI. Re-apply the
REM hardware-matched acceleration stack after extras so the final environment
REM is the one ClipForge will actually run with.
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_python_accel.ps1" -PythonExe "%VPY%" || echo [..] Final Python acceleration check had warnings - CPU fallbacks remain available.

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
REM Skipped silently when offline; YOLO/OpenCV fallbacks and center crop still work.
if not exist "backend\data\models\face_detection_yunet_2023mar.onnx" (
    mkdir backend\data\models 2>nul
    curl -fsSL --max-time 30 -o "backend\data\models\face_detection_yunet_2023mar.onnx" "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" >nul 2>&1 && (
        echo [OK] YuNet face model
    ) || (
        del /q "backend\data\models\face_detection_yunet_2023mar.onnx" 2>nul
        echo [..] YuNet model skipped - using available OpenCV fallback
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
