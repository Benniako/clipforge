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
echo    Setup complete!  Now double-click run.bat
echo ==========================================
pause
