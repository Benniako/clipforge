@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM ---- Tuned for this machine: Ryzen 5 7600 (6c/12t) + RTX 5060 Ti 16GB ----
REM Whisper model + device stay on auto: with CUDA working the app already
REM picks large-v3 on this card, and falls back safely to CPU if not.
REM 4 parallel renders: encoding runs on NVENC, so ffmpeg barely loads the CPU.
set CLIPFORGE_RENDER_WORKERS=4
REM Uncomment for ~6x faster transcription at near-large-v3 quality:
REM set CLIPFORGE_WHISPER_MODEL=large-v3-turbo

echo Starting ClipForge backend in a new window...
start "ClipForge backend - close this window to stop" .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
echo Waiting for the server to come up, then opening your browser...
timeout /t 4 /nobreak >nul
start "" http://localhost:8000
echo(
echo ClipForge is running at  http://localhost:8000
echo To stop it, close the "ClipForge backend" window.
echo(
