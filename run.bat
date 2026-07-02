@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM ---- GPU-maximised throughput for this workstation -------------------
REM The backend auto-selects Whisper size and batch from actually usable CUDA
REM and VRAM. max_gpu saturates the GPU for transcription + VLM scoring.
REM Render fan-out and pipeline workers are tuned for the RTX 5060 Ti 16 GB
REM NVENC encoder (6 concurrent sessions) so batch creation is never CPU-bound.
set CLIPFORGE_DEFAULT_POWER_MODE=max_gpu
set CLIPFORGE_RENDER_WORKERS=6
set CLIPFORGE_PIPELINE_WORKERS=2
REM Ollama flash attention: ~2x faster LLM inference for AI titles/virality.
set OLLAMA_FLASH_ATTENTION=1
REM Keep Ollama model loaded between requests (avoids reload delays).
set OLLAMA_KEEP_ALIVE=5m
REM Add optional tools to PATH for capability detection and yt-dlp YouTube parsing.
set PATH=%~dp0.tools\deno;%PATH%;%LOCALAPPDATA%;%ProgramFiles%\Tesseract-OCR
REM Pull private/local account settings written by setup.bat into this process.
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('HF_TOKEN','User')"`) do if not "%%T"=="" set "HF_TOKEN=%%T"
REM AI titles/vision auto-pick the strongest installed Ollama models. setup.bat
REM pulls hardware-fit defaults; set CLIPFORGE_LLM_MODEL / CLIPFORGE_VLM_MODEL
REM only if you want to force a specific model.
REM Uncomment for AV1 output (better quality/bitrate; H.264 plays everywhere):
REM set CLIPFORGE_CODEC=av1

echo Starting Ollama if available...
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_ollama_models.ps1" -StartOnly -MaxWaitSeconds 3 >nul 2>&1

echo Checking ClipForge backend...
set CLIPFORGE_PORT=8000
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:%CLIPFORGE_PORT%/api/ready' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
if errorlevel 1 (
    for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$used = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | ForEach-Object LocalPort); if ($used -contains 8000) { 8010..8099 | Where-Object { $used -notcontains $_ } | Select-Object -First 1 } else { 8000 }"`) do set "CLIPFORGE_PORT=%%P"
    if "!CLIPFORGE_PORT!"=="" set CLIPFORGE_PORT=8010
    echo Starting ClipForge backend in a new window...
    start "ClipForge backend - close this window to stop" "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --port !CLIPFORGE_PORT!
) else (
    echo ClipForge backend is already running.
)

echo Waiting until the server is ready...
for /l %%I in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:!CLIPFORGE_PORT!/api/ready' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 goto ready
    timeout /t 1 /nobreak >nul
)

echo(
echo ClipForge did not become ready on http://localhost:!CLIPFORGE_PORT!.
echo If the backend window closed, scroll up there for the Python error.
echo You can also run this for details:
echo   .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port !CLIPFORGE_PORT!
echo(
pause
exit /b 1

:ready
start "" http://localhost:!CLIPFORGE_PORT!
echo(
echo ClipForge is running at  http://localhost:!CLIPFORGE_PORT!
echo To stop it, close the "ClipForge backend" window.
echo(
