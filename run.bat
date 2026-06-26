@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM ---- Tuned for this machine: Ryzen 5 7600 (6c/12t) + RTX 5060 Ti 16GB ----
REM Whisper model + device stay on auto: with CUDA working the app already
REM picks large-v3 on this card, and falls back safely to CPU if not.
REM 8 parallel renders: NVENC barely loads the CPU, so 8 workers saturate
REM the NVENC encoder without contention. (4 was leaving throughput on the table.)
set CLIPFORGE_RENDER_WORKERS=8
REM ~6x faster transcription at near-large-v3 quality on this GPU.
REM Batch 48: with 16 GB VRAM the model (~3.5 GB) + batch buffers (~6 GB)
REM fit comfortably, maximizing GPU utilization on longer videos.
set CLIPFORGE_WHISPER_MODEL=large-v3-turbo
set CLIPFORGE_WHISPER_BATCH=48
REM 2 concurrent projects: the GPU can time-slice transcription and rendering
REM when both are in-flight, keeping utilization high.
set CLIPFORGE_PIPELINE_WORKERS=2
REM New projects default to the strongest local path on this workstation.
set CLIPFORGE_DEFAULT_POWER_MODE=max_gpu
REM Ollama flash attention: ~2x faster LLM inference for AI titles/virality.
set OLLAMA_FLASH_ATTENTION=1
REM Keep Ollama model loaded between requests (avoids reload delays).
set OLLAMA_KEEP_ALIVE=5m
REM Add optional tools to PATH for capability detection.
set PATH=%PATH%;%LOCALAPPDATA%;%ProgramFiles%\Tesseract-OCR
REM Pull private/local account settings written by setup.bat into this process.
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('HF_TOKEN','User')"`) do if not "%%T"=="" set "HF_TOKEN=%%T"
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('CLIPFORGE_ASD_DIR','User')"`) do if not "%%T"=="" set "CLIPFORGE_ASD_DIR=%%T"
REM AI titles/vision auto-pick the strongest installed Ollama models. setup.bat
REM pulls hardware-fit defaults; set CLIPFORGE_LLM_MODEL / CLIPFORGE_VLM_MODEL
REM only if you want to force a specific model.
REM Uncomment for AV1 output (better quality/bitrate; H.264 plays everywhere):
REM set CLIPFORGE_CODEC=av1

echo Starting Ollama if available...
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_ollama_models.ps1" -StartOnly >nul 2>&1

echo Checking ClipForge backend...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/ready' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
if errorlevel 1 (
    echo Starting ClipForge backend in a new window...
    start "ClipForge backend - close this window to stop" "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --port 8000
) else (
    echo ClipForge backend is already running.
)

echo Waiting until the server is ready...
for /l %%I in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/api/ready' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1" >nul 2>&1
    if not errorlevel 1 goto ready
    timeout /t 1 /nobreak >nul
)

echo(
echo ClipForge did not become ready on http://localhost:8000.
echo If the backend window closed, scroll up there for the Python error.
echo You can also run this for details:
echo   .venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
echo(
pause
exit /b 1

:ready
start "" http://localhost:8000
echo(
echo ClipForge is running at  http://localhost:8000
echo To stop it, close the "ClipForge backend" window.
echo(
