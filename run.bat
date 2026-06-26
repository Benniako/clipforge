@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM ---- Responsive defaults for this workstation ------------------------
REM The backend auto-selects Whisper size from actually usable CUDA. Do not
REM force large-v3-turbo here: when cuBLAS/cuDNN are missing, that would push
REM a huge model onto CPU and make long videos feel frozen.
REM Render fan-out is kept moderate so the UI stays usable while clips render.
set CLIPFORGE_RENDER_WORKERS=4
set CLIPFORGE_PIPELINE_WORKERS=1
REM Ollama flash attention: ~2x faster LLM inference for AI titles/virality.
set OLLAMA_FLASH_ATTENTION=1
REM Keep Ollama model loaded between requests (avoids reload delays).
set OLLAMA_KEEP_ALIVE=5m
REM Add optional tools to PATH for capability detection and yt-dlp YouTube parsing.
set PATH=%~dp0.tools\deno;%PATH%;%LOCALAPPDATA%;%ProgramFiles%\Tesseract-OCR
REM Pull private/local account settings written by setup.bat into this process.
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('HF_TOKEN','User')"`) do if not "%%T"=="" set "HF_TOKEN=%%T"
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('CLIPFORGE_ASD_DIR','User')"`) do if not "%%T"=="" set "CLIPFORGE_ASD_DIR=%%T"
REM AI titles/vision auto-pick the strongest installed Ollama models. setup.bat
REM pulls hardware-fit defaults; set CLIPFORGE_LLM_MODEL / CLIPFORGE_VLM_MODEL
REM only if you want to force a specific model.
REM Uncomment for AV1 output (better quality/bitrate; H.264 plays everywhere):
REM set CLIPFORGE_CODEC=av1

echo Starting Ollama if available...
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0scripts\setup_ollama_models.ps1" -StartOnly -MaxWaitSeconds 3 >nul 2>&1

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
