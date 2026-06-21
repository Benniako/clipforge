param(
    [switch]$StartOnly
)

$ErrorActionPreference = "Continue"

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    foreach ($path in $candidates) {
        if ($path -and (Test-Path $path)) { return $path }
    }
    return $null
}

function Test-OllamaServer {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Start-OllamaServer($ollama) {
    if (Test-OllamaServer) {
        Write-Host "[OK] Ollama server is running"
        return $true
    }
    Write-Host "Starting Ollama server..."
    try {
        Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    } catch {
        Write-Host "[..] Could not start Ollama automatically: $($_.Exception.Message)"
        return $false
    }
    for ($i = 0; $i -lt 20; $i++) {
        Start-Sleep -Seconds 1
        if (Test-OllamaServer) {
            Write-Host "[OK] Ollama server is running"
            return $true
        }
    }
    Write-Host "[..] Ollama did not answer yet; run.bat will try again later"
    return $false
}

function Install-Ollama {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[..] Ollama not found and winget is unavailable."
        Write-Host "     Install Ollama from https://ollama.com/download/windows for AI titles/vision."
        return $null
    }

    Write-Host "Ollama not found - installing with winget..."
    winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[..] Ollama install was skipped or failed. ClipForge still runs without it."
        return $null
    }
    return Find-Ollama
}

function Pull-Model($ollama, $model, $fallback = $null) {
    Write-Host "Pulling Ollama model: $model"
    & $ollama pull $model
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $model"
        return
    }
    if ($fallback) {
        Write-Host "[..] $model failed; trying $fallback"
        & $ollama pull $fallback
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $fallback"
            return
        }
    }
    Write-Host "[..] Model pull skipped: $model"
}

function Get-HardwareProfile {
    $vramMb = 0
    try {
        $raw = & nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            $vramMb = [int](($raw -split "`n")[0].Trim())
        }
    } catch {}

    $ramGb = 0
    try {
        $ramBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        $ramGb = [math]::Round($ramBytes / 1GB)
    } catch {}

    return [pscustomobject]@{
        VramMb = $vramMb
        VramGb = [math]::Round($vramMb / 1024, 1)
        RamGb = $ramGb
    }
}

function Select-ModelPlan($hw) {
    if ($hw.VramMb -ge 20000 -and $hw.RamGb -ge 48) {
        return [pscustomobject]@{
            Vision = "qwen2.5vl:32b"
            VisionFallback = "qwen2.5vl:7b"
            Text = "qwen3:32b"
            TextFallback = "qwen3:14b"
        }
    }
    if ($hw.VramMb -ge 10000 -and $hw.RamGb -ge 24) {
        return [pscustomobject]@{
            Vision = "qwen2.5vl:7b"
            VisionFallback = "qwen2.5vl:3b"
            Text = "qwen3:14b"
            TextFallback = "qwen3:8b"
        }
    }
    if ($hw.VramMb -ge 6000 -and $hw.RamGb -ge 16) {
        return [pscustomobject]@{
            Vision = "qwen2.5vl:7b"
            VisionFallback = "qwen2.5vl:3b"
            Text = "qwen3:8b"
            TextFallback = "qwen3:4b"
        }
    }
    return [pscustomobject]@{
        Vision = "qwen2.5vl:3b"
        VisionFallback = "llava:7b"
        Text = "qwen3:4b"
        TextFallback = "llama3.2:3b"
    }
}

$ollama = Find-Ollama
if (-not $ollama -and -not $StartOnly) {
    $ollama = Install-Ollama
}

if (-not $ollama) {
    Write-Host "[..] Ollama unavailable. Local AI titles/vision are optional and will be disabled."
    exit 0
}

Write-Host "[OK] Ollama: $ollama"
$server = Start-OllamaServer $ollama

if ($StartOnly) {
    exit 0
}

if ($server) {
    $hw = Get-HardwareProfile
    $plan = Select-ModelPlan $hw
    Write-Host ("Detected hardware: {0} GB VRAM, {1} GB RAM" -f $hw.VramGb, $hw.RamGb)
    Write-Host ("Best default models: vision={0}, text={1}" -f $plan.Vision, $plan.Text)
    Pull-Model $ollama $plan.Vision $plan.VisionFallback
    Pull-Model $ollama $plan.Text $plan.TextFallback
}

exit 0
