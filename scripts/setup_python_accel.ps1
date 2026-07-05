param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Continue"

function Invoke-Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $PythonExe -m pip @Args
    return $LASTEXITCODE
}

function Has-NvidiaGpu {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $cmd) { return $false }
    & $cmd.Source >$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

Write-Host "Setting up Python acceleration stack..."

if (Has-NvidiaGpu) {
    Write-Host "[OK] NVIDIA GPU detected"
    Write-Host "Installing PyTorch CUDA wheels (WhisperX-compatible 2.8 stack; cu128, fallback cu126)..."
    $torchOk = $false
    Invoke-Pip install --upgrade "torch~=2.8.0" "torchvision~=0.23.0" "torchaudio~=2.8.0" --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -eq 0) {
        $torchOk = $true
    } else {
        Write-Host "[..] cu128 failed; trying cu126"
        Invoke-Pip install --upgrade "torch~=2.8.0" "torchvision~=0.23.0" "torchaudio~=2.8.0" --index-url https://download.pytorch.org/whl/cu126
        $torchOk = ($LASTEXITCODE -eq 0)
    }
    if (-not $torchOk) {
        Write-Host "[!] CUDA PyTorch install failed; installing CPU PyTorch so optional engines can still import."
        Invoke-Pip install --upgrade "torch~=2.8.0" "torchvision~=0.23.0" "torchaudio~=2.8.0" --index-url https://download.pytorch.org/whl/cpu | Out-Host
    }

    Write-Host "Installing CUDA runtime DLL wheels for CTranslate2/faster-whisper..."
    Invoke-Pip install --upgrade `
        "nvidia-cublas-cu12" `
        "nvidia-cudnn-cu12>=9" `
        "nvidia-cuda-runtime-cu12" `
        "nvidia-cuda-nvrtc-cu12" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] NVIDIA runtime wheels failed; ASR will fall back to CPU until fixed."
    }
} else {
    Write-Host "[..] No NVIDIA GPU detected; installing CPU PyTorch."
    Invoke-Pip install --upgrade "torch~=2.8.0" "torchvision~=0.23.0" "torchaudio~=2.8.0" --index-url https://download.pytorch.org/whl/cpu | Out-Host
}

$sanity = @'
import ctypes
import os
from pathlib import Path

try:
    import nvidia
    for root in getattr(nvidia, "__path__", []):
        for bin_dir in Path(root).glob("*/bin"):
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass
except Exception:
    pass

def can_load(names):
    for name in names:
        try:
            ctypes.WinDLL(name)
            return name
        except Exception:
            pass
    return ""

torch_cuda = False
try:
    import torch
    torch_cuda = bool(torch.cuda.is_available())
except Exception as exc:
    print(f"[!] torch import failed: {exc}")

ct2_devices = 0
try:
    import ctranslate2
    ct2_devices = int(ctranslate2.get_cuda_device_count())
except Exception as exc:
    print(f"[..] ctranslate2 CUDA check failed: {exc}")

cublas = can_load(["cublas64_12.dll"])
cudnn = can_load(["cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_ops_infer64_8.dll"])
print(f"[check] torch_cuda={torch_cuda} ctranslate2_cuda_devices={ct2_devices} cublas={cublas or 'missing'} cudnn={cudnn or 'missing'}")
'@

$tmp = New-TemporaryFile
try {
    Set-Content -LiteralPath $tmp.FullName -Value $sanity -Encoding UTF8
    & $PythonExe $tmp.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[..] Acceleration sanity check had warnings; setup can continue with CPU fallbacks."
    }
} finally {
    Remove-Item -LiteralPath $tmp.FullName -Force -ErrorAction SilentlyContinue
}
