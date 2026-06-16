param(
    [string]$PythonExe = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

function Read-PlainSecret([string]$Prompt) {
    $secure = Read-Host $Prompt -AsSecureString
    if ($secure.Length -eq 0) { return "" }
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    } finally {
        if ($bstr -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
}

function Test-HfToken([string]$Token, [string]$Py) {
    $old = $env:HF_TOKEN
    $env:HF_TOKEN = $Token
    $code = @'
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

token = os.environ.get("HF_TOKEN")
try:
    who = HfApi().whoami(token=token)
    name = who.get("name") or who.get("fullname") or "ok"
    hf_hub_download(
        repo_id="pyannote/speaker-diarization-community-1",
        filename="config.yaml",
        token=token,
    )
    print(name)
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
'@
    try {
        $out = & $Py -c $code 2>&1
        if ($LASTEXITCODE -eq 0) {
            return @{ ok = $true; message = ($out | Select-Object -First 1) }
        }
        return @{ ok = $false; message = ($out -join "`n") }
    } finally {
        $env:HF_TOKEN = $old
    }
}

$pyPath = Resolve-Path $PythonExe -ErrorAction SilentlyContinue
if (-not $pyPath) {
    Write-Host "[..] Hugging Face token setup skipped - Python env not found: $PythonExe"
    exit 0
}
$PythonExe = $pyPath.Path

$existing = $env:HF_TOKEN
if (-not $existing) {
    $existing = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
}

if ($existing) {
    Write-Host "Checking existing HF_TOKEN..."
    $check = Test-HfToken $existing $PythonExe
    if ($check.ok) {
        [Environment]::SetEnvironmentVariable("HF_TOKEN", $existing, "User")
        Write-Host "[OK] HF_TOKEN is valid for pyannote diarization ($($check.message))."
        exit 0
    }
    Write-Host "[!] Existing HF_TOKEN did not validate for pyannote diarization."
}

Write-Host ""
Write-Host "WhisperX diarization needs a Hugging Face token and pyannote model access."
Write-Host "I will open the two pages you need: accept the pyannote terms, then create a Read token."
Write-Host "Paste the token here when ready. Leave blank to skip for now."
Start-Process "https://huggingface.co/pyannote/speaker-diarization-community-1" | Out-Null
Start-Process "https://huggingface.co/settings/tokens" | Out-Null

$token = Read-PlainSecret "HF token"
if (-not $token) {
    Write-Host "[..] HF_TOKEN skipped. Diarization will stay off until a valid token is set."
    exit 0
}

Write-Host "Validating token and pyannote access..."
$result = Test-HfToken $token $PythonExe
if (-not $result.ok) {
    Write-Host "[!] Token could not access pyannote/speaker-diarization-community-1."
    Write-Host "    Accept the model conditions, create a Read/fine-grained token, then run setup.bat again."
    Write-Host "    Details: $($result.message)"
    exit 0
}

[Environment]::SetEnvironmentVariable("HF_TOKEN", $token, "User")
$env:HF_TOKEN = $token
Write-Host "[OK] HF_TOKEN saved to your Windows user environment ($($result.message))."
