param(
    [string]$InstallDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $InstallDir) {
    $repo = Split-Path -Parent $PSScriptRoot
    $InstallDir = Join-Path $repo ".tools\deno"
}

$denoExe = Join-Path $InstallDir "deno.exe"
if (Test-Path $denoExe) {
    try {
        $version = & $denoExe --version 2>$null | Select-Object -First 1
        Write-Host "[OK] Deno: $version"
        exit 0
    } catch {
        Write-Host "[..] Existing Deno did not run; reinstalling local copy."
    }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("clipforge-deno-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$zip = Join-Path $tmp "deno.zip"

try {
    $url = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip"
    Write-Host "Installing Deno locally..."
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 120
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
    $downloaded = Get-ChildItem -LiteralPath $tmp -Recurse -Filter "deno.exe" | Select-Object -First 1
    if (-not $downloaded) {
        throw "Downloaded archive did not contain deno.exe"
    }
    Copy-Item -LiteralPath $downloaded.FullName -Destination $denoExe -Force
    $version = & $denoExe --version 2>$null | Select-Object -First 1
    Write-Host "[OK] Deno: $version"
} catch {
    Write-Host "[..] Deno install skipped: $($_.Exception.Message)"
    Write-Host "     YouTube URL imports may be capped at 360p until Deno is installed."
    exit 0
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
