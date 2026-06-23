param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$dataDir = Join-Path $env:USERPROFILE "panns_data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$labels = Join-Path $dataDir "class_labels_indices.csv"
$ckpt = Join-Path $dataDir "Cnn14_mAP=0.431.pth"

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$Path,
        [int64]$MinBytes
    )
    if ((Test-Path $Path) -and ((Get-Item $Path).Length -ge $MinBytes)) {
        Write-Host "[OK] $(Split-Path $Path -Leaf)"
        return
    }
    Write-Host "Downloading $(Split-Path $Path -Leaf)..."
    $tmp = "$Path.download"
    Remove-Item -Force -ErrorAction SilentlyContinue $tmp
    Invoke-WebRequest -Uri $Url -OutFile $tmp
    if ((Get-Item $tmp).Length -lt $MinBytes) {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp
        throw "downloaded file was too small: $Path"
    }
    Move-Item -Force $tmp $Path
    Write-Host "[OK] $(Split-Path $Path -Leaf)"
}

Download-IfMissing `
    -Url "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv" `
    -Path $labels `
    -MinBytes 10000

Download-IfMissing `
    -Url "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" `
    -Path $ckpt `
    -MinBytes 300000000

if ($PythonExe -and (Test-Path $PythonExe)) {
    @'
from pathlib import Path
from panns_inference import AudioTagging

ckpt = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
assert ckpt.exists() and ckpt.stat().st_size > 300_000_000
print("[OK] PANNs checkpoint ready")
'@ | & $PythonExe -
}
