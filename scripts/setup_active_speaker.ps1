param(
    [string]$PythonExe = ".venv\Scripts\python.exe",
    [string]$RepoDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $RepoDir) {
    $RepoDir = Join-Path $root "backend\data\models\LR-ASD"
}

function Test-CommandExists([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Patch-LrAsdForWindows([string]$Path) {
    $script = Join-Path $Path "Columbia_test.py"
    if (-not (Test-Path $script)) { return }
    $text = Get-Content $script -Raw
    $patched = $text.Replace("file.split('/')[-1]", "os.path.basename(file)")
    $oldImports = @'
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
'@
    $newImports = @'
try:
	from scenedetect.video_manager import VideoManager
	from scenedetect.scene_manager import SceneManager
	from scenedetect.frame_timecode import FrameTimecode
	from scenedetect.stats_manager import StatsManager
	from scenedetect.detectors import ContentDetector
except Exception:
	from scenedetect import SceneManager, open_video
	from scenedetect.frame_timecode import FrameTimecode
	from scenedetect.detectors import ContentDetector
	VideoManager = None
	StatsManager = None
'@
    $patched = $patched.Replace($oldImports, $newImports)
    $oldScene = @'
def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList
'@
    $newScene = @'
def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	if VideoManager is None:
		video = open_video(args.videoFilePath)
		sceneManager = SceneManager()
		sceneManager.add_detector(ContentDetector())
		sceneManager.detect_scenes(video)
		sceneList = sceneManager.get_scene_list()
		if sceneList == []:
			fps = float(getattr(video, 'frame_rate', 25.0) or 25.0)
			end_tc = getattr(video, 'duration', None)
			if end_tc is None:
				end_tc = FrameTimecode(int(getattr(video, 'frame_number', 0) or 0), fps)
			sceneList = [(FrameTimecode(0, fps), end_tc)]
	else:
		videoManager = VideoManager([args.videoFilePath])
		statsManager = StatsManager()
		sceneManager = SceneManager(statsManager)
		sceneManager.add_detector(ContentDetector())
		baseTimecode = videoManager.get_base_timecode()
		videoManager.set_downscale_factor()
		videoManager.start()
		sceneManager.detect_scenes(frame_source = videoManager)
		sceneList = sceneManager.get_scene_list(baseTimecode)
		if sceneList == []:
			sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList
'@
    $patched = $patched.Replace($oldScene, $newScene)
    if ($patched -ne $text) {
        Set-Content -Path $script -Value $patched -Encoding UTF8
        Write-Host "[OK] Patched LR-ASD demo for Windows paths and PySceneDetect compatibility."
    }
}

$pyPath = Resolve-Path $PythonExe -ErrorAction SilentlyContinue
if (-not $pyPath) {
    Write-Host "[..] LR-ASD setup skipped - Python env not found: $PythonExe"
    exit 0
}
$PythonExe = $pyPath.Path

if (-not (Test-Path $RepoDir)) {
    if (-not (Test-CommandExists "git")) {
        Write-Host "[..] LR-ASD setup skipped - git was not found."
        exit 0
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $RepoDir) | Out-Null
    Write-Host "Cloning LR-ASD active speaker model..."
    git clone --depth 1 https://github.com/Junhua-Liao/LR-ASD "$RepoDir"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[..] LR-ASD clone failed; active speaker will stay off."
        exit 0
    }
} else {
    Write-Host "[OK] LR-ASD checkout found."
}

Write-Host "Installing LR-ASD helper packages..."
& $PythonExe -m pip install "python_speech_features>=0.6" "gdown>=5.2" | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "[..] LR-ASD helper install failed; active speaker may stay off."
}

Patch-LrAsdForWindows $RepoDir

$required = @(
    "ASD.py",
    "Columbia_test.py",
    "model\Model.py",
    "weight\pretrain_AVA.model"
)
$missing = @()
foreach ($rel in $required) {
    if (-not (Test-Path (Join-Path $RepoDir $rel))) { $missing += $rel }
}
if ($missing.Count -gt 0) {
    Write-Host "[..] LR-ASD checkout is missing: $($missing -join ', ')"
    exit 0
}
$weight = Join-Path $RepoDir "weight\pretrain_AVA.model"
if ((Get-Item $weight).Length -lt 100000) {
    Write-Host "[..] LR-ASD weight file looks incomplete. If Git LFS is installed, run:"
    Write-Host "    git -C `"$RepoDir`" lfs pull"
    exit 0
}
$s3fdWeight = Join-Path $RepoDir "model\faceDetector\s3fd\sfd_face.pth"
if ((-not (Test-Path $s3fdWeight)) -or (Get-Item $s3fdWeight).Length -lt 100000) {
    Write-Host "Downloading LR-ASD face-detector weight..."
    & $PythonExe -m gdown "https://drive.google.com/uc?id=1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt" -O "$s3fdWeight" | Out-Host
    if (($LASTEXITCODE -ne 0) -or (-not (Test-Path $s3fdWeight)) -or (Get-Item $s3fdWeight).Length -lt 100000) {
        Write-Host "[..] S3FD face-detector weight is missing; active speaker will stay off."
        exit 0
    }
}

[Environment]::SetEnvironmentVariable("CLIPFORGE_ASD_DIR", $RepoDir, "User")
$env:CLIPFORGE_ASD_DIR = $RepoDir
Write-Host "[OK] LR-ASD active speaker path saved: $RepoDir"
