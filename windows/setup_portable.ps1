# =====================================================================
# USB 휴대형 Python 환경 설치 (학교 PC: 재부팅 시 초기화되므로 USB에 설치)
#  - 관리자 권한 불필요, 시스템에 아무것도 설치하지 않음
#  - 저장소 폴더 안의 python\ 폴더에 Python 3.11 임베디드판 + 패키지 설치
# 사용: windows\setup_portable.bat 더블클릭 (또는 PowerShell에서 실행)
#   옵션: -Cuda  → PyTorch CUDA 13.0 빌드 설치 (약 2.5 GB, GPU 비교 실험용)
# =====================================================================
param([switch]$Cuda, [string]$PyVer = "3.11.9")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root "python"
$Tag = ($PyVer.Split(".")[0..1] -join "")
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (-not (Test-Path (Join-Path $Py "python.exe"))) {
    Write-Host "[1/4] Python $PyVer 임베디드판 다운로드..."
    New-Item -ItemType Directory -Force -Path $Py | Out-Null
    $zip = Join-Path $env:TEMP "py-embed.zip"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath $Py -Force
    Remove-Item $zip
    # site-packages 사용 가능하도록 ._pth 수정
    $pth = Join-Path $Py "python$Tag._pth"
    (Get-Content $pth) -replace "^#\s*import site", "import site" | Set-Content $pth
    Add-Content $pth "Lib\site-packages"
    Write-Host "[2/4] pip 설치..."
    $gp = Join-Path $Py "get-pip.py"
    Invoke-WebRequest "https://bootstrap.pypa.io/get-pip.py" -OutFile $gp
    & (Join-Path $Py "python.exe") $gp --no-warn-script-location
} else {
    Write-Host "[1-2/4] 기존 휴대형 Python 사용: $Py"
}

$exe = Join-Path $Py "python.exe"
Write-Host "[3/4] 패키지 설치..."
& $exe -m pip install --no-warn-script-location -r (Join-Path $Root "requirements.txt")
Write-Host "[4/4] PyTorch 설치..."
if ($Cuda) {
    # requirements 설치가 CPU 전용 torch를 깔기 때문에, 그 뒤에 CUDA 빌드로 교체한다 (Windows 휠은 CUDA 런타임 포함)
    & $exe -m pip install --no-warn-script-location --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu130
}
& $exe -c "import mujoco, stable_baselines3, torch; print('mujoco', mujoco.__version__, '| sb3', stable_baselines3.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
Write-Host "완료. 동작 확인: windows\quick_test.bat"
