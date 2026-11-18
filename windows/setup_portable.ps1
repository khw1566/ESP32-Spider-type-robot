# =====================================================================
# USB 휴대형 Python 환경 설치 (학교 PC: 재부팅 시 초기화되므로 USB에 설치)
#  - 관리자 권한 불필요, 시스템에 아무것도 설치하지 않음
#  - 저장소 폴더 안의 python\ 폴더에 Python 3.11 (tkinter 포함) + 패키지 설치
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
    Write-Host "[1/4] Python $PyVer 다운로드 (관제 대시보드용 tkinter 포함 정식 설치판)..."
    $inst = Join-Path $env:TEMP "python-$PyVer-amd64.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PyVer/python-$PyVer-amd64.exe" -OutFile $inst
    Write-Host "[2/4] USB 폴더에 설치: $Py (시스템 PATH·연결프로그램은 건드리지 않음)"
    $instArgs = @("/quiet", "InstallAllUsers=0", "TargetDir=$Py", "Include_launcher=0", "Include_test=0",
              "Include_doc=0", "Shortcuts=0", "AssociateFiles=0", "PrependPath=0", "Include_tcltk=1", "Include_pip=1")
    $p = Start-Process -FilePath $inst -ArgumentList $instArgs -Wait -PassThru
    Remove-Item $inst
    if (-not (Test-Path (Join-Path $Py "python.exe"))) {
        throw "Python 설치 실패 (종료코드 $($p.ExitCode)). 이 PC에 같은 버전 Python이 이미 설치되어 있으면 -PyVer 3.11.8 처럼 다른 버전을 지정하세요."
    }
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
& $exe -c "import tkinter, mujoco, stable_baselines3, torch; print('mujoco', mujoco.__version__, '| sb3', stable_baselines3.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
Write-Host "완료. 동작 확인: windows\quick_test.bat"
