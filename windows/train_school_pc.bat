@echo off
REM 학교 PC 8시간 세션: 이전 체크포인트가 있으면 이어서 학습
REM   사용: train_school_pc.bat               (처음부터)
REM         train_school_pc.bat runs\...\checkpoints\latest.zip   (이어받기)
call "%~dp0_env.bat"
if "%~1"=="" (
  "%PY%" scripts\train.py --override configs\school_pc.yaml --tag school
) else (
  "%PY%" scripts\train.py --override configs\school_pc.yaml --tag school --resume "%~1"
)
pause
