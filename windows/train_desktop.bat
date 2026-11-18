@echo off
call "%~dp0_env.bat"
if "%~1"=="" (
  "%PY%" scripts\train.py --override configs\desktop.yaml --tag desktop
) else (
  "%PY%" scripts\train.py --override configs\desktop.yaml --tag desktop --resume "%~1"
)
pause
