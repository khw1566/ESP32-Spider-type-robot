@echo off
chcp 65001 >nul
REM USB 휴대형 Python 환경 설치. GPU용 PyTorch가 필요하면: setup_portable.bat -Cuda
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_portable.ps1" %*
pause
