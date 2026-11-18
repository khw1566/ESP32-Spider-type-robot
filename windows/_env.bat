@echo off
chcp 65001 >nul
set "ROOT=%~dp0.."
REM 우선순위: USB 휴대형 python\ > 가상환경 .venv\ > 시스템 python
set "PY=%ROOT%\python\python.exe"
if not exist "%PY%" set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
