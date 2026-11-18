@echo off
chcp 65001 >nul
set "ROOT=%~dp0.."
set "PY=%ROOT%\python\python.exe"
if not exist "%PY%" (
  REM 휴대형 환경이 없으면 시스템 python 사용
  set "PY=python"
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
