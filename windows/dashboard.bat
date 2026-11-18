@echo off
REM 학습 관제 대시보드 (학습 시작/정지, 시간 제한 등 학습 중 설정 변경, 실시간 그래프)
call "%~dp0_env.bat"
set "PYW=%PY:python.exe=pythonw.exe%"
if exist "%PYW%" (start "" "%PYW%" scripts\dashboard.py) else ("%PY%" scripts\dashboard.py)
