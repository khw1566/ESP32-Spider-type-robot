@echo off
REM 브라우저에서 http://localhost:6006 접속
call "%~dp0_env.bat"
"%PY%" -m tensorboard.main --logdir runs
pause
