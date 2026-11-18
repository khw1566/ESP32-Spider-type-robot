@echo off
call "%~dp0_env.bat"
"%PY%" -m pytest -q tests
"%PY%" scripts\train.py --override configs\quick_test.yaml --tag quicktest
pause
