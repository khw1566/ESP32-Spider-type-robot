@echo off
call "%~dp0_env.bat"
"%PY%" scripts\view_model.py %*
