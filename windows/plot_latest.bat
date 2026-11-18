@echo off
REM 가장 최근 실행 폴더의 보고서용 그래프 생성 (runs\<실행>\figures)
call "%~dp0_env.bat"
for /f "delims=" %%d in ('dir /b /ad /o-d runs') do (set "LAST=runs\%%d" & goto :found)
:found
"%PY%" scripts\plot_report.py "%LAST%"
pause
