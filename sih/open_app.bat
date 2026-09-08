@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python environment not found at %PYTHON_EXE%
    pause
    exit /b 1
)
start "" "%PYTHON_EXE%" "%~dp0\main.py"
exit /b 0
