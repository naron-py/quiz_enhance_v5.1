@echo off
setlocal
cd /d "%~dp0"

:: Check for Virtual Environment
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [FIRST RUN DETECTED]
    echo Virtual environment not found. Running setup...
    call "%~dp0setup_env.bat"
)

"%~dp0venv\Scripts\python.exe" ui_webview.py
