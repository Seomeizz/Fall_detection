@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Khong tim thay moi truong .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Starting FallGuard AI...
python -u backend\run_server.py

echo.
echo FallGuard server stopped.
pause