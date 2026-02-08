@echo off
cd /d %~dp0

echo [Konusan Foto] Portable mode starting...
echo Using Python: %cd%\python\python.exe
echo.

cd /d "%cd%\konusan-ui"
"%cd%\..\python\python.exe" ui_app.py

echo.
echo Program kapandi.
pause
