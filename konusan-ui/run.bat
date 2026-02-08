@echo off
setlocal
cd /d %~dp0

REM venv yoksa oluştur
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] venv olusturuluyor...
  python -m venv .venv
)

echo [INFO] venv aktif ediliyor...
call ".venv\Scripts\activate.bat"

echo [INFO] pip guncelleniyor...
python -m pip install --upgrade pip

echo [INFO] gereksinimler kuruluyor...
python -m pip install PyQt6

echo [INFO] UI baslatiliyor...
python ui_app.py

endlocal
