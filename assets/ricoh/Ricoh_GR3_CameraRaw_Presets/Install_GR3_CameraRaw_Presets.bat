@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install_GR3_CameraRaw_Presets.ps1"
echo.
pause