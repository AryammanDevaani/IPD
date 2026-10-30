@echo off
title ScriptSense Full ML Training Pipeline
cd /d "%~dp0"
echo ========================================================
echo   Starting ScriptSense ML Training Pipeline...
echo ========================================================
echo.
.\.venv\Scripts\python.exe -u run_pipeline.py
echo.
echo ========================================================
echo   Training pipeline finished!
echo ========================================================
pause

