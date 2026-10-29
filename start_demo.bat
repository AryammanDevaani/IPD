@echo off
title ScriptSense Live Demo Server
cd /d "%~dp0"
echo ========================================================
echo   Starting ScriptSense Local ML Engine Demo Server...
echo ========================================================
echo.
.\.venv\Scripts\python.exe demo_app.py --web
pause

