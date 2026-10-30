@echo off
title ScriptSense
cd /d "%~dp0"
.\.venv\Scripts\python.exe demo_app.py --web
pause
