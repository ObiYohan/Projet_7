@echo off
echo Starting Event Chatbot API...
cd /d "%~dp0"
uvicorn api.main_api:app --reload --host 0.0.0.0 --port 8000
pause