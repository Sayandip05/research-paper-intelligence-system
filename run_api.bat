@echo off
echo Starting Research Paper Analyzer API...
cd backend
..\venv_clean\Scripts\uvicorn app.main:app --reload
pause
