@echo off
setlocal EnableDelayedExpansion
title PaperCheckAI - AI Examination Platform
cd /d "%~dp0"
echo =====================================================================
echo   PaperCheckAI : AI Examination Checking and Academic Mentoring
echo =====================================================================

rem ---------- 1. Python (no fixed install path assumed) ----------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if not defined PY ( where py >nul 2>nul && set "PY=py -3" )
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo [ERROR] Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".
  pause & exit /b 1
)
for /f "delims=" %%v in ('%PY% -c "import sys;print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))"') do set PYVER=%%v
echo [OK] Python %PYVER%  ^(%PY%^)
%PY% -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" || ( echo [ERROR] Python 3.10 or newer is required. & pause & exit /b 1 )

rem ---------- 2. .env ----------
if not exist ".env" (
  copy /y ".env.example" ".env" >nul
  echo [OK] Created .env from .env.example ^(Ollama defaults, no API key needed^)
)

rem ---------- 3. Dependencies ----------
%PY% -c "import fastapi, uvicorn, sqlalchemy, streamlit, requests, dotenv" >nul 2>nul
if errorlevel 1 (
  echo [*] Installing dependencies ^(first run can take several minutes^)...
  %PY% -m pip install --upgrade pip
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 ( echo [ERROR] pip install failed - see messages above. & pause & exit /b 1 )
) else (
  echo [OK] Core dependencies are installed
)

rem ---------- 4. Ollama + model ----------
set "OLLAMA_URL=http://localhost:11434"
set "OLLAMA_MODEL=llama3.2:3b"
for /f "usebackq tokens=1,* delims==" %%a in (`findstr /b /i "OLLAMA_BASE_URL= OLLAMA_MODEL=" .env 2^>nul`) do (
  if /i "%%a"=="OLLAMA_BASE_URL" set "OLLAMA_URL=%%b"
  if /i "%%a"=="OLLAMA_MODEL" set "OLLAMA_MODEL=%%b"
)
%PY% scripts\check_ollama.py "%OLLAMA_URL%" "%OLLAMA_MODEL%"
set OLLAMA_RC=%errorlevel%
if %OLLAMA_RC%==2 (
  echo [WARN] Ollama is not running. Trying to start it...
  where ollama >nul 2>nul && ( start "" /b ollama serve & timeout /t 5 /nobreak >nul )
  %PY% scripts\check_ollama.py "%OLLAMA_URL%" "%OLLAMA_MODEL%"
  set OLLAMA_RC=!errorlevel!
)
if %OLLAMA_RC%==3 (
  echo [*] Model %OLLAMA_MODEL% is missing. Downloading with: ollama pull %OLLAMA_MODEL%
  where ollama >nul 2>nul && ollama pull %OLLAMA_MODEL%
)
if not %OLLAMA_RC%==0 echo [WARN] The app will start, but AI answers stay unavailable ^(and are flagged for teacher review^) until Ollama + %OLLAMA_MODEL% are ready.

rem ---------- 4b. Tesseract OCR (needed to read answer sheets) ----------
%PY% -c "import sys;from backend.tesseract_locator import configure as c;r=c();print('[OK] Tesseract OCR: '+r['path'] if r['available'] else '[WARN] '+r['error']);sys.exit(0 if r['available'] else 1)"
if errorlevel 1 echo [WARN] Answer-sheet OCR will not work until Tesseract is installed. Handwriting is never graded silently: unreadable sheets are rejected with this message.

rem ---------- 5. Database (existing data is never overwritten) ----------
if exist "intelligrade.db" (
  %PY% scripts\backup_db.py backup >nul 2>nul && echo [OK] Database backup created in backups\
) else (
  echo [*] No database yet - it will be created now.
)
%PY% run.py init || ( echo [ERROR] Database initialisation failed. & pause & exit /b 1 )

echo.
echo =====================================================================
echo   Web portal : http://localhost:8501
echo   API docs   : http://localhost:8000/docs
echo   AI status  : http://localhost:8000/system/llm-status
echo =====================================================================
%PY% run.py
pause
