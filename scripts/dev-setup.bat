@echo off
REM =============================================================================
REM Development Setup Script for Windows
REM =============================================================================

echo ========================================
echo ArXivFuturaSearch - Development Setup
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo [+] Creating .env file from .env.example...
    copy .env.example .env >nul
    echo [!] WARNING: Please edit .env with your configuration
    echo     Especially SECRET_KEY and API keys!
    echo.
)

REM Check Python
echo [~] Checking Python version...
python --version
echo.

REM Install dependencies
echo [+] Installing dependencies...
pip install -e .
echo.

REM Create data directories
echo [+] Creating data directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\chroma_db mkdir data\chroma_db
if not exist data\exports mkdir data\exports
if not exist logs mkdir logs
echo.

REM Download NLTK data
echo [+] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
echo.

REM Check Docker
echo [~] Checking Docker...
where docker >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [+] Docker is installed
    where docker-compose >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [+] Docker Compose is installed
    ) else (
        echo [!] Docker Compose not found
    )
) else (
    echo [!] Docker not found
)
echo.

REM Run migrations
echo [+] Running database migrations...
alembic upgrade head
if %ERRORLEVEL% NEQ 0 (
    echo [!] Migration failed (database might not be running)
)
echo.

echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the application:
echo   Local:   uvicorn app.main:app --reload
echo   Docker:  docker-compose -f docker-compose.dev.yml up -d
echo.
echo Available services:
echo   Application:   http://localhost:8000
echo   API Docs:      http://localhost:8000/api/docs
echo   Adminer:       http://localhost:8080
echo   Redis GUI:     http://localhost:8081
echo   Prometheus:    http://localhost:9090
echo   Grafana:       http://localhost:3000 (admin/admin)
echo   Jaeger:        http://localhost:16686
echo.
pause
