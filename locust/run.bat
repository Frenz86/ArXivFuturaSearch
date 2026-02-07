@echo off
REM Locust load testing startup script for Windows

echo ====================================
echo ArXivFuturaSearch Load Testing
echo ====================================
echo.

REM Check if application is running
echo Checking if application is running on http://localhost:8000...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ERROR: Application is not running!
    echo Please start the application first:
    echo   uvicorn app.main:app --reload
    echo.
    pause
    exit /b 1
)

echo Application is running!
echo.

REM Display menu
echo Choose load test intensity:
echo   1. Light (50 users, 10/s spawn)
echo   2. Medium (200 users, 20/s spawn)
echo   3. Heavy (500 users, 50/s spawn)
echo   4. Custom
echo   5. Web UI mode (interactive)
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    set USERS=50
    set SPAWN=10
    goto :run_headless
)
if "%choice%"=="2" (
    set USERS=200
    set SPAWN=20
    goto :run_headless
)
if "%choice%"=="3" (
    set USERS=500
    set SPAWN=50
    goto :run_headless
)
if "%choice%"=="4" (
    set /p USERS="Enter number of users: "
    set /p SPAWN="Enter spawn rate (users per second): "
    goto :run_headless
)
if "%choice%"=="5" (
    goto :run_web
)

echo Invalid choice!
pause
exit /b 1

:run_headless
echo.
echo Starting load test with %USERS% users (spawn rate: %SPAWN%/s)...
echo Press Ctrl+C to stop.
echo.

python -m locust -f locust/locustfile.py --headless --host=http://localhost:8000 --users %USERS% --spawn-rate %SPAWN% --run-time 300s --csv locust/results/headless

echo.
echo Load test completed! Results saved to locust/results/
pause
exit /b 0

:run_web
echo.
echo Starting Locust web interface...
echo Open http://localhost:8089 in your browser to configure and run tests.
echo Press Ctrl+C to stop.
echo.

python -m locust -f locust/locustfile.py --host=http://localhost:8000

pause
exit /b 0
