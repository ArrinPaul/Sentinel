@echo off
echo ========================================
echo Proof-of-Life Authentication System
echo Local Development Startup
echo ========================================
echo.

echo Starting Backend Server...
start "Backend Server" cmd /k "cd backend && venv311\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 5 /nobreak > nul

echo Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo Services Starting...
echo ========================================
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo Verify:   http://localhost:3000/verify-glass
echo ========================================
echo.
echo Press any key to open browser...
pause > nul

start http://localhost:3000/verify-glass

echo.
echo All services started!
echo Close this window or press Ctrl+C to stop
pause
