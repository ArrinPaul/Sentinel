@echo off
echo ========================================
echo Proof-of-Life Frontend Deployment
echo ========================================
echo.

cd frontend

echo Step 1: Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo Error: npm install failed
    pause
    exit /b 1
)

echo.
echo Step 2: Building application...
call npm run build
if %errorlevel% neq 0 (
    echo Error: Build failed
    pause
    exit /b 1
)

echo.
echo Step 3: Deploying to Vercel...
call vercel --prod
if %errorlevel% neq 0 (
    echo Error: Vercel deployment failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Frontend deployed successfully!
echo ========================================
echo.
echo IMPORTANT: Copy the Vercel URL and update:
echo 1. backend/.env.production - CORS_ORIGINS
echo 2. Then deploy the backend
echo.
pause
