@echo off
echo ========================================
echo Proof-of-Life Backend AWS Deployment
echo ========================================
echo.

REM Configuration
set AWS_REGION=us-east-1
set ECR_REPO_NAME=proof-of-life-backend
set ECS_CLUSTER_NAME=proof-of-life-cluster
set TASK_FAMILY=proof-of-life-task

echo Checking AWS CLI...
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: AWS CLI is not installed
    echo Please install AWS CLI from: https://aws.amazon.com/cli/
    pause
    exit /b 1
)

echo Checking Docker...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not running
    echo Please install Docker Desktop and ensure it's running
    pause
    exit /b 1
)

echo.
echo Step 1: Getting AWS Account ID...
for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set AWS_ACCOUNT_ID=%%i
if "%AWS_ACCOUNT_ID%"=="" (
    echo Error: Failed to get AWS Account ID
    echo Please run: aws configure
    pause
    exit /b 1
)
echo AWS Account ID: %AWS_ACCOUNT_ID%

set ECR_URI=%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%
echo ECR URI: %ECR_URI%

echo.
echo Step 2: Creating ECR repository...
aws ecr create-repository --repository-name %ECR_REPO_NAME% --region %AWS_REGION% 2>nul
if %errorlevel% equ 0 (
    echo ECR repository created successfully
) else (
    echo ECR repository already exists or creation failed
)

echo.
echo Step 3: Logging in to ECR...
for /f "tokens=*" %%i in ('aws ecr get-login-password --region %AWS_REGION%') do set ECR_PASSWORD=%%i
echo %ECR_PASSWORD% | docker login --username AWS --password-stdin %ECR_URI%
if %errorlevel% neq 0 (
    echo Error: Failed to login to ECR
    pause
    exit /b 1
)

echo.
echo Step 4: Building Docker image...
docker build -t %ECR_REPO_NAME%:latest .
if %errorlevel% neq 0 (
    echo Error: Docker build failed
    pause
    exit /b 1
)

echo.
echo Step 5: Tagging image...
docker tag %ECR_REPO_NAME%:latest %ECR_URI%:latest
if %errorlevel% neq 0 (
    echo Error: Failed to tag image
    pause
    exit /b 1
)

echo.
echo Step 6: Pushing image to ECR...
docker push %ECR_URI%:latest
if %errorlevel% neq 0 (
    echo Error: Failed to push image to ECR
    pause
    exit /b 1
)

echo.
echo ========================================
echo Docker Image Deployed Successfully!
echo ========================================
echo.
echo ECR Image: %ECR_URI%:latest
echo.
echo Next Steps:
echo 1. Create ECS cluster:
echo    aws ecs create-cluster --cluster-name %ECS_CLUSTER_NAME% --region %AWS_REGION%
echo.
echo 2. Create CloudWatch log group:
echo    aws logs create-log-group --log-group-name /ecs/proof-of-life --region %AWS_REGION%
echo.
echo 3. Update task-definition.json with ECR URI: %ECR_URI%:latest
echo.
echo 4. Register task definition:
echo    aws ecs register-task-definition --cli-input-json file://task-definition.json --region %AWS_REGION%
echo.
echo 5. Create Application Load Balancer via AWS Console
echo.
echo 6. Create ECS Service with the ALB
echo.
echo See DEPLOYMENT_STATUS.md for detailed instructions
echo.
pause
