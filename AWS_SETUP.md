# AWS Setup Guide for Backend Deployment

## Step 1: Install AWS CLI

### Windows Installation
1. Download the AWS CLI MSI installer:
   - 64-bit: https://awscli.amazonaws.com/AWSCLIV2.msi
   
2. Run the installer:
   ```cmd
   msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
   ```

3. Verify installation:
   ```cmd
   aws --version
   ```
   You should see: `aws-cli/2.x.x Python/3.x.x Windows/...`

## Step 2: Get AWS Credentials

You need an AWS account with appropriate permissions. If you don't have one:

1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow the registration process

### Create IAM User for Deployment

1. Log in to AWS Console: https://console.aws.amazon.com/
2. Go to IAM (Identity and Access Management)
3. Click "Users" → "Add users"
4. User name: `proof-of-life-deployer`
5. Select "Programmatic access"
6. Attach policies:
   - AmazonECS_FullAccess
   - AmazonEC2ContainerRegistryFullAccess
   - CloudWatchLogsFullAccess
   - IAMReadOnlyAccess
7. Click "Next" → "Create user"
8. **IMPORTANT**: Save the Access Key ID and Secret Access Key

## Step 3: Configure AWS CLI

Run the configuration command:
```bash
aws configure
```

Enter the following when prompted:
```
AWS Access Key ID: [Your Access Key ID]
AWS Secret Access Key: [Your Secret Access Key]
Default region name: us-east-1
Default output format: json
```

## Step 4: Verify Configuration

Test your AWS CLI setup:
```bash
# Check your identity
aws sts get-caller-identity

# List ECR repositories (should return empty list if none exist)
aws ecr describe-repositories --region us-east-1
```

## Step 5: Install Docker Desktop

The backend deployment requires Docker to build container images.

1. Download Docker Desktop for Windows:
   https://www.docker.com/products/docker-desktop/

2. Install and start Docker Desktop

3. Verify Docker is running:
   ```cmd
   docker --version
   docker ps
   ```

## Step 6: Deploy Backend

Once AWS CLI and Docker are set up:

```bash
cd backend
bash aws-deploy.sh
```

Or if bash is not available on Windows, run the commands manually:

```cmd
cd backend

# Get AWS account ID
aws sts get-caller-identity --query Account --output text

# Set variables (replace YOUR_ACCOUNT_ID with actual ID)
set AWS_ACCOUNT_ID=YOUR_ACCOUNT_ID
set AWS_REGION=us-east-1
set ECR_REPO_NAME=proof-of-life-backend

# Create ECR repository
aws ecr create-repository --repository-name %ECR_REPO_NAME% --region %AWS_REGION%

# Login to ECR
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

# Build Docker image
docker build -t %ECR_REPO_NAME%:latest .

# Tag image
docker tag %ECR_REPO_NAME%:latest %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:latest

# Push to ECR
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/%ECR_REPO_NAME%:latest
```

## Troubleshooting

### AWS CLI Not Found
- Restart your terminal/PowerShell after installation
- Check if AWS CLI is in your PATH

### Docker Not Running
- Start Docker Desktop
- Wait for Docker to fully start (check system tray icon)

### Permission Denied
- Ensure your IAM user has the required permissions
- Check if MFA is required for your account

### ECR Login Failed
- Verify your AWS credentials are correct
- Check if your region is correct (us-east-1)
- Ensure your IAM user has ECR permissions

## Cost Considerations

AWS services used:
- **ECR**: $0.10 per GB-month for storage
- **ECS Fargate**: ~$0.04 per vCPU per hour + ~$0.004 per GB per hour
- **Application Load Balancer**: ~$0.0225 per hour + data processing charges
- **CloudWatch Logs**: $0.50 per GB ingested

Estimated monthly cost for minimal usage: $15-30

## Alternative: AWS Free Tier

If you're within the AWS Free Tier (first 12 months):
- 750 hours of ECS Fargate per month (limited)
- 50 GB of ECR storage
- Some ALB hours included

Check: https://aws.amazon.com/free/
