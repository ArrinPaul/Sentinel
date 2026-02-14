# Production Deployment Guide

## Prerequisites
- Vercel CLI installed (`npm i -g vercel`)
- AWS CLI installed and configured
- Docker installed
- Convex account (already configured)
- Clerk account (already configured)

## Step 1: Deploy Frontend to Vercel

### 1.1 Navigate to frontend directory
```bash
cd frontend
```

### 1.2 Deploy to Vercel
```bash
vercel --prod
```

### 1.3 Note your Vercel URL
After deployment, Vercel will provide a URL like: `https://your-app.vercel.app`

## Step 2: Deploy Backend to AWS

### 2.1 Install AWS CLI (if not installed)
Windows: Download from https://aws.amazon.com/cli/
```bash
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

### 2.2 Configure AWS CLI
```bash
aws configure
```
Enter your:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: us-east-1
- Default output format: json

### 2.3 Update backend environment variables
Edit `backend/.env.production` and update:
- `JWT_SECRET_KEY`: Generate a secure key
- `CORS_ORIGINS`: Add your Vercel URL from Step 1.3

### 2.4 Run deployment script
```bash
cd backend
bash aws-deploy.sh
```

### 2.5 Create ECS Infrastructure
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name proof-of-life-cluster --region us-east-1

# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/proof-of-life --region us-east-1

# Register task definition (update task-definition.json with your ECR URI first)
aws ecs register-task-definition --cli-input-json file://task-definition.json --region us-east-1

# Create Application Load Balancer (ALB)
# This requires VPC and subnet configuration - see AWS Console for easier setup
```

### 2.6 Note your backend URL
After ALB setup, you'll get a URL like: `https://your-alb.us-east-1.elb.amazonaws.com`

## Step 3: Update Cross-References

### 3.1 Update Frontend Environment Variables
```bash
cd frontend
vercel env rm NEXT_PUBLIC_API_URL production
vercel env rm NEXT_PUBLIC_WS_URL production

# Add new values with your actual backend URL
echo "https://your-alb.us-east-1.elb.amazonaws.com" | vercel env add NEXT_PUBLIC_API_URL production
echo "wss://your-alb.us-east-1.elb.amazonaws.com" | vercel env add NEXT_PUBLIC_WS_URL production
```

### 3.2 Redeploy Frontend
```bash
vercel --prod
```

### 3.3 Update Backend CORS
Update the ECS task definition environment variable `CORS_ORIGINS` with your Vercel URL and redeploy.

## Step 4: Verify Deployment

### 4.1 Test Frontend
Visit your Vercel URL and check:
- Page loads correctly
- Clerk authentication works
- Convex database connection works

### 4.2 Test Backend
```bash
curl https://your-backend-url/health
```

### 4.3 Test Integration
- Start a verification session from the frontend
- Check that WebSocket connection establishes
- Verify face detection and challenges work

## Environment Variables Summary

### Frontend (.env.production)
- `CONVEX_DEPLOYMENT`: dev:keen-lion-797
- `NEXT_PUBLIC_CONVEX_URL`: https://keen-lion-797.convex.cloud
- `NEXT_PUBLIC_CONVEX_SITE_URL`: https://keen-lion-797.convex.site
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`: pk_test_...
- `CLERK_SECRET_KEY`: sk_test_...
- `NEXT_PUBLIC_API_URL`: Your AWS backend URL
- `NEXT_PUBLIC_WS_URL`: Your AWS backend WebSocket URL

### Backend (.env.production)
- `CONVEX_URL`: https://keen-lion-797.convex.cloud
- `JWT_SECRET_KEY`: Secure production key
- `JWT_ALGORITHM`: RS256
- `JWT_EXPIRY_MINUTES`: 15
- `SESSION_TIMEOUT_SECONDS`: 120
- `MAX_FAILED_ATTEMPTS`: 3
- `CORS_ORIGINS`: Your Vercel frontend URL
- `USE_WSS`: true

## Troubleshooting

### Frontend Issues
- Check Vercel deployment logs: `vercel logs`
- Verify environment variables: `vercel env ls`

### Backend Issues
- Check ECS logs in CloudWatch
- Verify security groups allow traffic on port 8000
- Check ALB health checks

### Integration Issues
- Verify CORS settings match frontend URL
- Check WebSocket connection in browser console
- Verify Convex and Clerk credentials are correct
