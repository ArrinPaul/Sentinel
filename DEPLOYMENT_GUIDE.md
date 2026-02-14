# Proof-of-Life Authentication System - Deployment Guide

## Overview
This guide covers local testing and production deployment of the complete proof-of-life authentication system with frontend and backend integration.

## Architecture
- **Frontend**: Next.js 14 (React) with Convex for data persistence
- **Backend**: FastAPI (Python) with WebSocket support for real-time verification
- **ML Models**: MediaPipe for face detection, custom models for liveness/deepfake detection
- **Database**: Convex (cloud-hosted)

---

## Local Development Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- Git

### 1. Clone and Setup

```bash
# Already cloned at D:\TechX
cd D:\TechX
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download ML models (if not already present)
# MediaPipe model will be downloaded automatically on first run
# Or manually download to ~/.mediapipe_models/face_landmarker.task

# Verify .env configuration
# Make sure CONVEX_URL and CORS_ORIGINS are set correctly
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install dependencies
npm install

# Verify .env.local configuration
# Make sure NEXT_PUBLIC_API_URL and NEXT_PUBLIC_WS_URL point to backend
```

### 4. Start Services

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 - Convex (if needed):**
```bash
cd frontend
npx convex dev
```

### 5. Test the Integration

1. Open browser to `http://localhost:3000`
2. Navigate to `/verify-glass` page
3. Click "Start Verification"
4. Allow camera access
5. Follow on-screen challenges
6. Verify real-time scores and feedback

---

## Production Deployment

### Option 1: Cloud Deployment (Recommended)

#### Frontend Deployment (Vercel)

1. **Push to GitHub** (already done)
   ```bash
   git push origin main
   ```

2. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository: `https://github.com/ArrinPaul/Proof-of-life.git`
   - Configure project:
     - Framework: Next.js
     - Root Directory: `frontend`
     - Build Command: `npm run build`
     - Output Directory: `.next`
   
3. **Set Environment Variables in Vercel:**
   ```
   NEXT_PUBLIC_CONVEX_URL=https://keen-lion-797.convex.cloud
   NEXT_PUBLIC_API_URL=https://your-backend-url.com
   NEXT_PUBLIC_WS_URL=wss://your-backend-url.com
   NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
   CLERK_SECRET_KEY=sk_test_...
   CONVEX_DEPLOYMENT=prod:keen-lion-797
   ```

4. **Deploy:** Click "Deploy" and wait for build to complete

#### Backend Deployment (Railway/Render/AWS)

**Option A: Railway (Easiest)**

1. Go to [railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Select `backend` directory as root
4. Add environment variables:
   ```
   CONVEX_URL=https://keen-lion-797.convex.cloud
   JWT_SECRET_KEY=your-production-secret-key
   JWT_ALGORITHM=RS256
   JWT_EXPIRY_MINUTES=15
   SESSION_TIMEOUT_SECONDS=120
   MAX_FAILED_ATTEMPTS=3
   CORS_ORIGINS=https://your-frontend-url.vercel.app
   USE_WSS=true
   WEBSOCKET_HOST=your-backend-url.railway.app
   ```
5. Railway will auto-detect Python and deploy

**Option B: Render**

1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Configure:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (same as Railway)

**Option C: AWS EC2**

1. Launch EC2 instance (Ubuntu 22.04)
2. SSH into instance
3. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv nginx -y
   ```
4. Clone repository and setup:
   ```bash
   git clone https://github.com/ArrinPaul/Proof-of-life.git
   cd Proof-of-life/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. Create systemd service for auto-restart
6. Configure Nginx as reverse proxy
7. Setup SSL with Let's Encrypt

### Option 2: Docker Deployment

#### Create Docker Files

**Backend Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - CONVEX_URL=${CONVEX_URL}
      - CORS_ORIGINS=http://localhost:3000
    volumes:
      - ./backend:/app

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000
      - NEXT_PUBLIC_CONVEX_URL=${NEXT_PUBLIC_CONVEX_URL}
    depends_on:
      - backend
```

**Deploy with Docker:**
```bash
docker-compose up -d
```

---

## Environment Variables Reference

### Frontend (.env.local)
```bash
# Convex
NEXT_PUBLIC_CONVEX_URL=https://keen-lion-797.convex.cloud
CONVEX_DEPLOYMENT=prod:keen-lion-797

# Backend API (update for production)
NEXT_PUBLIC_API_URL=http://localhost:8000  # Change to production URL
NEXT_PUBLIC_WS_URL=ws://localhost:8000     # Change to wss://production-url

# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
```

### Backend (.env)
```bash
# Convex
CONVEX_URL=https://keen-lion-797.convex.cloud

# JWT
JWT_SECRET_KEY=your-secret-key-here  # Change for production!
JWT_ALGORITHM=RS256
JWT_EXPIRY_MINUTES=15

# Session
SESSION_TIMEOUT_SECONDS=120
MAX_FAILED_ATTEMPTS=3

# ML Models
MEDIAPIPE_MODEL_PATH=~/.mediapipe_models/face_landmarker.task
DEEPFAKE_MODEL_PATH=~/.deepfake_models/mesonet4_weights.h5

# CORS (update for production)
CORS_ORIGINS=http://localhost:3000  # Add production frontend URL

# WebSocket (for production)
USE_WSS=true
WEBSOCKET_HOST=your-backend-domain.com
```

---

## Testing the Deployment

### Health Checks

**Backend:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","services":{"api":"operational","database":"operational"}}
```

**Frontend:**
```bash
curl http://localhost:3000
# Should return HTML
```

### Integration Test

1. Open frontend URL in browser
2. Navigate to `/verify-glass`
3. Click "Start Verification"
4. Allow camera access
5. Complete challenges
6. Verify token issuance

### WebSocket Test

```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/ws/verify/test-session-id');
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', e.data);
```

---

## Monitoring and Maintenance

### Logs

**Backend:**
```bash
# Local
tail -f backend/logs/app.log

# Docker
docker logs -f proof-of-life-backend

# Production (systemd)
journalctl -u proof-of-life-backend -f
```

**Frontend:**
```bash
# Vercel
# Check dashboard at vercel.com

# Local
npm run dev  # Shows logs in terminal
```

### Database (Convex)

- Monitor at: https://dashboard.convex.dev
- View tables: verification_results, nonces, audit_logs, token_issuance

### Performance Monitoring

- Frontend: Vercel Analytics
- Backend: Add APM tool (New Relic, DataDog, etc.)
- Uptime: UptimeRobot or Pingdom

---

## Security Checklist

- [ ] Change JWT_SECRET_KEY for production
- [ ] Update CORS_ORIGINS to production frontend URL only
- [ ] Enable HTTPS/WSS for production
- [ ] Set up rate limiting on backend
- [ ] Configure firewall rules
- [ ] Enable Convex authentication
- [ ] Rotate Clerk API keys
- [ ] Set up monitoring and alerts
- [ ] Regular security audits
- [ ] Keep dependencies updated

---

## Troubleshooting

### Issue: WebSocket connection fails

**Solution:**
- Check CORS_ORIGINS includes frontend URL
- Verify WebSocket URL uses correct protocol (ws:// or wss://)
- Check firewall allows WebSocket connections
- Ensure backend is running and accessible

### Issue: Camera not working

**Solution:**
- Verify HTTPS is enabled (required for camera access)
- Check browser permissions
- Test on different browsers
- Verify camera is not in use by another app

### Issue: ML models not loading

**Solution:**
- Check model paths in .env
- Download models manually if needed
- Verify file permissions
- Check disk space

### Issue: CORS errors

**Solution:**
- Add frontend URL to CORS_ORIGINS in backend .env
- Restart backend after changing .env
- Clear browser cache
- Check browser console for exact error

---

## Support and Resources

- **GitHub Repository:** https://github.com/ArrinPaul/Proof-of-life.git
- **Convex Dashboard:** https://dashboard.convex.dev
- **Documentation:** See README.md in each directory

---

## Quick Start Commands

**Local Development:**
```bash
# Terminal 1 - Backend
cd backend && venv\Scripts\activate && uvicorn app.main:app --reload

# Terminal 2 - Frontend
cd frontend && npm run dev

# Terminal 3 - Convex
cd frontend && npx convex dev
```

**Production Check:**
```bash
# Test backend
curl https://your-backend-url.com/health

# Test frontend
curl https://your-frontend-url.vercel.app

# Test WebSocket
wscat -c wss://your-backend-url.com/ws/verify/test
```

---

## Next Steps

1. ✅ Local testing complete
2. ⬜ Deploy backend to Railway/Render
3. ⬜ Deploy frontend to Vercel
4. ⬜ Update environment variables
5. ⬜ Test production deployment
6. ⬜ Set up monitoring
7. ⬜ Configure custom domain (optional)
8. ⬜ Enable SSL/TLS
9. ⬜ Set up CI/CD pipeline
10. ⬜ Production launch! 🚀
