# Proof-of-Life Authentication System

A comprehensive biometric authentication system with real-time liveness detection, emotion analysis, and deepfake detection using ML models.

## 🚀 Quick Start

### Local Development

1. **Start all services:**
   ```bash
   # Windows
   start-local.bat
   
   # Or manually:
   # Terminal 1: cd backend && venv\Scripts\activate && uvicorn app.main:app --reload
   # Terminal 2: cd frontend && npm run dev
   ```

2. **Open browser:**
   - Frontend: http://localhost:3000
   - Verification: http://localhost:3000/verify-glass
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

3. **Test the system:**
   - Click "Start Verification"
   - Allow camera access
   - Follow on-screen challenges
   - Watch real-time scores update

## 📋 Features

### Frontend
- ✅ Next.js 14 with TypeScript
- ✅ Real-time WebSocket communication
- ✅ Camera capture at 10 FPS
- ✅ Glassmorphism UI with Framer Motion
- ✅ Clerk authentication integration
- ✅ Convex database for persistence
- ✅ Comprehensive test suite (48 tests)

### Backend
- ✅ FastAPI with WebSocket support
- ✅ ML-powered verification pipeline
- ✅ MediaPipe face detection
- ✅ Liveness detection
- ✅ Emotion analysis
- ✅ Deepfake detection
- ✅ JWT token issuance
- ✅ Audit logging
- ✅ Replay attack prevention

## 🏗️ Architecture

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│                 │◄──────────────────────────►│                 │
│   Frontend      │         HTTP API           │    Backend      │
│   (Next.js)     │◄──────────────────────────►│   (FastAPI)     │
│                 │                             │                 │
└────────┬────────┘                             └────────┬────────┘
         │                                               │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                             ┌─────────────────┐
│     Convex      │                             │   ML Models     │
│   (Database)    │                             │  - MediaPipe    │
│                 │                             │  - Liveness     │
└─────────────────┘                             │  - Deepfake     │
                                                └─────────────────┘
```

## 📁 Project Structure

```
D:\TechX\
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # Next.js app router
│   │   ├── components/      # React components
│   │   ├── lib/             # Core libraries
│   │   │   ├── api.ts       # API client
│   │   │   ├── camera.ts    # Camera capture
│   │   │   └── websocket.ts # WebSocket client
│   │   └── test/            # Test files
│   ├── convex/              # Convex database schema
│   └── package.json
│
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── services/        # Business logic
│   │   └── models/          # Data models
│   ├── requirements.txt
│   └── .env
│
├── .kiro/                    # Spec files
│   └── specs/
│       └── backend-frontend-integration/
│
├── DEPLOYMENT_GUIDE.md       # Deployment instructions
├── start-local.bat           # Quick start script
└── README.md                 # This file
```

## 🔧 Configuration

### Frontend Environment Variables

Create `frontend/.env.local`:
```bash
# Convex
NEXT_PUBLIC_CONVEX_URL=https://keen-lion-797.convex.cloud
CONVEX_DEPLOYMENT=dev:keen-lion-797

# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_key_here
CLERK_SECRET_KEY=your_secret_here
```

### Backend Environment Variables

Create `backend/.env`:
```bash
# Convex
CONVEX_URL=https://keen-lion-797.convex.cloud

# JWT
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=RS256
JWT_EXPIRY_MINUTES=15

# CORS
CORS_ORIGINS=http://localhost:3000
```

## 🧪 Testing

### Run Frontend Tests
```bash
cd frontend
npm test
```

**Test Coverage:**
- 9 test files
- 48 tests total
- API client tests
- Camera capture tests
- WebSocket tests
- Property-based tests
- Integration tests

### Test Backend
```bash
cd backend
pytest
```

## 📦 Installation

### Prerequisites
- Node.js 18+
- Python 3.9+
- Git

### Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/ArrinPaul/Proof-of-life.git
   cd Proof-of-life
   ```

2. **Backend setup:**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Frontend setup:**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure environment variables** (see Configuration section)

5. **Start services** (see Quick Start section)

## 🚀 Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed deployment instructions including:
- Local development setup
- Production deployment (Vercel + Railway/Render)
- Docker deployment
- Environment configuration
- Security checklist
- Troubleshooting

## 🔐 Security Features

- JWT token-based authentication
- Replay attack prevention with nonces
- CORS protection
- Rate limiting
- Audit logging
- Secure WebSocket (WSS) support
- Camera permission handling
- Session timeout management

## 📊 Monitoring

- **Frontend:** Vercel Analytics
- **Backend:** Health check endpoint at `/health`
- **Database:** Convex Dashboard
- **Logs:** Audit logs stored in Convex

## 🛠️ Tech Stack

### Frontend
- Next.js 14
- TypeScript
- Tailwind CSS
- Framer Motion
- Convex
- Clerk
- Vitest

### Backend
- FastAPI
- Python 3.9+
- WebSockets
- MediaPipe
- OpenCV
- NumPy
- JWT

## 📝 API Documentation

### REST Endpoints

**Create Session:**
```http
POST /api/auth/verify
Content-Type: application/json

{
  "user_id": "user_123"
}

Response:
{
  "session_id": "uuid",
  "websocket_url": "ws://localhost:8000/ws/verify/uuid",
  "message": "Session created successfully"
}
```

**Validate Token:**
```http
POST /api/token/validate
Content-Type: application/json

{
  "token": "jwt_token_here"
}

Response:
{
  "valid": true,
  "user_id": "user_123",
  "session_id": "uuid",
  "issued_at": 1234567890,
  "expires_at": 1234567890
}
```

### WebSocket Messages

**Client → Server:**
```json
{
  "type": "video_frame",
  "frame": "base64_encoded_image",
  "timestamp": 1234567890
}
```

**Server → Client:**
```json
{
  "type": "CHALLENGE_ISSUED",
  "message": "Please smile",
  "data": {
    "challenge_id": "uuid",
    "instruction": "Please smile",
    "timeout_seconds": 10
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👥 Authors

- Arrin Paul - [GitHub](https://github.com/ArrinPaul)

## 🙏 Acknowledgments

- MediaPipe for face detection
- Convex for database
- Clerk for authentication
- FastAPI and Next.js communities

## 📞 Support

- GitHub Issues: https://github.com/ArrinPaul/Proof-of-life/issues
- Documentation: See DEPLOYMENT_GUIDE.md

## 🎯 Roadmap

- [ ] Mobile app support
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Custom challenge creation
- [ ] Webhook integrations
- [ ] API rate limiting
- [ ] Enhanced security features

---

**Status:** ✅ Production Ready

**Last Updated:** February 2026

**Version:** 1.0.0
