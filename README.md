# SENTINEL — Proof-of-Life Biometric Authentication System

> Real-time biometric identity verification using ML-powered liveness detection, deepfake analysis, emotion recognition, and a tamper-proof blockchain ledger — all in one system.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Verification Flow](#verification-flow)
- [ML Pipeline](#ml-pipeline)
- [Scoring Algorithm](#scoring-algorithm)
- [Blockchain Ledger](#blockchain-ledger)
- [API Reference](#api-reference)
- [WebSocket Protocol](#websocket-protocol)
- [Security](#security)
- [Testing](#testing)
- [Author](#author)
- [License](#license)

---

## Overview

SENTINEL is a full-stack biometric authentication platform that verifies a user is a real, present human — not a photo, video replay, or deepfake. It combines three independent ML signals (liveness, deepfake detection, emotion analysis) into a single trust score, records every verification on an immutable blockchain ledger, and issues short-lived JWT tokens on success.

**Key capabilities:**

- **Liveness detection** — MediaPipe FaceLandmarker tracks 478 face landmarks in real time to verify physical head movements and expressions
- **Deepfake detection** — MesoNet-4 CNN analyzes mesoscopic facial features to catch synthetic media
- **Emotion recognition** — DeepFace validates the user can produce requested emotional expressions on demand
- **Blockchain audit trail** — SHA-256 hash chain with RSA-PSS digital signatures creates a tamper-proof verification record
- **Unique blockchain IDs** — Each successful verification produces a `SNTL-XXXXXXXX-XXXX` identifier
- **Time-bound tokens** — RS256-signed JWT tokens expire after exactly 15 minutes

---

## How It Works

1. User authenticates via Clerk (social/email sign-in)
2. User navigates to `/verify-glass` and grants camera access
3. Backend generates a random sequence of 8 challenges (head movements + facial expressions)
4. Frontend captures video frames at 10 FPS and streams them to the backend via WebSocket
5. Backend ML pipeline analyzes each frame for liveness, deepfake indicators, and emotion
6. User must pass at least 75% of challenges (6 out of 8)
7. Scoring engine computes a weighted composite score
8. If score ≥ 0.65, the verification passes:
   - A block is appended to the blockchain ledger
   - A unique `SNTL-XXXXXXXX-XXXX` blockchain ID is generated
   - A 15-minute JWT token is issued
9. Results are displayed with animated score breakdowns and blockchain ID

---

## Architecture

```
┌──────────────────────┐         WebSocket (10 FPS)        ┌──────────────────────┐
│                      │◄─────────────────────────────────►│                      │
│   Frontend           │         REST API                  │   Backend            │
│   Next.js 14         │◄─────────────────────────────────►│   FastAPI            │
│   React 18           │                                    │   Python 3.11        │
│   Tailwind + Motion  │                                    │                      │
└──────────┬───────────┘                                    └──────────┬───────────┘
           │                                                           │
           ▼                                                           ▼
┌──────────────────────┐                                    ┌──────────────────────┐
│   Clerk              │                                    │   ML Pipeline        │
│   Authentication     │                                    │   ├─ MediaPipe       │
│   (JWT + JWKS)       │                                    │   ├─ MesoNet-4       │
└──────────────────────┘                                    │   └─ DeepFace        │
                                                            └──────────┬───────────┘
                                                                       │
                                                            ┌──────────▼───────────┐
                                                            │   Blockchain Ledger  │
                                                            │   SHA-256 + RSA-PSS  │
                                                            │   JSON persistence   │
                                                            └──────────────────────┘
```

---

## Tech Stack

| Layer       | Technology                                                     |
| ----------- | -------------------------------------------------------------- |
| Frontend    | Next.js 14, React 18, TypeScript, Tailwind CSS, Framer Motion |
| Auth        | Clerk (JWKS verification, social + email providers)            |
| Backend     | FastAPI, Python 3.11, Uvicorn                                  |
| ML/CV       | MediaPipe FaceLandmarker, MesoNet-4 (CNN), DeepFace           |
| Crypto      | PyJWT (RS256), `cryptography` (RSA-PSS, SHA-256)              |
| Transport   | WebSocket (real-time), REST (API)                              |
| Storage     | In-memory sessions, JSON blockchain persistence                |
| Testing     | Vitest + Testing Library (frontend), pytest (backend)          |

---

## Getting Started

### Prerequisites

- **Python 3.11+** with `pip`
- **Node.js 18+** with `npm`
- A webcam (required for verification)
- A [Clerk](https://clerk.com) account (free tier works)

### 1. Clone the repository

```bash
git clone https://github.com/ArrinPaul/TechX.git
cd TechX
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv311
# Windows:
venv311\Scripts\activate
# macOS/Linux:
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the MediaPipe face landmarker model
python download_mediapipe_model.py

# Create .env (see Environment Variables section below)

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local (see Environment Variables section below)

# Start the dev server
npm run dev
```

### 4. Quick start (Windows)

```bash
# From the root directory — starts both servers
start-local.bat
```

### 5. Open in browser

| URL                                  | Description              |
| ------------------------------------ | ------------------------ |
| http://localhost:3000                 | Landing page             |
| http://localhost:3000/verify-glass    | Verification scanner     |
| http://localhost:3000/blockchain      | Blockchain ledger viewer |
| http://localhost:3000/profile         | User profile             |
| http://localhost:8000                 | Backend root             |
| http://localhost:8000/docs            | Interactive API docs     |
| http://localhost:8000/health          | Health check             |

---

## Environment Variables

### Backend — `backend/.env`

```bash
# Clerk JWT validation (leave empty to skip in dev)
CLERK_ISSUER_URL=

# MediaPipe model location
MEDIAPIPE_MODEL_PATH=~/.mediapipe_models/face_landmarker.task

# CORS — comma-separated origins
CORS_ORIGINS=http://localhost:3000

# JWT token settings
JWT_EXPIRY_MINUTES=15
JWT_PRIVATE_KEY=          # Auto-generated if empty
JWT_PUBLIC_KEY=           # Auto-generated if empty

# Deepfake detection (requires TensorFlow)
ENABLE_DEEPFAKE_DETECTION=false
DEEPFAKE_MODEL_PATH=

# Session limits
MAX_SESSION_DURATION_SECONDS=120
MAX_CONSECUTIVE_FAILURES=3
CHALLENGE_TIMEOUT_SECONDS=10
```

### Frontend — `frontend/.env.local`

```bash
# Backend connection
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Clerk authentication — get keys from https://dashboard.clerk.com
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
```

> **Security note:** Both `.env` and `.env.local` are in `.gitignore`. Never commit real keys.

---

## Project Structure

```
TechX/
├── frontend/                          # Next.js 14 application
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Landing page
│   │   │   ├── layout.tsx             # Root layout + Clerk provider
│   │   │   ├── globals.css            # Tailwind + custom styles
│   │   │   ├── verify-glass/          # Main verification page
│   │   │   │   └── page.tsx           # Scanner UI + WebSocket client + result screens
│   │   │   ├── blockchain/            # Blockchain ledger explorer
│   │   │   │   └── page.tsx           # Chain viewer with block details
│   │   │   ├── profile/               # User profile page
│   │   │   ├── sign-in/               # Clerk sign-in
│   │   │   └── sign-up/               # Clerk sign-up
│   │   ├── components/
│   │   │   ├── FaceIDScanner.tsx      # Animated face mesh wireframe with HUD
│   │   │   └── GlassCard.tsx          # Glassmorphism card component
│   │   ├── lib/
│   │   │   ├── api.ts                 # REST API client
│   │   │   ├── camera.ts              # Camera capture utilities
│   │   │   └── websocket.ts           # WebSocket connection manager
│   │   ├── middleware.ts              # Clerk auth middleware + public routes
│   │   ├── types/                     # TypeScript type definitions
│   │   └── test/                      # Frontend test suite
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vitest.config.ts
│
├── backend/                           # FastAPI application
│   ├── app/
│   │   ├── main.py                    # FastAPI app, all routes, WebSocket handler
│   │   ├── config.py                  # Environment-based configuration
│   │   ├── models/
│   │   │   └── data_models.py         # Pydantic models + dataclasses
│   │   └── services/
│   │       ├── blockchain_ledger.py   # SHA-256 hash chain + RSA-PSS signatures
│   │       ├── challenge_engine.py    # Random challenge generation + anti-replay
│   │       ├── cv_verifier.py         # MediaPipe face landmark detection
│   │       ├── deepfake_detector.py   # MesoNet-4 CNN deepfake analysis
│   │       ├── emotion_analyzer.py    # DeepFace emotion recognition
│   │       ├── scoring_engine.py      # Weighted composite scoring
│   │       ├── session_manager.py     # Session lifecycle management
│   │       ├── token_issuer.py        # RS256 JWT token generation
│   │       └── database_service.py    # In-memory data store + nonce tracking
│   ├── tests/                         # pytest test suite
│   ├── data/                          # Blockchain ledger JSON files
│   │   ├── verification_ledger.json   # Block chain data
│   │   └── ledger_keys.json           # RSA key pair (auto-generated)
│   ├── requirements.txt               # Python dependencies
│   └── pytest.ini                     # Test configuration
│
├── start-local.bat                    # Windows quick-start script
└── README.md                          # This file
```

---

## Verification Flow

```
User clicks "Begin Verification"
         │
         ▼
┌─────────────────────────────────┐
│ POST /api/auth/verify           │  Creates session + generates 8 challenges
│ ← session_id, challenges[]     │  (random mix of gestures + expressions)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ WS /ws/verify/{session_id}     │  Frontend opens WebSocket
│                                 │
│   For each challenge:           │
│   1. Display instruction        │  "Turn your head left", "Smile", etc.
│   2. Stream frames (10 FPS)     │  Base64 JPEG → backend
│   3. ML analysis per frame      │  Liveness + deepfake + emotion
│   4. Return pass/fail + scores  │  Real-time feedback to UI
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Challenge pass rate check       │  Must pass ≥ 75% (6 of 8)
│                                 │
│ Scoring engine computes:        │
│   0.4×liveness + 0.25×deepfake  │
│   + 0.35×emotion                │
│                                 │
│ Threshold: ≥ 0.65 to pass      │
└────────────────┬────────────────┘
                 │
          ┌──────┴──────┐
          │             │
       PASS           FAIL
          │             │
          ▼             ▼
  ┌──────────────┐  ┌──────────────┐
  │ Blockchain   │  │ Failure      │
  │ block added  │  │ recorded     │
  │ SNTL ID      │  │              │
  │ issued       │  │ "Try again"  │
  │ JWT token    │  │ screen       │
  │ generated    │  └──────────────┘
  └──────────────┘
```

---

## ML Pipeline

### 1. Liveness Detection — MediaPipe FaceLandmarker

- Tracks **478 face landmarks** in real time
- Detects head pose (yaw, pitch, roll) to verify gestures like nodding, turning, tilting
- Verifies physical presence — photos and static images fail movement challenges
- Model: `face_landmarker.task` (~4MB, downloaded to `~/.mediapipe_models/`)

### 2. Deepfake Detection — MesoNet-4

- 4-layer CNN designed to capture mesoscopic artifacts in synthetic faces
- Analyzes facial textures at a level between pixel-level and semantic-level
- Detects GAN-generated and face-swapped content
- Optional — requires TensorFlow (`ENABLE_DEEPFAKE_DETECTION=true`)

### 3. Emotion Recognition — DeepFace

- Analyzes 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
- Validates the user can produce a specific expression on demand
- Expression pools: `smile`, `frown`, `surprised`, `neutral`, `angry`
- A pre-recorded video cannot dynamically respond to random expression requests

### Challenge Types

| Type       | Pool Size | Examples                                                         |
| ---------- | --------- | ---------------------------------------------------------------- |
| Gestures   | 10        | Nod up/down, turn left/right, tilt left/right, blink, open mouth |
| Expressions| 5         | Smile, frown, surprised, neutral, angry                          |

Each session generates **8 random challenges** from both pools using `secrets.choice()`.

---

## Scoring Algorithm

The scoring engine combines three independent ML signals into a single trust score using a weighted formula:

$$\text{final\_score} = 0.4 \times \text{liveness} + 0.25 \times \text{deepfake} + 0.35 \times \text{emotion}$$

| Component | Weight | Source                    | Measures                       |
| --------- | ------ | ------------------------- | ------------------------------ |
| Liveness  | 40%    | MediaPipe FaceLandmarker  | Physical presence, head motion |
| Deepfake  | 25%    | MesoNet-4 CNN             | Synthetic media artifacts      |
| Emotion   | 35%    | DeepFace                  | Voluntary expression control   |

**Pass criteria:**
- Final score ≥ **0.65** (65%)
- Challenge pass rate ≥ **75%** (6 of 8 challenges)

Both conditions must be met for verification to succeed.

---

## Blockchain Ledger

Every verification (pass or fail) is recorded as a block on an immutable hash chain.

### Block Structure

```json
{
  "index": 3,
  "timestamp": 1718234567.89,
  "block_id": "uuid-v4",
  "previous_hash": "sha256-of-previous-block",
  "data": {
    "session_id": "...",
    "user_id": "...",
    "blockchain_id": "SNTL-A3F7B2C1-9D4E",
    "final_score": 0.82,
    "passed": true,
    "liveness_score": 0.91,
    "deepfake_score": 0.75,
    "emotion_score": 0.78,
    "challenges_passed": 7,
    "challenges_total": 8
  },
  "nonce": "cryptographic-nonce",
  "block_hash": "sha256-of-this-block",
  "signature": "rsa-pss-signature-base64"
}
```

### Properties

- **Immutability** — Each block's hash includes the previous block's hash. Modifying any block breaks the chain.
- **Digital signatures** — Every block is signed with the server's RSA private key (RSA-PSS, SHA-256).
- **Independent verification** — Anyone with the public key (`GET /api/blockchain/public-key`) can verify any block's signature.
- **SNTL IDs** — Successful verifications generate a unique `SNTL-XXXXXXXX-XXXX` blockchain ID.
- **Persistence** — Chain is stored as JSON in `backend/data/verification_ledger.json`.

---

## API Reference

### REST Endpoints

| Method | Path                                | Description                            |
| ------ | ----------------------------------- | -------------------------------------- |
| `GET`  | `/`                                 | API info                               |
| `GET`  | `/health`                           | Health check + service status          |
| `POST` | `/api/auth/verify`                  | Create verification session            |
| `POST` | `/api/token/validate`               | Validate an issued JWT token           |
| `GET`  | `/api/blockchain/stats`             | Chain statistics (total blocks, etc.)  |
| `GET`  | `/api/blockchain/chain`             | List all blocks                        |
| `GET`  | `/api/blockchain/block/{index}`     | Get specific block by index            |
| `GET`  | `/api/blockchain/verify`            | Verify entire chain integrity          |
| `GET`  | `/api/blockchain/verify/{index}`    | Verify specific block                  |
| `GET`  | `/api/blockchain/proof/{index}`     | Get cryptographic proof for block      |
| `GET`  | `/api/blockchain/session/{id}`      | Look up blocks by session ID           |
| `GET`  | `/api/blockchain/lookup/{sntl_id}`  | Look up block by SNTL blockchain ID    |
| `GET`  | `/api/blockchain/public-key`        | Export RSA public key (PEM)            |

### Health Check Response

```json
{
  "status": "healthy",
  "services": {
    "api": "operational",
    "database": "operational",
    "blockchain_ledger": "operational"
  },
  "blockchain": {
    "total_blocks": 3
  }
}
```

---

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/verify/{session_id}`

### Client → Server

```json
{
  "type": "frame",
  "data": "base64-encoded-jpeg",
  "challenge_index": 0
}
```

### Server → Client

```json
{
  "type": "challenge_result",
  "challenge_index": 0,
  "passed": true,
  "liveness_score": 0.92,
  "deepfake_score": 0.88,
  "emotion_score": 0.85,
  "feedback": "Challenge passed!"
}
```

### Final Result

```json
{
  "type": "verification_complete",
  "passed": true,
  "final_score": 0.82,
  "blockchain_id": "SNTL-A3F7B2C1-9D4E",
  "token": "eyJhbGciOiJSUzI1NiIs...",
  "scores": {
    "liveness": 0.91,
    "deepfake": 0.75,
    "emotion": 0.78
  }
}
```

---

## Security

| Measure                     | Implementation                                                    |
| --------------------------- | ----------------------------------------------------------------- |
| Authentication              | Clerk JWT with JWKS verification                                  |
| Anti-replay                 | Cryptographic nonces (`secrets.token_hex(16)`) per session        |
| Challenge randomness        | `secrets.choice()` — cryptographically secure PRNG                |
| Token signing               | RS256 (RSA 2048-bit) with 15-minute expiry                        |
| Blockchain signatures       | RSA-PSS with SHA-256                                              |
| CORS                        | Configurable origins, locked to `localhost:3000` by default       |
| Session limits              | 120-second max duration, 3 max consecutive failures               |
| No credential storage       | `.env` / `.env.local` are gitignored, no keys in source control   |

---

## Testing

### Backend

```bash
cd backend
venv311\Scripts\activate
pytest                      # Run full test suite
pytest -v                   # Verbose output
pytest tests/test_scoring_engine.py  # Run specific test file
```

### Frontend

```bash
cd frontend
npm test                    # Run Vitest
npm run test:ui             # Vitest with browser UI
```

### Test coverage includes:

- Scoring engine (weighted formula, threshold, edge cases)
- Challenge engine (randomness, nonce generation, pool sizes)
- Session manager (lifecycle, timeouts, failure limits)
- Token issuer (RS256 signing, validation, expiry)
- Blockchain ledger (hash chain integrity, signatures, SNTL IDs)
- Database service (CRUD, nonce tracking)
- Deepfake detector (model loading, inference)
- Emotion analyzer (expression detection)
- CV verifier (face landmark detection)
- Integration tests (end-to-end verification flow)

---

## Author

**Arrin Paul** — [GitHub](https://github.com/ArrinPaul)

## License

MIT
