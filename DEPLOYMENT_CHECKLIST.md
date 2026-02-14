# Deployment Checklist

## Pre-Deployment

### Code Quality
- [x] All tests passing (48/48 frontend tests)
- [x] No console errors in development
- [x] Code reviewed and documented
- [x] Git repository up to date
- [x] README.md complete

### Configuration
- [ ] Production environment variables set
- [ ] JWT secret key changed from default
- [ ] CORS origins updated for production
- [ ] Database connection verified
- [ ] ML models downloaded and accessible

### Security
- [ ] Secrets not committed to Git
- [ ] HTTPS/WSS enabled for production
- [ ] Rate limiting configured
- [ ] Firewall rules set
- [ ] Security headers configured

## Deployment Steps

### 1. Backend Deployment

#### Option A: Railway
- [ ] Create Railway account
- [ ] Connect GitHub repository
- [ ] Set root directory to `backend`
- [ ] Add environment variables
- [ ] Deploy and verify

#### Option B: Render
- [ ] Create Render account
- [ ] Create new Web Service
- [ ] Connect GitHub repository
- [ ] Configure build/start commands
- [ ] Add environment variables
- [ ] Deploy and verify

#### Option C: AWS EC2
- [ ] Launch EC2 instance
- [ ] Configure security groups
- [ ] Install dependencies
- [ ] Clone repository
- [ ] Setup systemd service
- [ ] Configure Nginx
- [ ] Setup SSL with Let's Encrypt

**Backend Verification:**
```bash
curl https://your-backend-url.com/health
# Expected: {"status":"healthy",...}
```

### 2. Frontend Deployment (Vercel)

- [ ] Create Vercel account
- [ ] Import GitHub repository
- [ ] Set root directory to `frontend`
- [ ] Add environment variables:
  - [ ] NEXT_PUBLIC_CONVEX_URL
  - [ ] NEXT_PUBLIC_API_URL (backend URL)
  - [ ] NEXT_PUBLIC_WS_URL (backend WSS URL)
  - [ ] NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
  - [ ] CLERK_SECRET_KEY
  - [ ] CONVEX_DEPLOYMENT
- [ ] Deploy and verify

**Frontend Verification:**
```bash
curl https://your-frontend-url.vercel.app
# Should return HTML
```

### 3. Database (Convex)

- [ ] Convex project already set up
- [ ] Production deployment created
- [ ] Environment variables updated
- [ ] Tables verified in dashboard

### 4. Integration Testing

- [ ] Open frontend URL in browser
- [ ] Navigate to /verify-glass
- [ ] Test camera access
- [ ] Complete verification flow
- [ ] Verify token issuance
- [ ] Check WebSocket connection
- [ ] Test error scenarios

## Post-Deployment

### Monitoring Setup
- [ ] Set up uptime monitoring (UptimeRobot)
- [ ] Configure error tracking (Sentry)
- [ ] Enable Vercel Analytics
- [ ] Set up log aggregation
- [ ] Configure alerts

### Performance
- [ ] Run Lighthouse audit
- [ ] Check page load times
- [ ] Verify WebSocket latency
- [ ] Test with multiple users
- [ ] Monitor resource usage

### Documentation
- [ ] Update README with production URLs
- [ ] Document deployment process
- [ ] Create runbook for common issues
- [ ] Update API documentation

### Security Audit
- [ ] Run security scan
- [ ] Check for exposed secrets
- [ ] Verify HTTPS/WSS
- [ ] Test CORS configuration
- [ ] Review access logs

## Environment Variables Checklist

### Frontend Production (.env.local)
```bash
✓ NEXT_PUBLIC_CONVEX_URL=https://keen-lion-797.convex.cloud
✓ CONVEX_DEPLOYMENT=prod:keen-lion-797
⚠ NEXT_PUBLIC_API_URL=https://your-backend-url.com  # UPDATE
⚠ NEXT_PUBLIC_WS_URL=wss://your-backend-url.com     # UPDATE
✓ NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_...
✓ CLERK_SECRET_KEY=sk_...
```

### Backend Production (.env)
```bash
✓ CONVEX_URL=https://keen-lion-797.convex.cloud
⚠ JWT_SECRET_KEY=CHANGE-THIS-IN-PRODUCTION  # UPDATE
✓ JWT_ALGORITHM=RS256
✓ JWT_EXPIRY_MINUTES=15
✓ SESSION_TIMEOUT_SECONDS=120
✓ MAX_FAILED_ATTEMPTS=3
⚠ CORS_ORIGINS=https://your-frontend-url.vercel.app  # UPDATE
✓ USE_WSS=true
⚠ WEBSOCKET_HOST=your-backend-url.com  # UPDATE
```

## Quick Deployment Commands

### Deploy Backend to Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project
railway link

# Deploy
railway up
```

### Deploy Frontend to Vercel
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd frontend
vercel --prod
```

## Rollback Plan

If deployment fails:

1. **Frontend:** Revert to previous deployment in Vercel dashboard
2. **Backend:** Redeploy previous version or rollback in Railway/Render
3. **Database:** Convex has automatic backups
4. **DNS:** Update DNS records if needed

## Success Criteria

- [ ] Frontend loads without errors
- [ ] Backend health check passes
- [ ] WebSocket connection establishes
- [ ] Camera access works
- [ ] Verification flow completes
- [ ] Token issuance successful
- [ ] All tests passing
- [ ] No console errors
- [ ] Performance metrics acceptable
- [ ] Security scan passes

## Support Contacts

- **GitHub Issues:** https://github.com/ArrinPaul/Proof-of-life/issues
- **Convex Support:** https://convex.dev/support
- **Vercel Support:** https://vercel.com/support

## Timeline

- **Preparation:** 1-2 hours
- **Backend Deployment:** 30-60 minutes
- **Frontend Deployment:** 15-30 minutes
- **Testing:** 30-60 minutes
- **Monitoring Setup:** 30 minutes
- **Total:** 3-5 hours

## Notes

- Keep this checklist updated as you deploy
- Document any issues encountered
- Save production URLs and credentials securely
- Schedule regular security audits
- Plan for scaling if needed

---

**Deployment Date:** _____________

**Deployed By:** _____________

**Production URLs:**
- Frontend: _____________
- Backend: _____________
- API Docs: _____________

**Status:** ⬜ Not Started | ⬜ In Progress | ⬜ Complete
