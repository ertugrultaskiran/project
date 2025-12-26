# 🚀 IT Ticket Classifier - Deployment Guide

## Quick Deploy to Railway

### Prerequisites
- GitHub account
- Railway account (free tier)

### Steps:

#### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - IT Ticket Classifier with Analytics Dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

#### 2. Deploy to Railway

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will auto-detect `railway.json` and `Dockerfile.web`
6. Click "Deploy"

#### 3. Get Your Live URL

After deployment (2-3 minutes):
- Railway will provide a URL like: `https://your-app.railway.app`
- Click "Generate Domain" if not auto-generated
- Test the URL!

### Environment Variables

No environment variables needed for demo version!

### What Gets Deployed

✅ **Analytics Dashboard**
- Real-time charts (Chart.js)
- Category distribution
- Model performance
- Confidence analysis
- Interactive filters

✅ **AI Features**
- Sentiment Analysis
- Priority Detection
- Smart Routing
- SLA Prediction
- Similar Tickets Finder

✅ **Mock Data**
- 50 pre-generated tickets
- All features working
- No model files needed (lightweight!)

### Features

- 📊 **Analytics Dashboard** with 4 interactive charts
- 🧠 **AI-Powered Classification** with sentiment analysis
- 🔥 **Priority Detection** (HIGH/MEDIUM/LOW)
- 👥 **Smart Routing** to departments
- ⏱️ **SLA Prediction** based on priority
- 📋 **Similar Tickets** finder
- 📈 **Real-time Statistics**

### Performance

- **Startup time**: ~30-60 seconds
- **Memory usage**: ~200-300 MB
- **Response time**: <1 second

### Troubleshooting

**Issue**: App not starting
- Check Railway logs
- Verify Dockerfile.web exists
- Check requirements.web.txt

**Issue**: 404 errors
- Wait for full deployment (2-3 min)
- Clear browser cache
- Try incognito mode

### Demo Credentials

No authentication needed - public demo!

### Support

Created for graduation project - December 2025

