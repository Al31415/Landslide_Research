# Deployment Setup Guide

This guide will help you set up automated deployment for the Landslide Research application.

## Prerequisites

1. **Fly.io Account**: Make sure you have a Fly.io account and the app `stability-predictor` is already created
2. **GitHub Repository**: This repository should be on GitHub with the `deploy-app` branch

## Step-by-Step Setup

### 1. Get Fly.io API Token

If you have flyctl installed locally:
```bash
fly auth token
```

If you don't have flyctl installed, you can get the token from the Fly.io dashboard:
1. Go to https://fly.io/dashboard
2. Click on your profile (top right)
3. Go to "Access Tokens"
4. Create a new token or copy an existing one

### 2. Add GitHub Secret

1. Go to your GitHub repository
2. Click on "Settings" tab
3. In the left sidebar, click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `FLY_API_TOKEN`
6. Value: Paste your Fly.io API token
7. Click "Add secret"

### 3. Verify Fly.io App Configuration

Make sure your `fly.toml` file has the correct app name:
```toml
app = 'stability-predictor'
```

### 4. Deploy

#### Option A: Push to deploy-app branch
```bash
git checkout deploy-app
git add .
git commit -m "Deploy latest changes"
git push origin deploy-app
```

#### Option B: Manual trigger from GitHub
1. Go to the "Actions" tab in your GitHub repository
2. Click on "Deploy to Fly.io" workflow
3. Click "Run workflow"
4. Select the branch and click "Run workflow"

### 5. Monitor Deployment

1. Go to the "Actions" tab to see the deployment progress
2. Once complete, check your app at: https://stability-predictor.fly.dev
3. You can also check the status with: `fly status --app stability-predictor`

## Troubleshooting

### Common Issues

1. **FLY_API_TOKEN not set**: Make sure the secret is properly added to GitHub
2. **App not found**: Verify the app name in `fly.toml` matches your Fly.io app
3. **Build failures**: Check the GitHub Actions logs for specific error messages
4. **Health check failures**: The app may be starting slowly; check Fly.io logs

### Checking Logs

From GitHub Actions:
- Go to the failed workflow run
- Click on the failed job
- Expand the failed step to see error details

From Fly.io:
```bash
fly logs --app stability-predictor
```

## Workflow Features

The deployment workflow includes:

- ✅ Automatic deployment on `deploy-app` branch pushes
- ✅ Manual deployment trigger
- ✅ Remote Docker builds (no local Docker required)
- ✅ Deployment status checks
- ✅ Health checks after deployment
- ✅ Detailed logging and error reporting

## Environment Configuration

The app is configured with:
- **Region**: Chicago (ord)
- **Resources**: 4 CPUs, 8GB RAM
- **Auto-scaling**: Enabled
- **HTTPS**: Enforced
- **Health checks**: Built-in Streamlit health endpoint

## Next Steps

After successful deployment:
1. Test the application at https://stability-predictor.fly.dev
2. Monitor performance and logs
3. Set up monitoring/alerting if needed
4. Consider setting up staging environment for testing 