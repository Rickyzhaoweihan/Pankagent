# GitHub Setup Guide

Complete guide for uploading this project to GitHub.

## 🔒 Pre-Upload Security Checklist

### ✅ Files Already Protected (in .gitignore)

These files will NOT be uploaded:
- ✅ `config.py` - Contains API keys
- ✅ `.env` - Environment variables
- ✅ `logs/` - May contain sensitive data
- ✅ `__pycache__/` - Compiled Python
- ✅ `*.log` - Log files

### ⚠️ Verify Before Upload

```bash
# Check what will be committed
git status

# Verify config.py is NOT listed
git status | grep config.py
# Should return nothing

# If you see config.py, it means .gitignore isn't working
# Remove it from tracking:
git rm --cached config.py
```

## 📋 Step-by-Step Upload Process

### Step 1: Initialize Git Repository

```bash
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/PlannerAgent_3_11-2-2025

# Initialize git (if not already done)
git init

# Check current status
git status
```

### Step 2: Add Files

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what will be committed
git status

# Check for sensitive files
git status | grep -E "(config.py|.env|*.key)"
# Should return nothing
```

### Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: PanKgraph AI Assistant with multi-agent system

- Multi-agent architecture (PlannerAgent, PankBaseAgent, GLKBAgent, TemplateToolAgent)
- FastAPI server with configurable port
- vLLM integration for local model inference
- Text-to-Cypher translation with validation
- AWS deployment guides
- Comprehensive documentation"
```

### Step 4: Create GitHub Repository

**Option A: Via GitHub Website (Recommended)**

1. Go to https://github.com/new
2. **Repository name:** `PanKgraph-AI-Assistant` (or your choice)
3. **Description:** "Multi-agent system for querying biomedical knowledge graphs using natural language"
4. **Visibility:** 
   - Choose **Private** if it contains proprietary code/data
   - Choose **Public** if you want to share it
5. **DO NOT** check "Initialize with README" (you already have one)
6. Click **"Create repository"**

**Option B: Via GitHub CLI**

```bash
# Install GitHub CLI if needed
# Mac: brew install gh
# Linux: sudo apt install gh

# Login
gh auth login

# Create private repository
gh repo create PanKgraph-AI-Assistant --private --source=. --remote=origin

# Or create public repository
gh repo create PanKgraph-AI-Assistant --public --source=. --remote=origin
```

### Step 5: Connect to GitHub

After creating the repo on GitHub, you'll see instructions. Use these commands:

```bash
# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/PanKgraph-AI-Assistant.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/PanKgraph-AI-Assistant.git

# Verify remote
git remote -v
```

### Step 6: Push to GitHub

```bash
# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## 🎉 Success!

Your repository should now be on GitHub at:
`https://github.com/YOUR_USERNAME/PanKgraph-AI-Assistant`

## 🔄 Future Updates

After the initial upload, use these commands for updates:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add feature: description of changes"

# Push to GitHub
git push
```

## 🚨 If You Accidentally Committed Secrets

If you accidentally pushed `config.py` or other sensitive files:

### Step 1: Remove from Git History

```bash
# Remove file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config.py" \
  --prune-empty --tag-name-filter cat -- --all

# Force push to GitHub
git push origin --force --all
```

### Step 2: Rotate All Exposed Secrets

**IMMEDIATELY** change any exposed API keys:
1. Go to Anthropic Console: https://console.anthropic.com/
2. Delete the exposed API key
3. Generate a new one
4. Update your local `config.py`

## 📝 Recommended GitHub Settings

### 1. Add Repository Topics

On GitHub, add topics to help others find your project:
- `biomedical-informatics`
- `knowledge-graph`
- `natural-language-processing`
- `multi-agent-system`
- `fastapi`
- `vllm`
- `text-to-cypher`

### 2. Set Up Branch Protection (Optional)

For team projects:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable "Require pull request reviews"

### 3. Add Collaborators (Optional)

Settings → Collaborators → Add people

## 🔍 Verify Upload

After pushing, verify on GitHub:

1. ✅ README.md displays correctly
2. ✅ `.gitignore` is present
3. ✅ `.env.example` is present
4. ✅ `config.py` is NOT present
5. ✅ No API keys visible in any file
6. ✅ Documentation files are readable

## 📚 Next Steps

1. Add a LICENSE file (if not present)
2. Set up GitHub Actions for CI/CD (optional)
3. Create releases/tags for versions
4. Add GitHub Issues templates
5. Set up GitHub Wiki for extended docs

## 🆘 Troubleshooting

### "Permission denied (publickey)"

Use HTTPS instead of SSH:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/PanKgraph-AI-Assistant.git
```

### "Repository not found"

Verify the repository name and your GitHub username:
```bash
git remote -v
```

### Large files rejected

GitHub has a 100MB file size limit. If you have large model files:
```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
```

Or add them to `.gitignore` and document where to download them.

## 📞 Support

For GitHub-specific issues:
- GitHub Docs: https://docs.github.com/
- GitHub Support: https://support.github.com/

For project-specific issues:
- Create an issue on your GitHub repository
- Check the documentation in the repo

