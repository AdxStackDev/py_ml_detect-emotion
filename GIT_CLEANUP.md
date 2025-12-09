# Git Repository Cleanup Guide

## Problem
Files were committed to Git before `.gitignore` was created. Now we need to remove them from Git while keeping them locally.

## Solution Steps

### Step 1: Verify Current Git Status
First, let's see what's currently tracked:
```bash
git status
```

### Step 2: Remove Cached Files from Git (Safe Method)
This removes files from Git tracking but keeps them on your local disk:

```bash
# Remove all files from Git's index (staging area)
git rm -r --cached .

# Re-add everything (now respecting .gitignore)
git add .

# Check what will be committed
git status
```

### Step 3: Commit the Changes
```bash
git commit -m "Clean up repository: Remove ignored files and apply .gitignore"
```

### Step 4: Push to Remote (if applicable)
```bash
git push origin main
# or
git push origin master
```

---

## Files That Will Be Removed from Git

Based on your `.gitignore`, these will be removed from Git tracking:

### Python Artifacts
- `__pycache__/` folders
- `*.pyc`, `*.pyo` files
- `.eggs/`, `*.egg-info/` folders

### Virtual Environment
- `.venv/`, `venv/`, `env/` folders

### Flask Generated Files
- `uploads/` folder (user-uploaded images)
- `results.json` (processing history)
- `instance/` folder

### IDE Files
- `.vscode/` folder
- `.idea/` folder
- `*.swp`, `*.swo` files

### Analysis Results
- `analysis_results/` folder
- Generated plots

### OS Files
- `.DS_Store`
- `Thumbs.db`

### Logs
- `*.log` files

---

## Files That Will Be KEPT in Git

- ✅ All source code (`.py`, `.js`, `.css`, `.html`)
- ✅ `sad_happy_angry.pth` (trained model)
- ✅ Sample images (`boy.png`, `crying.png`, `person.png`)
- ✅ Documentation (`.md` files)
- ✅ `requirements.txt`
- ✅ `templates/` and `static/` folders
- ✅ `.gitignore` itself

---

## Alternative: Remove Specific Files/Folders

If you want to remove specific items only:

```bash
# Remove specific folder
git rm -r --cached uploads/
git rm -r --cached __pycache__/
git rm -r --cached .venv/
git rm -r --cached .idea/
git rm -r --cached .vscode/
git rm -r --cached analysis_results/

# Remove specific file
git rm --cached results.json

# Commit
git commit -m "Remove ignored files from Git"
```

---

## Verification

After running the cleanup, verify:

```bash
# Check what's staged
git status

# See what files are tracked
git ls-files

# See what files exist locally but are ignored
git status --ignored
```

---

## Important Notes

⚠️ **WARNINGS:**
1. This only removes files from Git, NOT from your local disk
2. Make sure `.gitignore` is correct before running these commands
3. If working with a team, coordinate this cleanup
4. Consider backing up your repo before major cleanup

✅ **SAFE:**
- Your local files remain untouched
- You can always undo with `git reset`
- `.gitignore` prevents re-adding ignored files

---

## Troubleshooting

### If files still appear after cleanup:
```bash
# Clear Git cache completely
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules"
```

### If you accidentally removed important files:
```bash
# Undo the last commit (keeps changes)
git reset --soft HEAD~1

# Or restore specific file
git restore --staged <file>
```

### If .gitignore isn't working:
```bash
# Make sure .gitignore is committed
git add .gitignore
git commit -m "Add .gitignore"

# Then clean up
git rm -r --cached .
git add .
```

---

## Quick One-Liner Solution

For a quick cleanup (recommended):

```bash
git rm -r --cached . && git add . && git status
```

Then review the changes and commit:

```bash
git commit -m "Clean up: Remove ignored files and apply .gitignore"
git push
```

---

## What Happens Next?

After cleanup:
1. Ignored files are removed from Git history (in new commits)
2. They remain on your local disk
3. Future commits won't include them
4. Team members won't download them when they pull

---

## Repository Size

If your repo size is still large after cleanup, you may need to clean Git history:

```bash
# WARNING: This rewrites history - use with caution
git filter-branch --tree-filter 'rm -rf uploads' HEAD
git push --force
```

**Note:** Only use `filter-branch` if absolutely necessary and coordinate with your team!
