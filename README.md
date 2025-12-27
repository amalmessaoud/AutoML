# 1. Activate env (once per terminal session)

source .venv/bin/activate # or Windows equivalent

# 2. Code in your editor...

# 3. Before committing:

ruff format . # makes code pretty
ruff check . # shows remaining issues

# Fix any errors it points out

# 4. Test your code manually or with pytest

# 5. Git commit & push

git add .
git commit -m "Add Analyzer skeleton"
git push

# 1. Make sure you're up to date

git pull origin main

# 2. Create and switch to a new branch (example: start with Analyzer)

git checkout -b feature/analyzer-agent

# 3. Do your work, commit often

git add .
git commit -m "Add skeleton for Analyzer Agent"

# 4. Push the branch to GitHub

git push origin feature/analyzer-agent

# 5. Go to GitHub → it will suggest "Create Pull Request"

# → Open PR, add description, let CI run (Ruff + tests)

# → Merge into main when ready

# 6. After merge, switch back to main and update

git checkout main
git pull origin main
