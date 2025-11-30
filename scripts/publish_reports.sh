#!/usr/bin/env bash
# Copy report files from outputs/* to reports/ and commit them.
# Usage: ./scripts/publish_reports.sh --src outputs/bert --branch main

set -euo pipefail
SRC_DIR="${1:-outputs/bert}"
BRANCH="${2:-main}"

if [ ! -d "$SRC_DIR" ]; then
  echo "Source directory '$SRC_DIR' not found. Exiting."
  exit 1
fi

mkdir -p reports
cp -v "$SRC_DIR"/report.md reports/ 2>/dev/null || true
cp -v "$SRC_DIR"/eval_report.md reports/ 2>/dev/null || true
cp -v "$SRC_DIR"/*results*.json reports/ 2>/dev/null || true

# Add a README in reports if not present
if [ ! -f reports/README.md ]; then
  cat > reports/README.md <<'MD'
This folder contains training/evaluation reports (Markdown and JSON) for NER_Project.

Do NOT commit model weights or the entire `outputs/` directory. Large model files should be stored on the Hugging Face Hub or other artifact storage.
MD
fi

# Git add/commit/push (requires local git config and network)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add reports/ .gitignore
  if git diff --staged --quiet; then
    echo "No changes to commit."
  else
    git commit -m "Add/Update reports from $SRC_DIR"
    echo "Pushing to origin/$BRANCH..."
    git push origin "$BRANCH"
  fi
else
  echo "Not a git repo. Reports copied to ./reports but not committed."
fi

echo "Done. Reports are in ./reports/"
