#!/bin/bash
# Reset autoresearch to a clean state.
# Restores train.js and model.js from git, clears results, deletes the experiment branch.
# Data is preserved (expensive to regenerate).

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SMITH_ROOT="$(cd "$DIR/../.." && pwd)"

cd "$SMITH_ROOT"

# --- Detect current branch ---
BRANCH=$(git branch --show-current 2>/dev/null || echo "")

echo "Autoresearch reset"
echo "  Directory: $DIR"
echo "  Branch:    ${BRANCH:-detached HEAD}"
echo ""

# --- Restore modifiable files from git ---
echo "Restoring train.js and model.js from git..."
git checkout HEAD -- examples/autoresearch/train.js examples/autoresearch/model.js 2>/dev/null || true

# --- Clear results ---
if [ -d "$DIR/results" ]; then
  echo "Clearing results/"
  rm -rf "$DIR/results"
  mkdir -p "$DIR/results"
fi

# --- Clear Claude Code session state ---
if [ -d "$DIR/.claude" ]; then
  echo "Clearing .claude/"
  rm -rf "$DIR/.claude"
fi

# --- Switch back to main and optionally delete experiment branch ---
if [[ "$BRANCH" == autoresearch/* ]]; then
  echo ""
  echo "On experiment branch: $BRANCH"
  # Stash any uncommitted changes so checkout doesn't fail
  git stash -q 2>/dev/null || true
  git checkout main 2>/dev/null || git checkout master 2>/dev/null || {
    echo "Warning: could not switch to main/master"
    exit 0
  }
  echo -n "Delete branch $BRANCH? [y/N] "
  read -r answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    git branch -D "$BRANCH"
    echo "Deleted $BRANCH"
  fi
fi

echo ""
echo "Reset complete. Data preserved in data/."
echo "Run ./start.sh to begin a new experiment loop."
