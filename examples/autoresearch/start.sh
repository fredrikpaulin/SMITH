#!/bin/bash
# Start the autoresearch experiment loop.
# Prepares data if needed, then launches Claude Code.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SMITH_ROOT="$(cd "$DIR/../.." && pwd)"

cd "$DIR"

# Check bun
if ! command -v bun &>/dev/null; then
  echo "Bun not found. Install: curl -fsSL https://bun.sh/install | bash"
  exit 1
fi

# Check smith native lib
if [ ! -f "$SMITH_ROOT/native/libsmith.dylib" ]; then
  echo "Smith not built. Run: cd $SMITH_ROOT && bash build.sh"
  exit 1
fi

# Prepare data if missing or incomplete
if [ ! -f "$DIR/data/train.bin" ] || [ ! -f "$DIR/data/val.bin" ] || [ ! -f "$DIR/data/tokenizer.json" ]; then
  echo "Preparing data (downloading texts, training tokenizer)..."
  rm -rf "$DIR/data"
  bun "$DIR/prepare.js" --out "$DIR/data" --vocab 4096
  echo ""
fi

# Check claude is installed
if ! command -v claude &>/dev/null; then
  echo "Claude Code not found. Install: npm install -g @anthropic-ai/claude-code"
  exit 1
fi

echo "Starting autoresearch..."
echo "  Data:  $DIR/data/"
echo "  Model: model.js (modifiable)"
echo "  Train: train.js (modifiable)"
echo ""

# --allowedTools lets the agent run without permission prompts.
# Scoped to:
#   Bash(bun*)     — run training and research.js
#   Bash(git*)     — commit, revert, branch
#   Bash(grep*)    — parse logs
#   Bash(cat*)     — read files
#   Edit, Read     — modify model.js / train.js
claude "start autoresearch" --allowedTools "Bash(bun*),Bash(git*),Bash(grep*),Bash(cat*),Bash(tail*),Bash(head*),Edit,Read"
