#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Compile Objective-C Metal bridge
echo "smith: compiling native bridge..."
clang -O2 -shared -fobjc-arc \
  -framework Metal -framework Foundation \
  -o native/libsmith.dylib native/gpu_bridge.m

# 2. Compile Metal shaders
echo "smith: compiling shaders..."
for f in shaders/*.metal; do
  xcrun -sdk macosx metal -O2 -c "$f" -o "${f%.metal}.air"
done
xcrun -sdk macosx metallib -o shaders/smith.metallib shaders/*.air
rm -f shaders/*.air

echo "smith: build complete"
