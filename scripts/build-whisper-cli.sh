#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/.local/whisper.cpp"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. Install it with Homebrew (brew install cmake), then rerun this script." >&2
  exit 1
fi

if [ ! -d "$SOURCE_DIR/.git" ]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$SOURCE_DIR"
fi

cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 -DGGML_METAL=ON
cmake --build "$SOURCE_DIR/build" --config Release --target whisper-cli -j "$(getconf _NPROCESSORS_ONLN)"
echo "whisper-cli: $SOURCE_DIR/build/bin/whisper-cli"
