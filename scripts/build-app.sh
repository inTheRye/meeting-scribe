#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/dist/Meeting Scribe.app"
CONTENTS_DIR="$APP_DIR/Contents"
CLI_DIR="$ROOT_DIR/.local/whisper.cpp/build/bin"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
SIGN_IDENTITY="${MEETING_SCRIBE_CODESIGN_IDENTITY:--}"

if [ ! -x "$CLI_DIR/whisper-cli" ]; then
  echo "whisper-cli is missing. Run ./scripts/build-whisper-cli.sh first." >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/.build"
swiftc -parse-as-library -O \
  -sdk "$SDK_PATH" \
  -target arm64-apple-macosx15.0 \
  -module-cache-path "$ROOT_DIR/.build/module-cache" \
  -framework AppKit \
  -framework AVFoundation \
  -framework AudioToolbox \
  -framework CoreGraphics \
  -framework CoreMedia \
  -framework ScreenCaptureKit \
  -framework SwiftUI \
  -framework UniformTypeIdentifiers \
  -o "$ROOT_DIR/.build/MeetingScribe" \
  "$ROOT_DIR/Sources/MeetingScribe/BatchTranscriber.swift" \
  "$ROOT_DIR/Sources/MeetingScribe/BatchTranscriptQualityGuard.swift" \
  "$ROOT_DIR/Sources/MeetingScribe/SessionAudioArchive.swift" \
  "$ROOT_DIR/Sources/MeetingScribe/main.swift"
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS_DIR/MacOS" "$CONTENTS_DIR/Resources" "$CONTENTS_DIR/Frameworks"
cp "$ROOT_DIR/.build/MeetingScribe" "$CONTENTS_DIR/MacOS/MeetingScribe"
cp "$ROOT_DIR/Info.plist" "$CONTENTS_DIR/Info.plist"
cp "$ROOT_DIR/Assets/MeetingScribe.icns" "$CONTENTS_DIR/Resources/MeetingScribe.icns"

FRAMEWORKS_DIR="$CONTENTS_DIR/Frameworks"
cp "$CLI_DIR/whisper-cli" "$FRAMEWORKS_DIR/whisper-cli"
for library in libwhisper.1.dylib libggml.0.dylib libggml-base.0.dylib libggml-cpu.0.dylib libggml-blas.0.dylib libggml-metal.0.dylib; do
  cp -L "$CLI_DIR/$library" "$FRAMEWORKS_DIR/$library"
done
for binary in "$FRAMEWORKS_DIR/whisper-cli" "$FRAMEWORKS_DIR"/*.dylib; do
  if otool -l "$binary" | grep -Fq "path $CLI_DIR "; then
    install_name_tool -delete_rpath "$CLI_DIR" "$binary"
  fi
  if [ "$(basename "$binary")" = "whisper-cli" ]; then
    install_name_tool -add_rpath '@executable_path/../Frameworks' "$binary"
  else
    install_name_tool -add_rpath '@loader_path' "$binary"
  fi
done
cp "$ROOT_DIR/.local/whisper.cpp/LICENSE" "$CONTENTS_DIR/Resources/whisper.cpp-LICENSE"
printf 'APPL????' > "$CONTENTS_DIR/PkgInfo"
for library in "$FRAMEWORKS_DIR"/*.dylib; do
  codesign --force --sign "$SIGN_IDENTITY" "$library"
done
codesign --force --sign "$SIGN_IDENTITY" "$FRAMEWORKS_DIR/whisper-cli"
codesign --force --deep --sign "$SIGN_IDENTITY" "$APP_DIR"
printf 'Built: %s\n' "$APP_DIR"
