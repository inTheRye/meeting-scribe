#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$ROOT_DIR/dist/Meeting Scribe.app"
CONTENTS_DIR="$APP_DIR/Contents"
SDK_PATH="$(xcrun --sdk macosx --show-sdk-path)"
SIGN_IDENTITY="${MEETING_SCRIBE_CODESIGN_IDENTITY:--}"

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
  "$ROOT_DIR/Sources/MeetingScribe/SessionAudioArchive.swift" \
  "$ROOT_DIR/Sources/MeetingScribe/main.swift"
rm -rf "$APP_DIR"
mkdir -p "$CONTENTS_DIR/MacOS" "$CONTENTS_DIR/Resources"
cp "$ROOT_DIR/.build/MeetingScribe" "$CONTENTS_DIR/MacOS/MeetingScribe"
cp "$ROOT_DIR/Info.plist" "$CONTENTS_DIR/Info.plist"
printf 'APPL????' > "$CONTENTS_DIR/PkgInfo"
codesign --force --deep --sign "$SIGN_IDENTITY" "$APP_DIR"
printf 'Built: %s\n' "$APP_DIR"
