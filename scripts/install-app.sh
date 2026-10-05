#!/bin/bash
set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_APP="$ROOT_DIR/dist/Meeting Scribe.app"
INSTALL_DIR="${1:-/Applications}"
TARGET_APP="$INSTALL_DIR/Meeting Scribe.app"
EXPECTED_BUNDLE_ID="jp.local.meetingscribe"
STAGING_DIR=""
BACKUP_APP=""
LOCK_DIR=""
LOCK_ACQUIRED=0
PRESERVE_STAGE=0

fail() {
  echo "$1" >&2
  exit 1
}

validate_app() {
  local app_path="$1"
  local info_plist="$app_path/Contents/Info.plist"
  local bundle_id executable icon_file

  [ ! -L "$app_path" ] || return 1
  [ -d "$app_path" ] && [ -f "$info_plist" ] || return 1
  bundle_id="$(plutil -extract CFBundleIdentifier raw "$info_plist" 2>/dev/null)" || return 1
  [ "$bundle_id" = "$EXPECTED_BUNDLE_ID" ] || return 1
  executable="$(plutil -extract CFBundleExecutable raw "$info_plist" 2>/dev/null)" || return 1
  icon_file="$(plutil -extract CFBundleIconFile raw "$info_plist" 2>/dev/null)" || return 1
  [ -x "$app_path/Contents/MacOS/$executable" ] || return 1
  [ -f "$app_path/Contents/Resources/$icon_file.icns" ] || return 1
  [ -x "$app_path/Contents/Frameworks/whisper-cli" ] || return 1
  codesign --verify --deep --strict "$app_path" >/dev/null 2>&1 || return 1
}

check_existing_target() {
  if [ -L "$TARGET_APP" ]; then
    fail "Refusing to replace an app symlink: $TARGET_APP"
  fi
  if [ -e "$TARGET_APP" ] && ! validate_app "$TARGET_APP"; then
    fail "Refusing to replace an item that is not a valid Meeting Scribe app: $TARGET_APP"
  fi
}

cleanup() {
  if [ -n "$BACKUP_APP" ] && [ -e "$BACKUP_APP" ]; then
    if [ ! -e "$TARGET_APP" ]; then
      if mv "$BACKUP_APP" "$TARGET_APP"; then
        BACKUP_APP=""
      else
        PRESERVE_STAGE=1
        echo "Could not restore the previous app. It is preserved at: $BACKUP_APP" >&2
      fi
    else
      PRESERVE_STAGE=1
      echo "Previous app is preserved at: $BACKUP_APP" >&2
    fi
  fi

  if [ -n "$STAGING_DIR" ] && [ -d "$STAGING_DIR" ]; then
    if [ "$PRESERVE_STAGE" -eq 0 ]; then
      rm -rf "$STAGING_DIR"
    else
      echo "Temporary install files are preserved at: $STAGING_DIR" >&2
    fi
  fi
  if [ "$LOCK_ACQUIRED" -eq 1 ] && [ -n "$LOCK_DIR" ] && [ -d "$LOCK_DIR" ]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -L "$SOURCE_APP" ] || ! validate_app "$SOURCE_APP"; then
  fail "Built app is missing or invalid: $SOURCE_APP. Run ./scripts/build-app.sh first."
fi

if ! mkdir -p "$INSTALL_DIR"; then
  fail "Could not create install directory: $INSTALL_DIR. Try a writable directory such as \"\$HOME/Applications\"."
fi
if [ ! -w "$INSTALL_DIR" ]; then
  fail "Cannot write to install directory: $INSTALL_DIR. Try ./scripts/install-app.sh \"\$HOME/Applications\"."
fi

LOCK_DIR="$INSTALL_DIR/.meeting-scribe-install.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  fail "Another install may be running, or a stale lock exists at $LOCK_DIR. Remove that directory only after confirming no install is active."
fi
LOCK_ACQUIRED=1

if pgrep -x "MeetingScribe" >/dev/null 2>&1; then
  fail "Meeting Scribe is running. Quit it before installing or updating."
fi
check_existing_target

STAGING_DIR="$(mktemp -d "$INSTALL_DIR/.meeting-scribe-install.XXXXXX")"
STAGED_APP="$STAGING_DIR/Meeting Scribe.app"
ditto "$SOURCE_APP" "$STAGED_APP"
if ! validate_app "$STAGED_APP"; then
  fail "The staged app did not pass bundle and signature validation: $STAGED_APP"
fi

if pgrep -x "MeetingScribe" >/dev/null 2>&1; then
  fail "Meeting Scribe started during installation. Quit it and run the installer again."
fi
check_existing_target
if [ -e "$TARGET_APP" ]; then
  BACKUP_APP="$STAGING_DIR/Previous Meeting Scribe.app"
  mv "$TARGET_APP" "$BACKUP_APP"
fi

if ! mv "$STAGED_APP" "$TARGET_APP"; then
  fail "Could not move the verified app into place: $TARGET_APP"
fi

if [ -n "$BACKUP_APP" ] && [ -e "$BACKUP_APP" ]; then
  rm -rf "$BACKUP_APP"
fi
BACKUP_APP=""
rm -rf "$STAGING_DIR"
STAGING_DIR=""
rmdir "$LOCK_DIR"
LOCK_DIR=""
LOCK_ACQUIRED=0
trap - EXIT INT TERM HUP

echo "Installed: $TARGET_APP"
