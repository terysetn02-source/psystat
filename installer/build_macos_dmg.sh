#!/usr/bin/env bash
# Create a drag-and-drop disk image from dist/PsyStat.app
#   usage:  installer/build_macos_dmg.sh <version> <AppleSilicon|Intel>
# Run from the repository root after `pyinstaller --noconfirm psystat.spec`.
set -euo pipefail

VERSION="${1:?usage: build_macos_dmg.sh <version> <AppleSilicon|Intel>}"
ARCH="${2:?usage: build_macos_dmg.sh <version> <AppleSilicon|Intel>}"
APP="dist/PsyStat.app"
DMG="dist/PsyStat-${VERSION}-macOS-${ARCH}.dmg"

[ -d "$APP" ] || { echo "error: $APP not found - run pyinstaller first" >&2; exit 1; }

# Ad-hoc signature. Apple-silicon Macs refuse to run completely unsigned code; an ad-hoc signature is free and makes the
# app launchable after the user confirms it once (see the manual, section 2.2). It is NOT a notarisation.
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cp LICENSE NOTICE.md "$STAGE/"

rm -f "$DMG"
hdiutil create -volname "PsyStat ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
echo "created $DMG"
