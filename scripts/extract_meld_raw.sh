#!/usr/bin/env bash
# Extracts data/meld/raw/MELD.Raw.tar.gz. The official archive nests train/dev/test
# as their own tar(.gz) files inside the top-level one, so this also unpacks any
# nested archives it finds (harmless if there are none).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$ROOT/data/meld/raw/MELD.Raw.tar.gz"
DEST="$ROOT/data/meld/raw/extracted"

if [ ! -f "$ARCHIVE" ]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 1
fi

mkdir -p "$DEST"
echo "Extracting top-level archive to $DEST (this can take a few minutes)..."
tar xzf "$ARCHIVE" -C "$DEST"

while IFS= read -r nested; do
  echo "Extracting nested archive: $nested"
  tar xf "$nested" -C "$(dirname "$nested")"
done < <(find "$DEST" -maxdepth 3 \( -name "*.tar.gz" -o -name "*.tar" \))

echo "Done. Extracted layout:"
find "$DEST" -maxdepth 3 -type d | sort
