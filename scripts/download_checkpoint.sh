#!/usr/bin/env bash
# Fetches the submitted model from the repo's GitHub release into the path the
# demos and scripts expect (results/stage2_fusion_faces_only/seed1/), with its
# results.json and test_predictions.jsonl, and verifies the checkpoint's sha256.
# Idempotent: skips files that already exist with the right checksum/size.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/results/stage2_fusion_faces_only/seed1"
BASE="https://github.com/nicholasg1997/MLChallenge/releases/download/v1.0"
SHA256="9dfed4ec617a2b076a32394859a214c4cc04cec7c4c78addc13bae356189cf39"

mkdir -p "$DEST"
for f in results.json test_predictions.jsonl log.jsonl; do
  [ -s "$DEST/$f" ] || curl -fL --progress-bar -o "$DEST/$f" "$BASE/$f"
done

if [ -s "$DEST/best.pt" ] && echo "$SHA256  $DEST/best.pt" | shasum -a 256 -c --status; then
  echo "best.pt already present and verified"
else
  echo "downloading best.pt (886 MB) ..."
  curl -fL --progress-bar -o "$DEST/best.pt" "$BASE/best.pt"
  echo "$SHA256  $DEST/best.pt" | shasum -a 256 -c
fi
echo "-> $DEST"
