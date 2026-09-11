#!/usr/bin/env bash
# Downloads the OpenCV YuNet face detector model used by view_meld_clips.py's
# --faces overlay. ~230KB, from the official opencv/opencv_zoo GitHub repo.
#
# (We originally used MediaPipe's BlazeFace here, but its macOS Tasks-API build
# crashes with a GPU graph-service error unrelated to our code -- see
# https://github.com/google-ai-edge/mediapipe/issues/4594. YuNet runs through
# OpenCV's own DNN backend instead and has no such dependency.)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT/models"
DEST="$DEST_DIR/face_detection_yunet_2023mar.onnx"
URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

mkdir -p "$DEST_DIR"
echo "Downloading YuNet face detector model to $DEST"
curl -sL -o "$DEST" "$URL"
echo "Done ($(du -h "$DEST" | cut -f1))."
