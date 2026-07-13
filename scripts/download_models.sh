#!/usr/bin/env bash
# scripts/download_models.sh
#
# Downloads the three custom-trained YOLO model weights from Google Drive
# into the models/ directory at the repo root.
#
# Usage:
#   bash scripts/download_models.sh
#
# The script uses `gdown` to fetch files from Google Drive.  Install it with:
#   pip install gdown
#
# If you don't want to install gdown, you can download manually:
#   Player detector:        https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view
#   Ball detector:          https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view
#   Court keypoint model:   https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view
# Save them to models/ as player_detector.pt, ball_detector_model.pt,
# and court_keypoint_detector.pt respectively.
#
# Note: These weights are hosted by the original tutorial author (Abdullah
# Tarek). If the links go stale, re-train using the notebooks in
# training_notebooks/ or fine-tune from an Ultralytics COCO checkpoint with
# scripts/finetune_model.py.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
mkdir -p "$MODELS_DIR"

if ! command -v gdown &> /dev/null; then
    echo "[error] gdown not found. Install it with:  pip install gdown"
    exit 1
fi

download_if_missing() {
    local dest="$1"
    local gdrive_id="$2"
    local label="$3"

    if [[ -f "$dest" ]]; then
        echo "[skip] $label already exists at $dest"
        return
    fi

    echo "[download] $label -> $dest"
    gdown --id "$gdrive_id" --output "$dest"
}

download_if_missing \
    "$MODELS_DIR/player_detector.pt" \
    "1KejdrcEnto2AKjdgdo1U1syr5gODp6EL" \
    "Player detector"

download_if_missing \
    "$MODELS_DIR/ball_detector_model.pt" \
    "1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0" \
    "Ball detector"

download_if_missing \
    "$MODELS_DIR/court_keypoint_detector.pt" \
    "1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy" \
    "Court keypoint detector"

echo ""
echo "[done] Models are in $MODELS_DIR"
echo "       Run the pipeline with: python main.py <video.mp4>"
