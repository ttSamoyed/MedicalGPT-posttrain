#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/Users/kristianzeng/Documents/MedicalGPT-src}"
DEST="${2:-/Users/kristianzeng/Documents/MedicalGPT}"

if [ ! -d "$SRC" ]; then
  echo "Source directory not found: $SRC"
  echo "Clone source first:"
  echo "  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none https://github.com/shibing624/MedicalGPT.git $SRC"
  exit 2
fi

mkdir -p "$DEST"

rsync -av \
  --exclude '.git' \
  --exclude 'models' \
  --exclude 'outputs' \
  --exclude 'data' \
  --exclude 'runs' \
  --exclude 'wandb' \
  --exclude 'checkpoints' \
  --exclude 'README.md' \
  "$SRC"/ "$DEST"/

echo "Synced MedicalGPT source from $SRC to $DEST"
echo "Kept local README and excluded data/model/output directories."

