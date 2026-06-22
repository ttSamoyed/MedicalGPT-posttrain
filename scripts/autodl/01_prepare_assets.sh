#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_home}"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export DOWNLOAD_MEDICAL_DATASET="${DOWNLOAD_MEDICAL_DATASET:-0}"

mkdir -p "$HF_HOME" "$MEDICALGPT_ROOT/data" "$MEDICALGPT_ROOT/models/base"

cd "$MEDICALGPT_ROOT"

echo "Install project dependencies"
pip install -U pip
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]" modelscope tensorboard

if [ "$DOWNLOAD_MEDICAL_DATASET" = "1" ]; then
  echo "Download MedicalGPT full dataset"
  huggingface-cli download \
    --repo-type dataset shibing624/medical \
    --local-dir data/medical
else
  echo "Skip full shibing624/medical dataset download."
  echo "Set DOWNLOAD_MEDICAL_DATASET=1 when you are ready for the real medical reward data."
fi

echo "Download base model: $BASE_MODEL"
MODEL_DIR_NAME="$(basename "$BASE_MODEL")"
huggingface-cli download \
  "$BASE_MODEL" \
  --local-dir "models/base/$MODEL_DIR_NAME"

echo "Assets prepared."
echo "BASE_MODEL_PATH=$MEDICALGPT_ROOT/models/base/$MODEL_DIR_NAME"
