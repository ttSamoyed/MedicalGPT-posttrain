#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-$REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR_NAME="$(basename "$BASE_MODEL")"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/$MODEL_DIR_NAME}"
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-$BASE_MODEL_PATH}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$MEDICALGPT_ROOT/outputs/rm/$MODEL_DIR_NAME-rm}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/ppo_prompts}"
export VALID_FILE_DIR="${VALID_FILE_DIR:-$TRAIN_FILE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/ppo/$MODEL_DIR_NAME-smoke}"
export MAX_STEPS="${MAX_STEPS:-100}"

cd "$MEDICALGPT_ROOT"

mkdir -p "$OUTPUT_DIR"

if [[ "$BASE_MODEL_PATH" = /* && ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH"
  echo "Run first:"
  echo "  BASE_MODEL=$BASE_MODEL bash scripts/server/01_prepare_assets.sh"
  exit 2
fi

if [[ "$REWARD_MODEL_PATH" = /* && ! -d "$REWARD_MODEL_PATH" ]]; then
  echo "REWARD_MODEL_PATH does not exist: $REWARD_MODEL_PATH"
  echo "Run reward model training first or set REWARD_MODEL_PATH manually."
  exit 2
fi

python3 training/ppo_training.py \
  --sft_model_path "$SFT_MODEL_PATH" \
  --reward_model_path "$REWARD_MODEL_PATH" \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --dtype bfloat16 \
  --train_file_dir "$TRAIN_FILE_DIR" \
  --validation_file_dir "$VALID_FILE_DIR" \
  --max_source_length 1024 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --do_train \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --eval_strategy steps \
  --eval_steps 50 \
  --num_train_epochs 1 \
  --report_to tensorboard \
  --use_peft True \
  --load_in_4bit True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05
