#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-$REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR_NAME="$(basename "$BASE_MODEL")"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/$MODEL_DIR_NAME}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/reward}"
export VALID_FILE_DIR="${VALID_FILE_DIR:-$TRAIN_FILE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/dpo/$MODEL_DIR_NAME-baseline}"
export CACHE_DIR="${CACHE_DIR:-$MEDICALGPT_ROOT/cache}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"

cd "$MEDICALGPT_ROOT"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

if [[ "$BASE_MODEL_PATH" = /* && ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH"
  echo "Run first:"
  echo "  BASE_MODEL=$BASE_MODEL bash scripts/server/01_prepare_assets.sh"
  exit 2
fi

python3 training/dpo_training.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --train_file_dir "$TRAIN_FILE_DIR" \
  --validation_file_dir "$VALID_FILE_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --max_steps "$MAX_STEPS" \
  --eval_steps 20 \
  --save_steps 50 \
  --logging_steps 5 \
  --learning_rate 5e-6 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --output_dir "$OUTPUT_DIR" \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --cache_dir "$CACHE_DIR"
