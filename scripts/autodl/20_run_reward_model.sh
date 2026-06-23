#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-$REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_DIR_NAME="$(basename "$BASE_MODEL")"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/$MODEL_DIR_NAME}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/reward}"
export VALID_FILE_DIR="${VALID_FILE_DIR:-$TRAIN_FILE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/rm/$MODEL_DIR_NAME-rm}"
export CACHE_DIR="${CACHE_DIR:-$MEDICALGPT_ROOT/cache}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"

cd "$MEDICALGPT_ROOT"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

if [[ "$BASE_MODEL_PATH" = /* && ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH"
  echo "Run first:"
  echo "  BASE_MODEL=$BASE_MODEL bash scripts/autodl/01_prepare_assets.sh"
  exit 2
fi

python3 training/reward_modeling.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --train_file_dir "$TRAIN_FILE_DIR" \
  --validation_file_dir "$VALID_FILE_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --use_peft True \
  --load_in_4bit True \
  --seed 42 \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_steps 5 \
  --weight_decay 0.001 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 2 \
  --max_source_length 1024 \
  --max_target_length 512 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --logging_first_step True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bf16 \
  --torch_dtype bfloat16 \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing True \
  --cache_dir "$CACHE_DIR"
