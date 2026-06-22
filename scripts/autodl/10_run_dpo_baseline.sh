#!/usr/bin/env bash
set -euo pipefail

export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/Qwen2.5-7B-Instruct}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/reward}"
export VALID_FILE_DIR="${VALID_FILE_DIR:-$TRAIN_FILE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/dpo/qwen25-7b-baseline}"
export CACHE_DIR="${CACHE_DIR:-$MEDICALGPT_ROOT/cache}"
export MAX_STEPS="${MAX_STEPS:-100}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-50}"

cd "$MEDICALGPT_ROOT"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

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
