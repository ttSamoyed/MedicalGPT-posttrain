#!/usr/bin/env bash
set -euo pipefail

export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/Qwen2.5-7B-Instruct}"
export SFT_MODEL_PATH="${SFT_MODEL_PATH:-$BASE_MODEL_PATH}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$MEDICALGPT_ROOT/outputs/rm/qwen25-7b-rm}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/ppo_prompts}"
export VALID_FILE_DIR="${VALID_FILE_DIR:-$TRAIN_FILE_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/ppo/qwen25-7b-smoke}"
export MAX_STEPS="${MAX_STEPS:-100}"

cd "$MEDICALGPT_ROOT"

mkdir -p "$OUTPUT_DIR"

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
