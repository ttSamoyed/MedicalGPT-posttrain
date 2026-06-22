#!/usr/bin/env bash
set -euo pipefail

export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$MEDICALGPT_ROOT/models/base/Qwen2.5-7B-Instruct}"
export TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-$MEDICALGPT_ROOT/data/grpo}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MEDICALGPT_ROOT/outputs/grpo/qwen25-7b-smoke}"
export MAX_STEPS="${MAX_STEPS:-50}"
export TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"

cd "$MEDICALGPT_ROOT"
mkdir -p "$OUTPUT_DIR"

python3 training/grpo_training.py \
  --model_name_or_path "$BASE_MODEL_PATH" \
  --train_file_dir "$TRAIN_FILE_DIR" \
  --train_samples "$TRAIN_SAMPLES" \
  --max_steps "$MAX_STEPS" \
  --num_train_epochs 1 \
  --save_steps 25 \
  --save_strategy steps \
  --save_total_limit 2 \
  --output_dir "$OUTPUT_DIR" \
  --dtype bfloat16 \
  --bf16 True \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 5 \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --num_generations 2 \
  --gradient_accumulation_steps 4 \
  --max_completion_length 512
