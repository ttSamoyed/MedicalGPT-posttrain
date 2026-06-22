#!/usr/bin/env bash
set -euo pipefail

export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"

cd "$MEDICALGPT_ROOT"

python experiments/medical_posttrain/build_posttrain_data.py \
  --input data/sft/medical_sft_1K_format.jsonl \
  --output data/grpo/medical_sft_grpo_smoke.jsonl \
  --task grpo \
  --limit 500

mkdir -p data/ppo_prompts
python experiments/medical_posttrain/build_posttrain_data.py \
  --input data/sft/medical_sft_1K_format.jsonl \
  --output data/ppo_prompts/medical_sft_prompts.jsonl \
  --task ppo_prompts \
  --limit 500

echo "Smoke data prepared."

