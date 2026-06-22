#!/usr/bin/env bash
set -euo pipefail

export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-/root/autodl-tmp/medical/MedicalGPT}"
INPUT="${1:-$MEDICALGPT_ROOT/data/medical/reward/medical_reward.jsonl}"
OUTPUT="${2:-$MEDICALGPT_ROOT/reports/data_audit_medical_reward.md}"

cd "$MEDICALGPT_ROOT"
python experiments/medical_posttrain/data_audit.py \
  --input "$INPUT" \
  --output "$OUTPUT"

echo "Wrote $OUTPUT"

