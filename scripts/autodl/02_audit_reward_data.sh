#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MEDICALGPT_ROOT="${MEDICALGPT_ROOT:-$REPO_ROOT}"
INPUT="${1:-$MEDICALGPT_ROOT/data/medical/reward/medical_reward.jsonl}"
OUTPUT="${2:-$MEDICALGPT_ROOT/reports/data_audit_medical_reward.md}"

cd "$MEDICALGPT_ROOT"
python experiments/medical_posttrain/data_audit.py \
  --input "$INPUT" \
  --output "$OUTPUT"

echo "Wrote $OUTPUT"

