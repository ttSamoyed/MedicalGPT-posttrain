#!/usr/bin/env bash
set -euo pipefail

echo "== Python =="
python --version

echo "== CUDA =="
nvidia-smi || true

echo "== Git =="
git --version

echo "== Working directory =="
pwd

echo "== Important paths =="
echo "MEDICALGPT_ROOT=${MEDICALGPT_ROOT:-$(pwd)}"
echo "HF_ENDPOINT=${HF_ENDPOINT:-}"
echo "HF_HOME=${HF_HOME:-}"

