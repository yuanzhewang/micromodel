#!/usr/bin/env bash
set -uo pipefail
cd /lambda/nfs/micromodel/micromodel
export PYTHONPATH=/lambda/nfs/micromodel/micromodel
exec >>eval/results/grpo-v1.log 2>&1

echo "==================== GRPO-V1 START $(date) ===================="
venv/bin/python scripts/train_grpo.py --config configs/grpo-v1.yaml

if [ ! -f checkpoints/grpo-v1/model.safetensors ]; then
  echo "ABORTED: no checkpoint produced"
  exit 1
fi

echo "[eval] running chat-mode eval"
venv/bin/python eval/run_eval.py --model checkpoints/grpo-v1 --label grpo-v1 --mode chat --batch-size 48
echo "[eval] rc=$? $(date)"
echo "==================== GRPO-V1 DONE $(date) ===================="
