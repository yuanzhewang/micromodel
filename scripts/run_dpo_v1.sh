#!/usr/bin/env bash
set -uo pipefail
cd /lambda/nfs/micromodel/micromodel
exec >>eval/results/dpo-v1.log 2>&1

echo "==================== DPO-V1 START $(date) ===================="
venv/bin/python scripts/train_dpo.py --config configs/dpo-v1.yaml

if [ ! -f checkpoints/dpo-v1/model.safetensors ]; then
  echo "ABORTED: no checkpoint produced"
  exit 1
fi

echo "[eval] running chat-mode eval"
venv/bin/python eval/run_eval.py --model checkpoints/dpo-v1 --label dpo-v1 --mode chat --batch-size 48
echo "[eval] rc=$? $(date)"
echo "==================== DPO-V1 DONE $(date) ===================="
