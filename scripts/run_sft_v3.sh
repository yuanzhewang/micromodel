#!/usr/bin/env bash
# sft-v3: math-inclusive remix, 1 epoch, SDPA. Train from base -> chat eval.
# Fully detached; survives SSH disconnect / laptop sleep.
set -uo pipefail
cd /lambda/nfs/micromodel/micromodel
mkdir -p eval/results
LOG=eval/results/sft-v3.log
exec >>"$LOG" 2>&1
echo "==================== SFT-V3 START $(date) ===================="
VENV=venv/bin

echo "[train] launch $(date)"
"$VENV/python" scripts/train_sft.py --config configs/sft-v3.yaml 2>&1
TRAIN_RC=$?
echo "[train] done rc=$TRAIN_RC $(date)"

if [ ! -f checkpoints/sft-v3/model.safetensors ]; then
  echo "[train] NO CHECKPOINT produced -> aborting before eval"
  echo "==================== SFT-V3 ABORTED $(date) ===================="
  exit 1
fi

echo "[eval] chat-mode eval $(date)"
"$VENV/python" eval/run_eval.py --model checkpoints/sft-v3 --label sft-v3 --mode chat --batch-size 48 2>&1
echo "[eval] rc=$? $(date)"
echo "==================== SFT-V3 DONE $(date) ===================="
