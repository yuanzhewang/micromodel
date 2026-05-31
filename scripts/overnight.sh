#!/usr/bin/env bash
set -uo pipefail
cd /lambda/nfs/micromodel/micromodel
export HF_HOME=/lambda/nfs/micromodel/hf-cache
export MAX_JOBS=8
VENV=venv/bin
LOG=eval/results/overnight.log
exec >>"$LOG" 2>&1

echo "==================== OVERNIGHT START $(date) ===================="

CFG=configs/sft-v2.yaml   # default = proven SDPA path
echo "[fa] installing ninja + attempting flash-attn (<=40min)..."
$VENV/pip install -q ninja packaging 2>&1 | tail -2 || true
timeout 2400 $VENV/pip install flash-attn --no-build-isolation 2>&1 | tail -8 || echo "[fa] install failed/timed out"
if $VENV/python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null; then
  echo "[fa] flash-attn OK -> FAST packed config"
  CFG=configs/sft-v2-flash.yaml
else
  echo "[fa] flash-attn unavailable -> PROVEN SDPA config"
fi

echo "[train] using $CFG  $(date)"
PYTHONPATH=. $VENV/python scripts/train_sft.py --config "$CFG"
rc=$?
echo "[train] rc=$rc  $(date)"

if [ $rc -ne 0 ] || [ ! -f checkpoints/sft-v2/model.safetensors ]; then
  echo "[train] FAST path did not produce a checkpoint -> FALLBACK to proven SDPA config"
  PYTHONPATH=. $VENV/python scripts/train_sft.py --config configs/sft-v2.yaml
  echo "[train] fallback rc=$?  $(date)"
fi

if [ -f checkpoints/sft-v2/model.safetensors ]; then
  echo "[eval] starting chat-mode eval of sft-v2  $(date)"
  PYTHONPATH=. $VENV/python eval/run_eval.py --model checkpoints/sft-v2 --mode chat \
      --tasks gsm8k,ifeval,vibe --label sft-v2 --batch-size 48
  echo "[eval] rc=$?  $(date)"
else
  echo "[eval] SKIPPED — no checkpoint produced"
fi

echo "==================== OVERNIGHT DONE $(date) ===================="
