# micromodel

Post-training **Qwen3-1.7B-Base** into a small chat model on a single H100, through
**SFT → DPO → RLVR (GRPO)**, with a fixed eval harness measuring every stage.

The guiding question: *how does each post-training stage move the trade-off between math
reasoning (GSM8K) and instruction-following (IFEval)?*

## Results

GSM8K (exact-match) and IFEval (prompt-strict), chat mode / greedy (base is 8-shot):

| Stage | Model | GSM8K | IFEval |
|-------|-------|:---:|:---:|
| Base (8-shot) | Qwen3-1.7B-Base | 0.771 | 0.214 |
| SFT | sft-v1 | 0.735 | 0.333 |
| SFT | sft-v2 | 0.572 | **0.390** |
| DPO | dpo-v1 | 0.740 | 0.336 |
| **RLVR** | **grpo-v1** | **0.778** | 0.351 |

**grpo-v1** is the best all-round model: RLVR with a verifiable reward recovered the math
SFT had traded away (GSM8K 0.778, beating the base model's few-shot score), while keeping
SFT-level instruction-following. SFT owns the IFEval gains; conservative DPO was a near-no-op.

See **[REPORT.md](REPORT.md)** for the full write-up (goal, plan, execution, learnings, conclusions).

## Layout

- `scripts/train_sft.py · train_dpo.py · train_grpo.py` — the three trainers
- `scripts/rewards.py` — verifiable GSM8K reward for GRPO
- `scripts/run_*.sh` — detached launchers with guarded auto-eval
- `scripts/play.py · compare.py` — single-model REPL / side-by-side stage comparison
- `configs/*.yaml` — one config per run
- `eval/` — eval harness (GSM8K, IFEval w/ vendored checkers, vibe); results in `eval/results/`

## Quickstart

```bash
# Baseline
venv/bin/python eval/run_eval.py --model Qwen/Qwen3-1.7B-Base --mode base --label base

# Train a stage (each launcher auto-evals on completion)
bash scripts/run_dpo_v1.sh        # DPO from sft-v1
bash scripts/run_grpo_v1.sh       # RLVR/GRPO from dpo-v1

# Compare all stages side by side
PYTHONPATH=. venv/bin/python scripts/compare.py --demo
```

Stack: torch 2.6.0+cu124, transformers 5.9.0, trl 1.5.1 (SDPA attention; no flash-attn / no vLLM).
Checkpoints, logs, and the venv live on `$FS` and are gitignored.
