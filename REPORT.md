# Micromodel — Technical Report

**Taking Qwen3-1.7B-Base to a small chat model via SFT → DPO → RLVR**

Single-GPU (H100 80GB) post-training study. Dates: 2026-05-31 → 2026-06-01.
Model: `Qwen/Qwen3-1.7B-Base` (28 layers, hidden 2048, GQA 16/8 heads, vocab 151936, bf16).
Stack: torch 2.6.0+cu124, transformers 5.9.0, trl 1.5.1 (SDPA attention; no flash-attn / no vLLM on this box).

---

## 1. Goal

Learn modern post-training hands-on by building a small chat model from a base LM, with full
transparency at each stage and a fixed yardstick to measure progress. Concretely:

1. Turn an instruction-naive base model into a usable chat model (SFT).
2. Apply preference optimization (DPO).
3. Apply reinforcement learning with a verifiable reward (RLVR / GRPO).
4. Measure every stage on the same benchmarks and keep the whole thing reviewable.

The scientific question threaded through the project: **how does each post-training stage move
the trade-off between math reasoning (GSM8K) and instruction-following (IFEval)?**

## 2. Plan

A phased pipeline, stopping for review between phases:

| Phase | Stage | Method | Data |
|------|-------|--------|------|
| 1 | Eval harness + baseline | thin custom harness | GSM8K, IFEval, vibe |
| 2 | SFT (base → chat) | TRL `SFTTrainer`, assistant-only loss | smoltalk2 (chat/IF mix) |
| 3 | DPO | TRL `DPOTrainer`, full-FT, frozen ref | ultrafeedback_binarized |
| 4 | RLVR | TRL `GRPOTrainer`, verifiable reward | GSM8K train (verifier) |
| 5 | Play & compare | interactive CLIs | — |

**Key constraints (environment):** only `/lambda/nfs/micromodel` (`$FS`) persists across box
restarts, so all code/data/checkpoints/HF-cache live there. The installed library *majors* are
newer than most tutorials (trl 1.5, transformers 5.9), so APIs were verified against the
installed code before writing each trainer.

## 3. Evaluation harness (Phase 1)

A thin, dependency-light harness (`eval/`) reused unchanged at every stage — the fixed yardstick.

- **GSM8K** (`eval/tasks/gsm8k.py`): grade-school math, numeric exact-match. Base model evaluated
  8-shot CoT; chat models 0-shot with "put your final answer after `#### `". Forgiving answer
  extraction (prefers `#### N` / "The answer is N", falls back to last number).
- **IFEval** (`eval/tasks/ifeval.py`): instruction-following with Google's official checkers
  *vendored* in `eval/tasks/ifeval_src/`. Headline metric = prompt-level strict accuracy.
- **Vibe** (`eval/tasks/vibe.py`): 15 qualitative prompts; generations saved, not scored.
- Greedy decoding, `--mode base|chat`, results as JSON in `eval/results/`. Runner: `eval/run_eval.py`.

## 4. Execution

All training was run detached (`setsid`/`nohup`, reparented to init) so it survived SSH
disconnects, with guarded launchers that auto-ran the eval on completion.

### Phase 2 — SFT (`scripts/train_sft.py`, `configs/sft*.yaml`)

Full fine-tune from base. A plain **ChatML** template was adopted with `{% generation %}` markers
so loss is masked to assistant turns only (`assistant_only_loss=True`); Qwen3's default `<think>`
blocks were deliberately dropped for clean outputs. SDPA attention; `packing=false` (packing needs
flash-attn varlen masking, unavailable here). Data: HuggingFaceTB/smoltalk2 `_no_think` splits
(broad instruction + IF-personas + everyday chat + system-prompt following).

Three SFT runs formed a small sweep:
- **sft-v1** — 1 split (magpie), 8k examples, 1 epoch — pipeline validation.
- **sft-v2** — 4-split mix, 100k examples, 2 epochs.
- **sft-v3** — same 100k mix, **1 epoch** (single-variable over-training control).

### Phase 3 — DPO (`scripts/train_dpo.py`, `configs/dpo-v1.yaml`)

Full-FT DPO from `sft-v1`, `ref_model=None` (the frozen initial policy is the KL reference),
plain ChatML (no generation markers — DPO masks the prompt via the chosen/rejected pair).
Data: `trl-lib/ultrafeedback_binarized` (62k conversational preference pairs). **β=0.1**, sigmoid
loss, **LR 5e-7** (DPO needs a far smaller LR than SFT), 1 epoch.

### Phase 4 — RLVR / GRPO (`scripts/train_grpo.py`, `scripts/rewards.py`, `configs/grpo-v1.yaml`)

GRPO from `dpo-v1`. The reward is a **verifier, not a learned model**: `scripts/rewards.py`
reuses the eval harness's GSM8K answer-extraction so training optimizes exactly the reported
metric — `correctness` (0/1 exact-match) plus a small `format` bonus (0.2 for a well-formed
`#### N`). G=8 generations/prompt, temperature 1.0 (exploration), **β=0.04**, `num_iterations=1`
(reduces to REINFORCE-with-baseline), LR 1e-6, `max_steps=300`, `use_vllm=False`. Data: GSM8K
**train** split (7,473 prompts) — the test split was never touched.

## 5. Test results

All numbers are on the held-out benchmarks, chat mode (base is 8-shot), greedy decoding.

| Stage | Model | GSM8K (exact-match) | IFEval (prompt-strict) | IFEval (inst-strict) |
|-------|-------|:---:|:---:|:---:|
| Base (8-shot) | Qwen3-1.7B-Base | 0.771 | 0.214 | 0.321 |
| SFT | sft-v1 (8k, 1ep) | 0.735 | 0.333 | 0.458 |
| SFT | sft-v2 (100k, 2ep) | 0.572 | **0.390** | **0.504** |
| SFT | sft-v3 (100k, 1ep) | 0.562 | 0.373 | 0.488 |
| DPO | dpo-v1 (from sft-v1) | 0.740 | 0.336 | 0.463 |
| **RLVR** | **grpo-v1 (from dpo-v1)** | **0.778** | 0.351 | 0.466 |

Headline: **grpo-v1 reaches GSM8K 0.778 (1026/1319)** — the best in the project, and it *beats
the base model's 8-shot score (0.771) while answering 0-shot in chat format.* IFEval peaks at the
SFT stage (sft-v2, 0.390).

## 6. Learnings

**(a) The SFT data mix is a GSM8K ↔ IFEval trade-off, and it's about scale+mix, not epochs.**
sft-v1 (one broad split) kept GSM8K high (0.735) but modest IFEval (0.333). Adding three
chat/IF-heavy splits and scaling to 100k (sft-v2) raised IFEval to 0.390 but dropped GSM8K to
0.572. The natural hypothesis was over-training (2 epochs). **sft-v3 refuted it**: the identical
mix at 1 epoch landed at 0.562 — statistically the same as v2. So training *length* was not the
lever; the chat/IF data itself crowds out math and adds verbosity. (Diagnostics also showed median
completion length ~2.3× longer and frequent rambling in v2.)

**(b) Conservative DPO is a near-no-op.** dpo-v1 was tuned (LR 5e-7, β=0.1, 1 epoch) to *preserve*
sft-v1's math, and it succeeded so well it changed almost nothing: GSM8K +0.005, IFEval +0.003.
The training metrics confirm it — `train_loss` fell only ~0.007 from the `log 2 ≈ 0.693` starting
point, reward margin ≈ 0.03, ranking accuracy ≈ 0.56 (barely above chance). The KL leash was too
tight and the steps too small for the preference signal to take hold. A useful negative result:
DPO requires a more aggressive setting to contribute.

**(c) RLVR delivered, and the GRPO health signals held.** Verifiable-reward RL is the right tool
for a checkable skill like math. Train correctness reward rose from ~0.31 to ~0.7; crucially
`frac_reward_zero_std = 0` throughout — every group of 8 samples had mixed right/wrong answers, so
every prompt produced a usable advantage (GRPO standardizes reward within a group; a group that's
all-right or all-wrong yields zero gradient). KL stayed ~0.002 and entropy ~1.0 (no policy
collapse). RLVR recovered the math that SFT had traded away — without ever seeing the test set.

**(d) Environment / tooling lessons.**
- **vLLM is incompatible with this stack.** No vLLM version works with both torch 2.6 *and*
  transformers 5.9: latest vLLM demands torch 2.11/CUDA 13 (a 130-package upgrade that would break
  the pipeline), while the torch-2.6-compatible vLLM 0.8.5 calls `all_special_tokens_extended`,
  an API transformers 5.x removed. GRPO therefore ran `use_vllm=False` (HF-generate rollouts) —
  slower but zero dependency risk.
- **Verify library APIs before writing trainers.** trl 1.5.1's `DPOConfig`/`GRPOConfig` dropped
  kwargs present in tutorials (`max_prompt_length`; `dataset_num_proc` on GRPOConfig). Each caused
  an immediate `TypeError` until removed — cheap to catch by introspecting the dataclass fields.
- **NFS write-visibility + tool-output bundling lag** made naive monitoring unreliable; writing
  results to `/tmp` and reading them back, plus strict log markers, was the robust pattern.

## 7. Conclusions

The full **base → SFT → DPO → RLVR** pipeline was built, run, and measured end-to-end on a single
H100, and each stage's effect is cleanly attributable:

- **SFT** is what turns the base model into a chat model — it owns instruction-following (IFEval
  0.214 → 0.39) but trades away some math via verbose chat data.
- **DPO** (as configured) was a safe no-op; preference optimization needs a less conservative
  setting to move the model.
- **RLVR** is the math win: a verifiable reward pushed GSM8K to **0.778**, the project best,
  exceeding even the base model's few-shot score.

**Best all-round chat model: `grpo-v1`** (strong math + SFT-level instruction-following).

### Future work
- **grpo-v2**: ~40% of GRPO generations hit the 512-token cap (truncated → unscorable), so raising
  `max_completion_length` and training longer is the clearest lever for more GSM8K.
- **dpo-v2**: a more aggressive LR/β (and possibly 2 epochs) to make DPO actually move IFEval.
- A math-augmented SFT mix to test whether the GSM8K↔IFEval trade-off can be widened at the SFT
  stage rather than recovered later by RL.

## 8. Reproducing

```bash
# Phase 1 — baseline
venv/bin/python eval/run_eval.py --model Qwen/Qwen3-1.7B-Base --mode base --label base

# Phase 2 — SFT (each run auto-evals on completion via its launcher)
venv/bin/python scripts/train_sft.py --config configs/sft.yaml      # sft-v1
bash scripts/run_sft_v3.sh                                          # sft-v3 (detached pattern)

# Phase 3 — DPO
bash scripts/run_dpo_v1.sh

# Phase 4 — RLVR / GRPO
bash scripts/run_grpo_v1.sh

# Play / compare stages side by side
PYTHONPATH=. venv/bin/python scripts/compare.py --demo
```

### Repository map
- `scripts/train_sft.py · train_dpo.py · train_grpo.py` — the three trainers.
- `scripts/rewards.py` — verifiable GSM8K reward for GRPO.
- `scripts/run_*_v1.sh`, `run_sft_v3.sh`, `overnight.sh` — detached launchers with guarded auto-eval.
- `scripts/play.py · compare.py` — single-model REPL and side-by-side stage comparison.
- `configs/*.yaml` — one config per run.
- `eval/` — harness (`run_eval.py`, `common.py`, `tasks/`, vendored IFEval checkers); results in `eval/results/`.

*Checkpoints, logs, and the venv are gitignored (kept on `$FS`, not committed).*
