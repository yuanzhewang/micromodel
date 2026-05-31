# Eval harness

A thin, transparent eval suite used as the fixed yardstick across every stage of
the pipeline (base → SFT → DPO → RLVR). Same harness, two modes:

- `--mode base` — raw text completion (few-shot for GSM8K). Use on the base model.
- `--mode chat` — applies the tokenizer's chat template, 0-shot. Use on chat
  checkpoints (SFT and later).

## Tasks

| Task | What | Scoring | Notes |
|------|------|---------|-------|
| `gsm8k` | grade-school math | numeric exact-match | base: 8-shot CoT; chat: 0-shot "reason then `#### answer`" |
| `ifeval` | instruction following | Google's official checkers (vendored in `tasks/ifeval_src/`) | reports prompt/instruction × strict/loose; headline = `prompt_strict` |
| `vibe` | 15 fixed chat prompts | none (qualitative) | generations saved for eyeballing & side-by-side |

## Run

```bash
# from anywhere, with the project venv
venv/bin/python eval/run_eval.py \
    --model Qwen/Qwen3-1.7B-Base --mode base \
    --tasks gsm8k,ifeval,vibe --label base --batch-size 48
```

Outputs (under `eval/results/`):
- `<label>_<mode>.json` — summary: scores + config + per-task metrics
- `<label>_<mode>_<task>.jsonl` — every example with its generation (for inspection)

Plot the scored tasks across all runs:

```bash
venv/bin/python eval/plot.py     # writes eval/results/progression.png
```

## Layout

```
eval/
  common.py            model load + batched greedy generation (SDPA, left-pad)
  run_eval.py          CLI runner; resilient (one task failing doesn't kill others)
  plot.py              bar chart across runs
  vibe_prompts.json    the qualitative prompt set
  tasks/
    gsm8k.py  ifeval.py  vibe.py
    ifeval_src/          vendored google-research instruction_following_eval checkers
  results/             summaries + generations + plots (committed)
```

## Notes / decisions
- Greedy decoding (`do_sample=False`) for reproducible numbers.
- flash-attn isn't installed → `attn_implementation="sdpa"`.
- IFEval deps: `nltk` (punkt/punkt_tab), `langdetect`, `immutabledict`, `absl-py`.
- The base model ships **no chat template**, so `--mode chat` only works after SFT.
