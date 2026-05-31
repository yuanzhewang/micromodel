"""Run the eval suite on one model and save results.

Usage (from anywhere, with the project venv):
  venv/bin/python eval/run_eval.py --model Qwen/Qwen3-1.7B-Base --mode base \
      --tasks gsm8k,ifeval,vibe --label base

Writes per run:
  eval/results/<label>_<mode>.json          summary (scores + config)
  eval/results/<label>_<mode>_<task>.jsonl  per-example records (generations)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make `import eval...` work no matter the cwd or how this file is invoked
# (python -m eval.run_eval, python eval/run_eval.py, or from a background job).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.common import load_model_and_tokenizer, has_chat_template

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def get_task(name):
    """Lazy import so an optional task's missing deps don't block the others."""
    if name == "gsm8k":
        from eval.tasks import gsm8k
        return gsm8k.run
    if name == "vibe":
        from eval.tasks import vibe
        return vibe.run
    if name == "ifeval":
        from eval.tasks import ifeval
        return ifeval.run
    raise ValueError(f"unknown task: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--mode", choices=["base", "chat"], default="base")
    ap.add_argument("--tasks", default="gsm8k,ifeval,vibe")
    ap.add_argument("--label", default=None, help="short name for this run; defaults to model basename")
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (for quick smoke runs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override; default lets each task pick (gsm8k/vibe 512, ifeval 1024)")
    args = ap.parse_args()

    label = args.label or Path(args.model).name
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} (mode={args.mode}) ...", flush=True)
    model, tok = load_model_and_tokenizer(args.model)
    if args.mode == "chat" and not has_chat_template(tok):
        raise SystemExit("mode=chat but model has no chat_template; use --mode base for a raw base model.")
    print(f"  loaded. chat_template present: {has_chat_template(tok)}", flush=True)

    summary = {
        "label": label,
        "model": args.model,
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"limit": args.limit, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens},
        "scores": {},
    }

    extra = {"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else {}

    for tname in tasks:
        print(f"\n=== running {tname} ===", flush=True)
        t0 = time.time()
        try:
            run = get_task(tname)
            result = run(model, tok, args.mode, args.limit, args.batch_size, **extra)
        except Exception as e:
            print(f"  FAILED {tname}: {e!r}\n{traceback.format_exc()}", flush=True)
            summary["scores"][tname] = {"task": tname, "error": repr(e), "score": None}
            continue
        dt = time.time() - t0

        # split detailed records out to JSONL; keep scores in the summary
        records = result.pop("records", [])
        detail_path = RESULTS_DIR / f"{label}_{args.mode}_{tname}.jsonl"
        with detail_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        result["seconds"] = round(dt, 1)
        summary["scores"][tname] = result
        score = result.get("score")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  {tname}: score={score_str} (n={result.get('n')}, {dt:.1f}s) -> {detail_path.name}", flush=True)

    summary_path = RESULTS_DIR / f"{label}_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to {summary_path}")
    print(json.dumps({t: s.get("score") for t, s in summary["scores"].items()}, indent=2))


if __name__ == "__main__":
    main()
