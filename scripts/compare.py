"""Side-by-side playground — feel how each training stage changed the model.

Loads several checkpoints at once and, for every prompt, prints each model's reply
back to back so you can see the base->SFT->DPO->RLVR progression on the same input.

Defaults to the project ladder: base, sft-v1, dpo-v1, grpo-v1 (skips any missing).
Chat checkpoints use their ChatML template; the raw base model has no template, so
it runs in base/completion mode and is best read as "what the base would continue".

Five 1.7B models in bf16 ~= 17GB — fits one H100. Greedy by default (matches eval);
raise --temp for sampling.

Run:
  venv/bin/python scripts/compare.py
  venv/bin/python scripts/compare.py --prompt "If a shirt costs $12 and is 25% off, what is the price?"
  venv/bin/python scripts/compare.py --models checkpoints/sft-v1,checkpoints/grpo-v1
  venv/bin/python scripts/compare.py --demo            # fixed prompts, then exit
  venv/bin/python scripts/compare.py --temp 0.7

REPL commands:
  /temp X     sampling temp (0 = greedy)      /reppen X   repetition penalty
  /system <text>   set system prompt          /quit
"""
from __future__ import annotations
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.common import load_model_and_tokenizer, has_chat_template

C_USER, C_NAME, C_OFF, C_DIM = "\033[32m", "\033[1;36m", "\033[0m", "\033[90m"

# The project ladder, in order. Labels are short; paths are the checkpoints.
DEFAULT_LADDER = [
    ("base",    "Qwen/Qwen3-1.7B-Base"),
    ("sft-v1",  "checkpoints/sft-v1"),
    ("dpo-v1",  "checkpoints/dpo-v1"),
    ("grpo-v1", "checkpoints/grpo-v1"),
]

INSTRUCTION = "Please reason step by step, and put your final answer after '#### '."


@torch.no_grad()
def gen(model, tok, text, max_new_tokens, temperature, top_p, rep_pen):
    enc = tok(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if rep_pen and rep_pen != 1.0:
        kw["repetition_penalty"] = rep_pen
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw["do_sample"] = False
    out = model.generate(**enc, **kw)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build_input(tok, user_text, system):
    """Chat template if the checkpoint has one; else raw text (base model)."""
    if has_chat_template(tok):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma list of label:path or path; default = base,sft-v1,dpo-v1,grpo-v1")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default; matches eval)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-pen", type=float, default=1.0,
                    help="repetition penalty; base model loops less with ~1.3")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--system", default=None)
    ap.add_argument("--prompt", default=None, help="one-shot prompt then exit")
    ap.add_argument("--demo", action="store_true", help="run fixed demo prompts then exit")
    args = ap.parse_args()

    # Resolve the model list.
    if args.models:
        spec = []
        for item in args.models.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item and not item.startswith("Qwen"):
                label, path = item.split(":", 1)
            else:
                path = item
                label = os.path.basename(path.rstrip("/")) or path
            spec.append((label, path))
    else:
        spec = DEFAULT_LADDER

    # Load each (skip missing local checkpoints rather than crash).
    loaded = []
    for label, path in spec:
        is_local = "/" in path and not path.startswith("Qwen")
        if is_local and not os.path.exists(path):
            print(f"{C_DIM}skip {label}: {path} not found{C_OFF}")
            continue
        print(f"Loading {label} ({path}) ...", flush=True)
        model, tok = load_model_and_tokenizer(path)
        model.eval()
        loaded.append((label, model, tok))
    if not loaded:
        print("no models loaded"); return
    print(f"\nLoaded {len(loaded)} models: {', '.join(l for l, _, _ in loaded)}")

    temp, rep_pen, system = args.temp, args.rep_pen, args.system

    def run_prompt(user_text):
        for label, model, tok in loaded:
            text = build_input(tok, user_text, system)
            r = gen(model, tok, text, args.max_new_tokens, temp, args.top_p, rep_pen)
            print(f"\n{C_NAME}=== {label} ==={C_OFF}\n{r}")
        print()

    if args.demo:
        demo = [
            "If a shirt costs $12 and is now 25% off, what is the sale price?",
            "Natalia sold clips to 48 friends in April, then half as many in May. "
            "How many clips did she sell altogether? " + INSTRUCTION,
            "Write a haiku about the ocean.",
            "List exactly three fruits, each on its own line, numbered.",
        ]
        for p in demo:
            print(f"\n{C_USER}########## PROMPT ##########{C_OFF}\n{p}")
            run_prompt(p)
        return

    if args.prompt is not None:
        print(f"{C_USER}prompt>{C_OFF} {args.prompt}")
        run_prompt(args.prompt); return

    print(f"\nReady. {len(loaded)} models, temp={temp} rep_pen={rep_pen}. /help, /quit to exit.")
    print(f"{C_DIM}tip: append \"{INSTRUCTION}\" to math prompts to compare GSM8K-style answers.{C_OFF}\n")
    while True:
        try:
            user = input(f"{C_USER}you>{C_OFF} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user.startswith("/"):
            parts = user.split(maxsplit=1); cmd = parts[0]; arg = parts[1] if len(parts) > 1 else ""
            if cmd == "/quit":
                break
            elif cmd == "/help":
                print("/temp X   /reppen X   /system <text>   /quit")
            elif cmd == "/temp":
                try: temp = float(arg); print(f"temp={temp}" + ("  (greedy)" if temp == 0 else ""))
                except ValueError: print("usage: /temp 0.7")
            elif cmd == "/reppen":
                try: rep_pen = float(arg); print(f"rep_pen={rep_pen}")
                except ValueError: print("usage: /reppen 1.3")
            elif cmd == "/system":
                system = arg or None; print(f"system={'set' if system else 'cleared'}")
            else:
                print("unknown command; /help")
            continue
        run_prompt(user)


if __name__ == "__main__":
    main()
