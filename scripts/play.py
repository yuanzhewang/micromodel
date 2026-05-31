"""Interactive playground — feel the model's raw output.

Loads the model once, then a REPL. Two modes:
  chat  - applies the ChatML template, keeps multi-turn history.
          Ask COMPLETE questions/requests here:  "What is the capital of France?"
  base  - raw text completion (no template, no history).
          Give it a FRAGMENT to continue here:   "The capital of France is"

Picking the wrong mode for your input is the #1 cause of bad output on the BASE
model: a fragment ("the capital of france is") sent in chat mode looks like a
broken user message, so the model flails. Use base mode for fragments.

For the *base* model we default to GREEDY decoding (temp 0) + a repetition
penalty, which keeps it coherent. With sampling it tends to degenerate into
repeated-token loops on instruction-style prompts. After SFT this is robust and
you can raise temp freely.

Run:
  venv/bin/python scripts/play.py                      # chat mode, greedy
  venv/bin/python scripts/play.py --mode base
  venv/bin/python scripts/play.py --temp 0.7           # sampling (wilder; risky on base)
  venv/bin/python scripts/play.py --model checkpoints/sft-v1   # later: your SFT model
  venv/bin/python scripts/play.py --prompt "Once upon a time"  # one-shot
  venv/bin/python scripts/play.py --demo               # fixed demo prompts, then exit

REPL commands:
  /chat  /base      switch mode          /temp 0.9   sampling temp (0 = greedy)
  /mode chat|base   switch mode          /reppen 1.3 repetition penalty
  /system <text>    set system prompt    /reset      clear chat history
  /help   /quit
"""
from __future__ import annotations
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.common import load_model_and_tokenizer, has_chat_template
from transformers import TextStreamer

C_USER, C_BOT, C_OFF, C_DIM = "\033[32m", "\033[36m", "\033[0m", "\033[90m"


def chat_text(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen(model, tok, text, max_new_tokens, temperature, top_p, rep_pen, stream):
    enc = tok(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if rep_pen and rep_pen != 1.0:
        kw["repetition_penalty"] = rep_pen
    if stream:
        kw["streamer"] = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kw["do_sample"] = False
    out = model.generate(**enc, **kw)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--mode", choices=["chat", "base"], default="chat")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default; best for base model)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-pen", type=float, default=1.3, help="repetition penalty (suppresses loops)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--system", default=None)
    ap.add_argument("--prompt", default=None, help="one-shot prompt then exit")
    ap.add_argument("--demo", action="store_true", help="run fixed demo prompts then exit")
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    model, tok = load_model_and_tokenizer(args.model)
    mode = args.mode
    if mode == "chat" and not has_chat_template(tok):
        print("(no chat template; using base mode)"); mode = "base"
    temp, rep_pen, system, history = args.temp, args.rep_pen, args.system, []

    is_base = "base" in args.model.lower()
    if is_base:
        print(f"\n{C_DIM}  NOTE: raw BASE model — it completes text, it doesn't reliably follow")
        print(f"  instructions. In CHAT mode ask COMPLETE questions ('What is the capital")
        print(f"  of France?'); for FRAGMENTS ('The capital of France is') use /base.")
        print(f"  Use /reset between unrelated questions. Expect rambling — SFT fixes it.{C_OFF}")

    def reply_to(user_text, m):
        if m == "chat":
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   history + [{"role": "user", "content": user_text}]
            text = chat_text(tok, msgs)
        else:
            text = user_text
        print(C_BOT, end="")
        r = gen(model, tok, text, args.max_new_tokens, temp, args.top_p, rep_pen, stream=True)
        print(C_OFF, flush=True)
        return r

    if args.demo:
        demo = [("chat", "Give me three quick tips for staying focused while studying."),
                ("chat", "Write a haiku about the ocean."),
                ("base", "The capital of France is"),
                ("base", "def fibonacci(n):\n")]
        for m, p in demo:
            print(f"\n===== [{m}] {p!r} =====")
            reply_to(p, m)
        return

    if args.prompt is not None:
        reply_to(args.prompt, mode); return

    print(f"\nReady. mode={mode} temp={temp} rep_pen={rep_pen}. /help for commands, /quit to exit.")
    print(f"{C_DIM}tip: /chat for questions, /base for fragments, /reset to clear history.{C_OFF}\n")
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
                print("/chat  /base   /mode chat|base   /temp X   /reppen X   /system <text>   /reset   /quit")
            elif cmd in ("/chat", "/base"):
                mode = cmd[1:]; print(f"mode={mode}")
            elif cmd == "/mode" and arg in ("chat", "base"):
                mode = arg; print(f"mode={mode}")
            elif cmd == "/temp":
                try: temp = float(arg); print(f"temp={temp}" + ("  (greedy)" if temp == 0 else ""))
                except ValueError: print("usage: /temp 0.7")
            elif cmd == "/reppen":
                try: rep_pen = float(arg); print(f"rep_pen={rep_pen}")
                except ValueError: print("usage: /reppen 1.3")
            elif cmd == "/system":
                system = arg or None; history.clear(); print("system set; history cleared")
            elif cmd == "/reset":
                history.clear(); print("history cleared")
            else:
                print("unknown command; /help")
            continue
        r = reply_to(user, mode)
        if mode == "chat":
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": r})


if __name__ == "__main__":
    main()
