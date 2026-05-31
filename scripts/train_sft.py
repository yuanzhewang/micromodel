"""Phase 2 SFT — Qwen3-1.7B-Base -> chat model (full fine-tune, trl 1.5).

Reads a YAML config (configs/sft*.yaml). Key design points (verified against the
installed stack):
  - Plain ChatML chat template WITH {% generation %} markers, so TRL can mask the
    loss to assistant turns only (assistant_only_loss=True). We override Qwen3's
    default template to drop its empty <think></think> blocks — cleaner outputs.
  - Full fine-tune, bf16, attention impl from config (sdpa default; flash_attention_2
    when available). packing + gradient checkpointing for throughput/memory.
  - The trained tokenizer is saved with the SAME clean template, so eval (--mode
    chat) and scripts/play.py speak the same format the model was trained on.

Run:
  PYTHONPATH=. venv/bin/python scripts/train_sft.py --config configs/sft.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Plain ChatML with assistant-turn loss markers (no <think> blocks).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{{ '<|im_end|>' }}{% endgeneration %}"
    "{% else %}"
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endif %}"
    "{{ '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--subset-size", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.subset_size is not None:
        cfg["subset_size"] = args.subset_size
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    print(f"=== SFT config ===\n{yaml.dump(cfg, sort_keys=False)}", flush=True)

    # --- tokenizer with clean ChatML + generation markers ---
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.chat_template = CHATML_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- data: conversational 'messages' dataset ---
    # smoltalk2's 'SFT' config exposes per-source splits; we keep only the
    # 'messages' column so splits with differing extra columns (e.g. 'source',
    # 'chat_template_kwargs') concatenate cleanly into one training set.
    dcfg = cfg.get("dataset_config")
    splits = cfg.get("dataset_splits") or ["train"]
    # Optional per-split caps (parallel list to dataset_splits): take only the
    # first N of each split (after a per-split shuffle) before concatenating, so
    # the data mix is controllable (e.g. fixed math fraction). 0/null = keep all.
    caps = cfg.get("dataset_split_sizes")
    parts = []
    for i, sp in enumerate(splits):
        d = load_dataset(cfg["dataset"], dcfg, split=sp) if dcfg else load_dataset(cfg["dataset"], split=sp)
        if "messages" not in d.column_names and "conversations" in d.column_names:
            d = d.rename_column("conversations", "messages")
        d = d.remove_columns([c for c in d.column_names if c != "messages"])
        cap = caps[i] if caps and i < len(caps) else None
        if cap:
            k = min(cap, len(d))
            d = d.shuffle(seed=cfg["seed"]).select(range(k))
        print(f"  loaded split {sp!r}: {len(d)} examples"
              + (f" (capped to {cap})" if cap else ""), flush=True)
        parts.append(d)
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if cfg.get("subset_size"):
        n = min(cfg["subset_size"], len(ds))
        ds = ds.shuffle(seed=cfg["seed"]).select(range(n))
    print(f"train examples: {len(ds)} | columns: {ds.column_names}", flush=True)

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        seed=cfg["seed"],
        report_to=cfg["report_to"],
        max_length=cfg["max_length"],
        packing=cfg["packing"],
        assistant_only_loss=cfg["assistant_only_loss"],
        dataset_num_proc=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\n=== SFT done. Model + tokenizer saved to {cfg['output_dir']} ===", flush=True)


if __name__ == "__main__":
    main()
