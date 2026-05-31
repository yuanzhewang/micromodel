"""Phase 3 DPO — preference-tune an SFT checkpoint (full fine-tune, trl 1.5).

Reads a YAML config (configs/dpo*.yaml). Design points (verified vs installed stack):
  - Starts from an SFT checkpoint (e.g. checkpoints/sft-v1), NOT the base model.
  - Plain ChatML chat template WITHOUT {% generation %} markers: DPO masks the
    prompt via the chosen/rejected pair structure, not assistant-only loss markers.
    Template otherwise matches SFT so prompts are formatted identically.
  - Conversational preference data (chosen/rejected as message-lists). DPOTrainer
    applies tok.chat_template automatically for conversational rows.
  - ref_model=None -> the frozen initial policy is used as the DPO reference.
  - Tokenizer saved with the SAME clean template so eval (--mode chat) / play.py
    speak the format the model was trained on.

Run:
  venv/bin/python scripts/train_dpo.py --config configs/dpo-v1.yaml
"""
from __future__ import annotations

import argparse

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# Plain ChatML (no {% generation %} markers — DPO doesn't need assistant masking).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{{ message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo-v1.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"=== DPO config ===\n{yaml.dump(cfg, sort_keys=False)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.chat_template = CHATML_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- data: conversational preference pairs (chosen / rejected message-lists) ---
    dcfg = cfg.get("dataset_config")
    splits = cfg.get("dataset_splits") or ["train"]
    parts = []
    for sp in splits:
        d = load_dataset(cfg["dataset"], dcfg, split=sp) if dcfg else load_dataset(cfg["dataset"], split=sp)
        keep = {"chosen", "rejected"}
        d = d.remove_columns([c for c in d.column_names if c not in keep])
        print(f"  loaded split {sp!r}: {len(d)} examples", flush=True)
        parts.append(d)
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if cfg.get("subset_size"):
        n = min(cfg["subset_size"], len(ds))
        ds = ds.shuffle(seed=cfg["seed"]).select(range(n))
    print(f"train examples: {len(ds)} | columns: {ds.column_names}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False

    dpo_config = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=cfg.get("beta", 0.1),
        loss_type=cfg.get("loss_type", "sigmoid"),
        max_length=cfg["max_length"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
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
        dataset_num_proc=cfg.get("dataset_num_proc", 8),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\n=== DPO done. Model + tokenizer saved to {cfg['output_dir']} ===", flush=True)


if __name__ == "__main__":
    main()
