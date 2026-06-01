"""Phase 4 RLVR — GRPO with a verifiable GSM8K reward (full fine-tune, trl 1.5).

Reads a YAML config (configs/grpo*.yaml). Design points (verified vs installed stack):
  - Starts from an aligned checkpoint (e.g. checkpoints/dpo-v1), NOT base.
  - Plain ChatML template (same as SFT/DPO, no generation markers). GRPO applies the
    chat template to the conversational `prompt` column automatically.
  - Reward = verifier (scripts/rewards.py), reusing eval/tasks/gsm8k extraction so we
    optimize the exact metric we report. No reward model.
  - vLLM is INCOMPATIBLE with this box's torch 2.6 + transformers 5.9, so use_vllm=False:
    TRL generates rollouts with HF generate (slower but zero dependency risk).
  - ref_model is built internally from the initial policy when beta>0 (KL anchor).

Run:
  venv/bin/python scripts/train_grpo.py --config configs/grpo-v1.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rewards import correctness_reward, format_reward  # noqa: E402

# Plain ChatML (matches SFT/DPO; no {% generation %} markers needed for GRPO).
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{{ message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

INSTRUCTION = "Please reason step by step, and put your final answer after '#### '."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo-v1.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"=== GRPO config ===\n{yaml.dump(cfg, sort_keys=False)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.chat_template = CHATML_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- data: GSM8K train -> {prompt: [user turn], answer: raw "#### N" string} ---
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if cfg.get("subset_size"):
        ds = ds.select(range(min(cfg["subset_size"], len(ds))))

    def to_prompt(ex):
        return {
            "prompt": [{"role": "user", "content": f"{ex['question']}\n\n{INSTRUCTION}"}],
            "answer": ex["answer"],
        }

    ds = ds.map(to_prompt, remove_columns=ds.column_names)
    print(f"train prompts: {len(ds)} | columns: {ds.column_names}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False

    grpo_config = GRPOConfig(
        output_dir=cfg["output_dir"],
        num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        beta=cfg.get("beta", 0.04),
        num_iterations=cfg.get("num_iterations", 1),
        scale_rewards=cfg.get("scale_rewards", True),
        reward_weights=cfg.get("reward_weights"),
        use_vllm=False,  # incompatible stack; HF generate rollouts
        log_completions=cfg.get("log_completions", True),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant_with_warmup"),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 1),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 1),
        seed=cfg.get("seed", 42),
        report_to=cfg.get("report_to", "none"),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_config,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\n=== GRPO done. Model + tokenizer saved to {cfg['output_dir']} ===", flush=True)


if __name__ == "__main__":
    main()
