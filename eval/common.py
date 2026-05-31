"""Shared eval utilities: model loading + batched greedy generation.

Kept deliberately small and transparent so every step of how a prompt becomes a
score is visible. Works for both the raw base model (`mode="base"`, raw text
completion) and later chat checkpoints (`mode="chat"`, chat template applied).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path: str, dtype=torch.bfloat16, attn: str = "sdpa"):
    """Load a causal LM + tokenizer for evaluation.

    flash-attn isn't installed on this box, so we use PyTorch SDPA attention.
    Left padding is required for correct batched decoder-only generation.
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn,
        device_map="cuda",
    )
    model.eval()
    return model, tok


def has_chat_template(tok) -> bool:
    return getattr(tok, "chat_template", None) is not None


def build_inputs(tok, prompt: str, mode: str) -> str:
    """Turn a raw prompt string into the actual text fed to the model.

    base: returned unchanged (pure completion).
    chat: wrapped as a single user turn via the tokenizer's chat template,
          with the assistant generation prompt appended.
    """
    if mode == "base":
        return prompt
    if mode == "chat":
        if not has_chat_template(tok):
            raise ValueError(
                "mode='chat' but tokenizer has no chat_template "
                "(expected for a raw base model — use mode='base')."
            )
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def batched_generate(
    model,
    tok,
    texts: list[str],
    max_new_tokens: int = 512,
    batch_size: int = 32,
    stop: list[str] | None = None,
) -> list[str]:
    """Greedy-decode completions for a list of already-formatted text prompts.

    Greedy (do_sample=False) for reproducible baselines. Returns only the newly
    generated text (prompt stripped). `stop` truncates each completion at the
    first occurrence of any stop string.
    """
    outputs: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        # Strip the prompt: everything after the (left-padded) input length is new.
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for text in decoded:
            if stop:
                cut = len(text)
                for s in stop:
                    idx = text.find(s)
                    if idx != -1:
                        cut = min(cut, idx)
                text = text[:cut]
            outputs.append(text.strip())
    return outputs
