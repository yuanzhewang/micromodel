"""Vibe set: a small fixed prompt suite generated and saved for human eyeballing.

No automatic score — the point is to *read* the generations and watch them go
from "base model rambles / continues the prompt" to "chat model answers".
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common import build_inputs, batched_generate

PROMPTS_FILE = Path(__file__).resolve().parent.parent / "vibe_prompts.json"


def run(model, tok, mode: str, limit: int | None, batch_size: int, max_new_tokens: int = 512):
    prompts = json.loads(PROMPTS_FILE.read_text())
    if limit:
        prompts = prompts[:limit]

    texts = [build_inputs(tok, p["prompt"], mode) for p in prompts]
    completions = batched_generate(
        model, tok, texts, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    records = [
        {"id": p["id"], "category": p["category"], "prompt": p["prompt"], "completion": c}
        for p, c in zip(prompts, completions)
    ]
    return {
        "task": "vibe",
        "mode": mode,
        "n": len(records),
        "metric": "none (qualitative)",
        "score": None,
        "records": records,
    }
