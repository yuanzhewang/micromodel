"""Verifiable rewards for RLVR / GRPO.

The reward is a *verifier*, not a learned model: we check the model's final answer
against GSM8K's ground-truth number. We deliberately reuse the SAME answer-extraction
and gold-parsing as the eval harness (eval/tasks/gsm8k.py) so training optimizes exactly
the metric we report.

TRL GRPO calls each reward fn as f(completions, **cols) where cols are the non-"prompt"
dataset columns (here: "answer"). For conversational prompts, `completions` is a list of
message-lists ([{"role","content"}, ...]); for plain prompts it's a list of strings. We
normalize both.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.tasks.gsm8k import extract_pred, gold_answer  # noqa: E402  (reuse eval logic)

_FMT = re.compile(r"####\s*-?\$?\d")


def _texts(completions):
    """Normalize GRPO completions (conversational message-lists or strings) -> str list."""
    out = []
    for c in completions:
        if isinstance(c, list):  # conversational: last turn is the assistant reply
            out.append(c[-1]["content"])
        else:
            out.append(c)
    return out


def correctness_reward(completions, answer, **kwargs):
    """1.0 if the extracted final answer matches the GSM8K gold number, else 0.0."""
    rewards = []
    for txt, ans in zip(_texts(completions), answer):
        gold = gold_answer(ans) if isinstance(ans, str) else ans
        pred = extract_pred(txt)
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-4
        rewards.append(1.0 if ok else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """Small shaping bonus for emitting a well-formed '#### <number>' final answer."""
    return [0.2 if _FMT.search(t) else 0.0 for t in _texts(completions)]
