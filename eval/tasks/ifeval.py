"""IFEval: verifiable instruction-following, scored with Google's official checkers.

We vendor google-research/instruction_following_eval (instructions*.py) into
ifeval_src/ rather than reimplementing the ~25 instruction types, so the numbers
are comparable to the published metric. We compute the four standard accuracies:

  prompt_strict  : all instructions in a prompt satisfied (raw response)
  prompt_loose   : same, but each instruction may be satisfied by any of a few
                   lightly-cleaned response variants (the official "loose" rule)
  inst_strict    : fraction of individual instructions satisfied (raw)
  inst_loose     : fraction satisfied across variants

Headline `score` = prompt_strict (the most-cited single number).
"""
from __future__ import annotations

from datasets import load_dataset

from ..common import build_inputs, batched_generate
from .ifeval_src import instructions_registry


def _variants(response: str) -> list[str]:
    """The official 'loose' response transformations."""
    r = response
    rem_first = "\n".join(r.split("\n")[1:])
    rem_last = "\n".join(r.split("\n")[:-1])
    rev = r.replace("*", "")
    return [r, rem_first, rem_last, rev, rem_first.replace("*", ""), rem_last.replace("*", "")]


def _per_instruction(prompt, instr_ids, kwargs_list, response, loose: bool) -> list[bool]:
    responses = _variants(response) if loose else [response]
    out = []
    for i, iid in enumerate(instr_ids):
        cls = instructions_registry.INSTRUCTION_DICT[iid]
        instr = cls(iid)
        kw = {k: v for k, v in (kwargs_list[i] or {}).items() if v is not None}
        instr.build_description(**kw)
        args = instr.get_instruction_args()
        if args and "prompt" in args:
            kw2 = dict(kw)
            kw2["prompt"] = prompt
            instr.build_description(**kw2)
        followed = False
        for r in responses:
            try:
                if r.strip() and instr.check_following(r):
                    followed = True
                    break
            except Exception:
                continue
        out.append(followed)
    return out


def run(model, tok, mode: str, limit: int | None, batch_size: int, max_new_tokens: int = 1024):
    ds = load_dataset("google/IFEval", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    texts = [build_inputs(tok, p, mode) for p in ds["prompt"]]
    completions = batched_generate(
        model, tok, texts, max_new_tokens=max_new_tokens, batch_size=batch_size
    )

    n_prompt = len(ds)
    p_strict = p_loose = 0
    inst_total = inst_strict = inst_loose = 0
    records = []
    for prompt, iids, kwargs_list, comp in zip(
        ds["prompt"], ds["instruction_id_list"], ds["kwargs"], completions
    ):
        strict = _per_instruction(prompt, iids, kwargs_list, comp, loose=False)
        loose = _per_instruction(prompt, iids, kwargs_list, comp, loose=True)
        p_strict += all(strict)
        p_loose += all(loose)
        inst_total += len(strict)
        inst_strict += sum(strict)
        inst_loose += sum(loose)
        records.append({
            "prompt": prompt,
            "instruction_ids": iids,
            "strict": strict,
            "loose": loose,
            "completion": comp,
        })

    return {
        "task": "ifeval",
        "mode": mode,
        "n": n_prompt,
        "metric": "prompt_strict",
        "score": p_strict / n_prompt if n_prompt else 0.0,
        "prompt_strict": p_strict / n_prompt if n_prompt else 0.0,
        "prompt_loose": p_loose / n_prompt if n_prompt else 0.0,
        "inst_strict": inst_strict / inst_total if inst_total else 0.0,
        "inst_loose": inst_loose / inst_total if inst_total else 0.0,
        "records": records,
    }
