"""GSM8K: grade-school math word problems, generative + numeric exact-match.

base mode  -> classic 8-shot chain-of-thought completion prompt.
chat mode  -> 0-shot, ask the model to reason and end with '#### <answer>'.

Answer extraction is intentionally forgiving: prefer an explicit
'The answer is X' / '#### X', else fall back to the last number in the text.
"""
from __future__ import annotations

import re

from datasets import load_dataset

from ..common import build_inputs, batched_generate

# Canonical 8-shot CoT exemplars (Wei et al., 2022), the de-facto GSM8K few-shot set.
FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. "
     "So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
     "After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
     "So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
     "5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, "
     "from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
     "So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
     "How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
     "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 = 8 dollars left. The answer is 8."),
]

_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _normalize_number(s: str):
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(ref: str):
    """GSM8K gold answers are the number after '####'."""
    after = ref.split("####")[-1].strip()
    return _normalize_number(after)


def extract_pred(text: str):
    # 1) explicit markers
    for pat in (r"####\s*([^\n]+)", r"[Tt]he answer is\s*([^\n.]+)"):
        m = re.search(pat, text)
        if m:
            nums = _NUM.findall(m.group(1))
            if nums:
                return _normalize_number(nums[-1])
    # 2) fall back to the last number anywhere in the completion
    nums = _NUM.findall(text)
    if nums:
        return _normalize_number(nums[-1])
    return None


def _fewshot_block() -> str:
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in FEWSHOT)


def build_prompt(question: str, mode: str) -> str:
    if mode == "base":
        return f"{_fewshot_block()}\n\nQ: {question}\nA:"
    # chat
    return (
        f"{question}\n\n"
        "Please reason step by step, and put your final answer after '#### '."
    )


def run(model, tok, mode: str, limit: int | None, batch_size: int, max_new_tokens: int = 512):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    raw_prompts = [build_prompt(q, mode) for q in ds["question"]]
    texts = [build_inputs(tok, p, mode) for p in raw_prompts]
    # In base mode the model will happily hallucinate the next "Q:"; stop there.
    stop = ["\nQ:", "\n\nQ:"] if mode == "base" else None

    completions = batched_generate(
        model, tok, texts, max_new_tokens=max_new_tokens, batch_size=batch_size, stop=stop
    )

    records, correct = [], 0
    for q, ref, comp in zip(ds["question"], ds["answer"], completions):
        gold = gold_answer(ref)
        pred = extract_pred(comp)
        ok = pred is not None and gold is not None and abs(pred - gold) < 1e-4
        correct += ok
        records.append({"question": q, "gold": gold, "pred": pred, "correct": ok, "completion": comp})

    n = len(records)
    return {
        "task": "gsm8k",
        "mode": mode,
        "n": n,
        "metric": "exact_match",
        "score": correct / n if n else 0.0,
        "correct": correct,
        "records": records,
    }
