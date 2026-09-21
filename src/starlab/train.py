"""Supervised fine-tuning and exact-verification evaluation for the tiny model.

Everything here is plain PyTorch on CPU. ``sft`` trains next-token cross
entropy on the CoT span; ``evaluate`` samples the model's own chain and grades
it with the pure verifier from ``data.py`` — so accuracy is measured on real
generated arithmetic, never on teacher forcing.
"""

from __future__ import annotations

import random

import torch

from .data import Example, score_solution
from .model import TinyTransformer


def tensors(examples: list[Example]):
    prompts = torch.tensor([e.prompt() for e in examples], dtype=torch.long)
    cots = torch.tensor([e.cot() for e in examples], dtype=torch.long)
    return prompts, cots


def tensors_pairs(pairs: list[tuple[Example, list[int]]]):
    """Prompt + (self-generated) CoT tensors from (Example, cot) pairs."""
    prompts = torch.tensor([ex.prompt() for ex, _ in pairs], dtype=torch.long)
    cots = torch.tensor([cot for _, cot in pairs], dtype=torch.long)
    return prompts, cots


def sft_pairs(model: TinyTransformer, pairs: list[tuple[Example, list[int]]], *,
              steps: int = 400, batch: int = 128, lr: float = 3e-3, seed: int = 0):
    """Fine-tune on arbitrary (Example, cot) supervision — gold or self-generated."""
    torch.manual_seed(seed)
    prompts, cots = tensors_pairs(pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = prompts.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        loss = model.sft_loss(prompts[idx], cots[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def sft(model: TinyTransformer, examples: list[Example], *, steps: int = 400,
        batch: int = 128, lr: float = 3e-3, seed: int = 0,
        log_every: int = 50) -> list[float]:
    torch.manual_seed(seed)
    prompts, cots = tensors(examples)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = prompts.shape[0]
    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        loss = model.sft_loss(prompts[idx], cots[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            losses.append(round(float(loss.item()), 4))
    return losses


@torch.no_grad()
def evaluate(model: TinyTransformer, examples: list[Example], *, batch: int = 256,
             greedy: bool = True) -> dict:
    model.eval()
    prompts, _ = tensors(examples)
    n = prompts.shape[0]
    ans_ok = cot_ok = 0
    for start in range(0, n, batch):
        chunk = prompts[start : start + batch]
        cots = model.generate_cot(chunk, n_tokens=5, greedy=greedy).tolist()
        for ex, cot in zip(examples[start : start + batch], cots, strict=True):
            sc = score_solution(ex, cot)
            ans_ok += int(sc["answer_correct"])
            cot_ok += int(sc["cot_correct"])
    return {
        "n": n,
        "answer_acc": ans_ok / n,
        "cot_acc": cot_ok / n,
    }


def make_eval_set(n: int, seed: int, lo: int, hi: int) -> list[Example]:
    return [Example(a=random.Random(f"{seed}-{i}-a").randint(lo, hi),
                    b=random.Random(f"{seed}-{i}-b").randint(lo, hi))
            for i in range(n)]
