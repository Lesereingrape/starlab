"""STaR-style self-improvement: rejection-sample reasoning, verify, fine-tune.

The loop is the honest, small-scale version of Self-Taught Reasoner / RFT:

1. Start from a *tiny* gold seed (data-limited — the model cannot yet solve most
   prompts, so there is real headroom to grow).
2. Each round, let the current model attempt every prompt in an unlabeled pool
   several times at nonzero temperature.
3. Keep only chains a pure verifier accepts (the reconstructed digit answer must
   equal a+b). The model is never told the answer — it earns training data.
4. Fine-tune on the gold seed plus every self-generated correct chain, then
   re-evaluate on a fixed held-out set.

Because the pool's hardest prompts are initially out of reach, each round's
accepted chains teach exactly the skills that were just missing, so held-out
accuracy climbs across rounds. That curve — measured on generated arithmetic,
not teacher forcing — is the result this repo is built to show.
"""

from __future__ import annotations

import random

import torch

from .data import Example, score_solution
from .model import TinyTransformer
from .train import evaluate, sft_pairs, tensors


def sample_verified(
    model: TinyTransformer,
    pool: list[Example],
    *,
    k: int,
    temperature: float,
    keep: str,
    seed: int,
    batch: int = 256,
) -> list[tuple[Example, list[int]]]:
    """Draw ``k`` chains per prompt and keep the verifier-accepted ones.

    ``keep`` selects the acceptance criterion: ``"answer"`` (reconstructed sum is
    right) is the standard STaR reward; ``"cot"`` additionally requires every
    intermediate carry to be correct — a stricter, cleaner-but-scarcer signal.
    """
    torch.manual_seed(seed)
    model.eval()
    accepted: dict[tuple[int, int], list[int]] = {}
    for start in range(0, len(pool), batch):
        chunk = pool[start : start + batch]
        prompts, _ = tensors(chunk)
        # replicate each prompt k times so one forward pass samples all drafts
        reps = prompts.repeat_interleave(k, dim=0)
        cots = model.generate_cot(reps, n_tokens=5, greedy=False,
                                  temperature=temperature).tolist()
        for i, cot in enumerate(cots):
            ex = chunk[i // k]
            if ex in accepted:
                continue
            sc = score_solution(ex, cot)
            ok = sc["answer_correct"] if keep == "answer" else sc["cot_correct"]
            if ok:
                accepted[ex] = cot
    return list(accepted.items())


def star_run(
    *,
    seed_examples: list[Example],
    pool: list[Example],
    eval_examples: list[Example],
    rounds: int = 4,
    k: int = 8,
    temperature: float = 1.0,
    keep: str = "answer",
    ft_steps: int = 300,
    lr: float = 3e-3,
    init_steps: int = 400,
    model: TinyTransformer | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the full STaR loop and return per-round held-out metrics.

    Round 0 is the seed-only baseline. Rounds 1..N each add self-generated
    verified chains, warm-started from the previous model. ``accumulated`` grows
    monotonically because accepted chains are cached across rounds.
    """
    model = model or TinyTransformer()
    gold_pairs: list[tuple[Example, list[int]]] = [(ex, ex.cot()) for ex in seed_examples]

    # Round 0: seed-only supervised baseline.
    sft_pairs(model, gold_pairs, steps=init_steps, lr=lr, seed=seed)
    base = evaluate(model, eval_examples)
    curve = [{"round": 0, "train_pairs": len(gold_pairs), "accepted_this_round": 0,
              "answer_acc": base["answer_acc"], "cot_acc": base["cot_acc"]}]
    if verbose:
        print(f"round 0 (seed-only): answer={base['answer_acc']:.3f} "
              f"cot={base['cot_acc']:.3f} train_pairs={len(gold_pairs)}")

    accepted_cache: dict[tuple[int, int], list[int]] = {}
    for r in range(1, rounds + 1):
        fresh = sample_verified(model, pool, k=k, temperature=temperature,
                                keep=keep, seed=seed * 1000 + r)
        for ex, cot in fresh:
            accepted_cache.setdefault(ex, cot)
        train_pairs = gold_pairs + list(accepted_cache.items())
        sft_pairs(model, train_pairs, steps=ft_steps, lr=lr, seed=seed + r)
        m = evaluate(model, eval_examples)
        curve.append({"round": r, "train_pairs": len(train_pairs),
                      "accepted_this_round": len(fresh),
                      "answer_acc": m["answer_acc"], "cot_acc": m["cot_acc"]})
        if verbose:
            print(f"round {r}: answer={m['answer_acc']:.3f} cot={m['cot_acc']:.3f} "
                  f"train_pairs={len(train_pairs)} (+{len(fresh)} this round)")

    return {"curve": curve, "final": curve[-1], "delta": curve[-1]["answer_acc"] - curve[0]["answer_acc"],
            "model": model}


def make_split(seed: int, n_seed: int, n_pool: int, n_eval: int):
    """Disjoint seed / pool / eval splits over the full 0..99 operand range."""
    rng = random.Random(seed)

    def draw(n):
        seen = set()
        out = []
        while len(out) < n:
            ex = Example(a=rng.randint(0, 99), b=rng.randint(0, 99))
            if (ex.a, ex.b) not in seen:
                seen.add((ex.a, ex.b))
                out.append(ex)
        return out

    seed_ex = draw(n_seed)
    seed_keys = {(e.a, e.b) for e in seed_ex}
    pool = []
    while len(pool) < n_pool:
        ex = Example(a=rng.randint(0, 99), b=rng.randint(0, 99))
        key = (ex.a, ex.b)
        if key not in seed_keys and key not in {(e.a, e.b) for e in pool}:
            pool.append(ex)
    eval_keys = seed_keys | {(e.a, e.b) for e in pool}
    eval_ex = []
    while len(eval_ex) < n_eval:
        ex = Example(a=rng.randint(0, 99), b=rng.randint(0, 99))
        if (ex.a, ex.b) not in eval_keys:
            eval_keys.add((ex.a, ex.b))
            eval_ex.append(ex)
    return seed_ex, pool, eval_ex
