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

#: (seed, pool, eval) sizes and the round count the published curve was measured
#: with. They live in the package rather than in the study script so the
#: quickstart command cannot quietly drift below them: 240 seed examples is the
#: bootstrap threshold the README documents, and a demo run under it prints a
#: flat curve in front of a table that climbs.
PUBLISHED_SPLIT = (240, 1000, 1000)
PUBLISHED_ROUNDS = 6

#: Gold-seed set sizes the study sweeps to locate the bootstrap threshold. The
#: published 240-seed split is one of them, and the sets are *nested* (see
#: :func:`published_split`), so every sweep row trains on a prefix of the same
#: draw and is scored on the same held-out questions as the headline curve.
PUBLISHED_SEED_SIZES = (30, 60, 120, 240, 480)


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


def new_model(seed: int = 0) -> TinyTransformer:
    """A freshly initialised transformer whose weights depend on ``seed`` alone.

    Parameter init draws from torch's *global* RNG, so a bare ``TinyTransformer()``
    makes one run's result depend on how many models the calling script happened to
    build before it — the same configuration then scores differently depending on
    where in the file it sits. Every study run is born through this function, which
    is what lets the temperature-1.0 ablation row be checked against the main curve.
    """
    torch.manual_seed(seed)
    return TinyTransformer()


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
    model = model or new_model(seed)
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


def make_nested_split(seed: int, seed_sizes, n_pool: int, n_eval: int):
    """Splits for a seed-set sweep: nested seed sets, one shared pool and eval set.

    ``make_split`` draws the seed set *first*, so asking for a different number of
    seed examples also reshuffles the pool and the held-out set — a sweep over seed
    sizes built that way would compare accuracies measured on different questions.
    Here the largest seed set, the pool and the eval slice are drawn once, and each
    requested seed set is a prefix of the next.
    """
    rng = random.Random(seed)
    largest = max(seed_sizes)
    seen: set[tuple[int, int]] = set()
    everything: list[Example] = []
    while len(everything) < largest + n_pool + n_eval:
        ex = Example(a=rng.randint(0, 99), b=rng.randint(0, 99))
        if (ex.a, ex.b) not in seen:
            seen.add((ex.a, ex.b))
            everything.append(ex)
    pool = everything[largest:largest + n_pool]
    eval_ex = everything[largest + n_pool:]
    return {n: everything[:n] for n in seed_sizes}, pool, eval_ex


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


def published_split(seed: int, n_seed: int = PUBLISHED_SPLIT[0]):
    """``(gold_seed, pool, eval)`` for the run the README table reports.

    ``make_split`` would draw the seed set *first*, so changing ``n_seed`` also
    reshuffles the pool and the held-out set — the sweep rows would then measure
    accuracy on different questions than the headline. ``make_nested_split`` keeps
    one shared pool/eval and hands back seed sets that are prefixes of each other,
    which is what lets a test assert that the sweep row at 240 seeds *is* the
    headline curve truncated to the sweep's round count.
    """
    if n_seed not in PUBLISHED_SEED_SIZES:
        raise ValueError(f"n_seed must be one of {PUBLISHED_SEED_SIZES}, got {n_seed}")
    seed_sets, pool, eval_ex = make_nested_split(
        seed, PUBLISHED_SEED_SIZES, PUBLISHED_SPLIT[1], PUBLISHED_SPLIT[2]
    )
    return seed_sets[n_seed], pool, eval_ex
