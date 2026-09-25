"""Run the full self-improvement study and write measured results to JSON.

Produces ``results/star.json`` consumed by the README. Everything is CPU-only and
seeded, so re-running reproduces the same numbers *in the recorded environment*.
The study has five parts:

1. Main STaR curve across seeds (accuracy vs round) — the headline result.
2. A *matched-compute* control: keep training on the fixed gold seed for the same
   number of gradient steps. This isolates the real question — does STaR win
   because of self-generated data, or merely because it trained longer?
3. Keep-criterion ablation (verifier on final answer vs the full carry chain).
4. Temperature ablation for the rejection-sampling drafts.
5. Seed-set sweep over nested gold-seed sizes, which is where the README's
   bootstrap-threshold claim comes from.

    python experiments/run_study.py [--out PATH]

``--out`` exists so a second run can be written to a scratch path and diffed field
for field against the committed artifact - which is how the reproducibility
sentence above gets checked rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from starlab.model import TinyTransformer, count_parameters
from starlab.star import (
    PUBLISHED_ROUNDS,
    PUBLISHED_SEED_SIZES,
    PUBLISHED_SPLIT,
    make_nested_split,
    new_model,
    published_split,
    star_run,
)
from starlab.train import evaluate, sft_pairs

#: Rounds the seed-size sweep runs: short enough to afford five sizes times three
#: seeds, long enough that a bootstrapping seed has already taken off.
SWEEP_ROUNDS = 3


def _mean_curve(runs):
    n = min(len(r["curve"]) for r in runs)
    curve = []
    for i in range(n):
        accs = [r["curve"][i]["answer_acc"] for r in runs]
        cots = [r["curve"][i]["cot_acc"] for r in runs]
        tp = [r["curve"][i].get("train_pairs") for r in runs]
        entry = {
            "round": i,
            "answer_mean": round(statistics.fmean(accs), 4),
            "answer_std": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
            "cot_mean": round(statistics.fmean(cots), 4),
        }
        vals = [v for v in tp if v is not None]
        entry["train_pairs_mean"] = round(statistics.fmean(vals), 1) if vals else None
        curve.append(entry)
    return curve


def matched_compute_control(*, seed_examples, eval_examples, rounds, ft_steps,
                            init_steps, lr, seed):
    """Train on the fixed gold seed only, matching STaR's per-round step budget.

    Round 0 uses ``init_steps`` (same as STaR's seed-only baseline); rounds
    1..``rounds`` each add ``ft_steps`` — so control round *r* has seen exactly
    the same number of gradient steps as STaR round *r*, isolating the effect of
    *what data* those steps trained on (frozen seed vs self-generated chains).
    """
    model = new_model(seed)
    gold = [(ex, ex.cot()) for ex in seed_examples]
    curve = []
    sft_pairs(model, gold, steps=init_steps, lr=lr, seed=seed)
    m = evaluate(model, eval_examples)
    curve.append({"round": 0, "answer_acc": m["answer_acc"], "cot_acc": m["cot_acc"]})
    for r in range(1, rounds + 1):
        sft_pairs(model, gold, steps=ft_steps, lr=lr, seed=seed + r)
        m = evaluate(model, eval_examples)
        curve.append({"round": r, "answer_acc": m["answer_acc"], "cot_acc": m["cot_acc"]})
    return curve


def _environment() -> dict:
    """The machine that produced these numbers, recorded next to them.

    Within one environment the study is bit-exact: two runs of this file on this
    machine agree field for field apart from the wall-clock. Across environments
    that is not true — float reduction order over a batch depends on the thread
    count and the torch build — so the artifact names the one it was measured in
    instead of the README claiming a reproducibility it cannot deliver.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def run_study(seeds=(0, 1, 2), rounds=PUBLISHED_ROUNDS, out="results/star.json"):
    t0 = time.time()
    n_seed, n_pool, n_eval = PUBLISHED_SPLIT
    print(f"params/model = {count_parameters(TinyTransformer()):,}")

    print("== main STaR (keep=answer, temp=1.0) ==")
    main_runs = []
    for s in seeds:
        seed_ex, pool, ev = published_split(s)
        main_runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                  rounds=rounds, k=8, temperature=1.0, keep="answer",
                                  seed=s))

    print("== matched-compute control (seed-only) ==")
    control = []
    for s in seeds:
        seed_ex, _, ev = published_split(s)
        control.append(matched_compute_control(seed_examples=seed_ex, eval_examples=ev,
                                               rounds=rounds, ft_steps=300,
                                               init_steps=400, lr=3e-3, seed=s))

    print("== ablation keep=cot ==")
    cot_runs = []
    for s in seeds:
        seed_ex, pool, ev = published_split(s)
        cot_runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                 rounds=rounds, k=8, temperature=1.0, keep="cot", seed=s))

    print("== ablation temperature ==")
    temps = {}
    temp_seeds = seeds
    for temp in (0.7, 1.0, 1.3):
        runs = []
        for s in temp_seeds:
            seed_ex, pool, ev = published_split(s)
            runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                 rounds=rounds, k=8, temperature=temp, keep="answer",
                                 seed=s))
        temps[str(temp)] = _mean_curve(runs)

    print("== seed-set sweep ==")
    # One nested draw per seed: each sweep size is a prefix of the same gold seed,
    # and the pool and held-out set are shared with the headline curve, so the rows
    # differ only in how much verified supervision the loop starts with.
    splits = {s: make_nested_split(s, PUBLISHED_SEED_SIZES, n_pool, n_eval) for s in seeds}
    sweep = []
    for n in PUBLISHED_SEED_SIZES:
        starts, ends = [], []
        for s in seeds:
            seed_sets, pool, ev = splits[s]
            curve = star_run(seed_examples=seed_sets[n], pool=pool, eval_examples=ev,
                             rounds=SWEEP_ROUNDS, k=8, temperature=1.0, keep="answer",
                             seed=s, verbose=False)["curve"]
            starts.append(curve[0]["answer_acc"])
            ends.append(curve[-1]["answer_acc"])
        sweep.append({
            "n_seed": n,
            "rounds": SWEEP_ROUNDS,
            "start_mean": round(statistics.fmean(starts), 4),
            "final_mean": round(statistics.fmean(ends), 4),
            "final_std": round(statistics.pstdev(ends), 4) if len(ends) > 1 else 0.0,
            "per_seed": [round(v, 4) for v in ends],
        })
        print(f"n_seed={n:>4}: {sweep[-1]['start_mean']:.3f} -> {sweep[-1]['final_mean']:.3f} "
              f"+/- {sweep[-1]['final_std']:.3f}")

    out_curve = _mean_curve(main_runs)
    result = {
        "config": {"seeds": list(seeds), "rounds": rounds, "n_seed": n_seed,
                   "n_pool": n_pool, "n_eval": n_eval, "k": 8,
                   "sweep_seed_sizes": list(PUBLISHED_SEED_SIZES),
                   "sweep_rounds": SWEEP_ROUNDS,
                   "params": count_parameters(TinyTransformer())},
        "main": {"curve": out_curve,
                 "start": out_curve[0]["answer_mean"],
                 "start_per_seed": [round(r["curve"][0]["answer_acc"], 4)
                                    for r in main_runs],
                 "end": out_curve[-1]["answer_mean"],
                 "end_per_seed": [round(r["curve"][-1]["answer_acc"], 4)
                                  for r in main_runs],
                 # Per-seed curves let a test check that the temperature ablation at
                 # T=1.0, and the sweep row at the published seed size, really are
                 # this same experiment rather than a look-alike rerun.
                 "per_seed": {str(s): [round(p["answer_acc"], 4) for p in r["curve"]]
                              for s, r in zip(seeds, main_runs, strict=True)},
                 "gain": round(out_curve[-1]["answer_mean"] - out_curve[0]["answer_mean"], 4)},
        "control_matched_compute": {"curve": _mean_curve(
            [{"curve": c} for c in control])},
        "ablation_keep_cot": {"curve": _mean_curve(cot_runs)},
        "ablation_temperature": temps,
        "seed_set_sweep": sweep,
        "environment": _environment(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["main"], indent=2))
    print("control final:", result["control_matched_compute"]["curve"][-1]["answer_mean"])
    print(f"wrote {path} in {result['runtime_sec']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_study")
    parser.add_argument("--out", default="results/star.json",
                        help="where to write the artifact; point it at a scratch path "
                             "to rerun and diff against the committed one")
    args = parser.parse_args()
    run_study(out=args.out)
