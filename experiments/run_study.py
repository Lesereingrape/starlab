"""Run the full self-improvement study and write measured results to JSON.

Produces ``results/star.json`` consumed by the README. Everything is CPU-only and
seeded, so re-running reproduces the same numbers. The study has four parts:

1. Main STaR curve across seeds (accuracy vs round) — the headline result.
2. A *matched-compute* control: keep training on the fixed gold seed for the same
   number of gradient steps. This isolates the real question — does STaR win
   because of self-generated data, or merely because it trained longer?
3. Keep-criterion ablation (verifier on final answer vs the full carry chain).
4. Temperature ablation for the rejection-sampling drafts.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from starlab.model import TinyTransformer, count_parameters
from starlab.star import make_split, star_run
from starlab.train import evaluate, sft_pairs


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
    model = TinyTransformer()
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


def run_study(seeds=(0, 1, 2), rounds=6, out="results/star.json"):
    t0 = time.time()
    n_seed, n_pool, n_eval = 240, 1000, 1000
    print(f"params/model = {count_parameters(TinyTransformer()):,}")

    print("== main STaR (keep=answer, temp=1.0) ==")
    main_runs = []
    for s in seeds:
        seed_ex, pool, ev = make_split(s, n_seed, n_pool, n_eval)
        main_runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                  rounds=rounds, k=8, temperature=1.0, keep="answer",
                                  seed=s))

    print("== matched-compute control (seed-only) ==")
    control = []
    for s in seeds:
        seed_ex, _, ev = make_split(s, n_seed, n_pool, n_eval)
        control.append(matched_compute_control(seed_examples=seed_ex, eval_examples=ev,
                                               rounds=rounds, ft_steps=300,
                                               init_steps=400, lr=3e-3, seed=s))

    print("== ablation keep=cot ==")
    cot_runs = []
    for s in seeds:
        seed_ex, pool, ev = make_split(s, n_seed, n_pool, n_eval)
        cot_runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                 rounds=rounds, k=8, temperature=1.0, keep="cot", seed=s))

    print("== ablation temperature ==")
    temps = {}
    temp_seeds = seeds[:2]
    for temp in (0.7, 1.0, 1.3):
        runs = []
        for s in temp_seeds:
            seed_ex, pool, ev = make_split(s, n_seed, n_pool, n_eval)
            runs.append(star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
                                 rounds=rounds, k=8, temperature=temp, keep="answer",
                                 seed=s))
        temps[str(temp)] = _mean_curve(runs)

    out_curve = _mean_curve(main_runs)
    result = {
        "config": {"seeds": list(seeds), "rounds": rounds, "n_seed": n_seed,
                   "n_pool": n_pool, "n_eval": n_eval, "k": 8,
                   "params": count_parameters(TinyTransformer())},
        "main": {"curve": out_curve,
                 "start": out_curve[0]["answer_mean"],
                 "end": out_curve[-1]["answer_mean"],
                 "gain": round(out_curve[-1]["answer_mean"] - out_curve[0]["answer_mean"], 4)},
        "control_matched_compute": {"curve": _mean_curve(
            [{"curve": c} for c in control])},
        "ablation_keep_cot": {"curve": _mean_curve(cot_runs)},
        "ablation_temperature": temps,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["main"], indent=2))
    print("control final:", result["control_matched_compute"]["curve"][-1]["answer_mean"])
    print(f"wrote {path} in {result['runtime_sec']}s")


if __name__ == "__main__":
    run_study()
