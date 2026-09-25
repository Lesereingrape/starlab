"""Render README result tables directly from results/star.json.

Keeps the numbers in the README mechanically tied to the committed artifact: run
``python experiments/run_study.py`` then ``python experiments/make_report.py`` and
paste the output. Nothing here is hand-copied from prose.
"""

from __future__ import annotations

import json
from pathlib import Path


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


#: Held-out answer accuracy at which a run counts as having "bootstrapped" — one
#: number shared by every verdict sentence below, so the prose cannot quietly use
#: two different bars.
BOOTSTRAP_ACC = 0.8


def _seeds(names: list[str]) -> str:
    """"Seed 2" / "Seeds 0, 1" — so a one-seed sentence does not read as two."""
    return ("Seed " if len(names) == 1 else "Seeds ") + ", ".join(names)


def build(data: dict) -> str:
    cfg = data["config"]
    main = data["main"]
    curve = main["curve"]
    seeds = ", ".join(str(s) for s in cfg["seeds"])
    n_seeds = len(cfg["seeds"])
    out: list[str] = []

    out.append(f"- model parameters: **{cfg['params']:,}**")
    out.append(f"- seed / pool / eval sizes: {cfg['n_seed']} / {cfg['n_pool']} / {cfg['n_eval']}")
    out.append(f"- drafts per prompt per round: {cfg['k']}; rounds: {cfg['rounds']}; "
               f"seeds: {seeds}")
    env = data["environment"]
    out.append(f"- measured under: Python {env['python']} on {env['platform']}, "
               f"torch {env['torch']}, {env['threads']} CPU threads, {env['device']}")
    out.append("")

    out.append(f"### Held-out answer accuracy vs STaR round (mean over {n_seeds} seeds)\n")
    out.append("| round | train pairs | answer acc (mean) | std | carry-correct (mean) |")
    out.append("|------:|------------:|-------------------:|----:|---------------------:|")
    for row in curve:
        tp = round(row["train_pairs_mean"])
        out.append(f"| {row['round']} | {tp} | {_pct(row['answer_mean'])} | "
                   f"{row['answer_std']:.3f} | {_pct(row['cot_mean'])} |")
    out.append("")
    std_first = curve[0]["answer_std"]
    std_last = curve[-1]["answer_std"]
    converged = std_last < std_first
    out.append(f"Overall gain: **{_pct(main['start'])} → {_pct(main['end'])}** "
               f"(+{main['gain'] * 100:.1f} pts). The seed-to-seed std "
               f"{'shrinks' if converged else 'grows'}")
    out.append(f"from {std_first:.2f} to {std_last:.2f}, so the runs "
               + ("converge on one another."
                  if converged else
                  "move apart: the loop compounds small early differences instead of "
                  "smoothing them."))
    out.append("")

    seed_list = "/".join(str(s) for s in cfg["seeds"])
    per_seed = main["per_seed"]
    out.append("### The same curve, seed by seed\n")
    out.append("| seed | " + " | ".join(f"round {row['round']}" for row in curve) + " |")
    out.append("|-----:|" + "-------:" * len(curve) + "|")
    for s in sorted(per_seed, key=int):
        cells = " | ".join(_pct(v) for v in per_seed[s])
        out.append(f"| {s} | {cells} |")
    out.append("")
    reached = sorted((s for s, v in per_seed.items() if v[-1] >= BOOTSTRAP_ACC), key=int)
    stuck = sorted((s for s, v in per_seed.items() if v[-1] < BOOTSTRAP_ACC), key=int)
    if not stuck:
        out.append(f"Every seed bootstraps past {BOOTSTRAP_ACC:.0%}: the loop is reliable at this "
                   f"size, {len(reached)} for {len(per_seed)}.")
    elif not reached:
        out.append(f"No seed clears {BOOTSTRAP_ACC:.0%} — on this split the pool never yields "
                   "enough verified")
        out.append("chains to lift the model out of its baseline.")
    else:
        verb = "bootstraps" if len(reached) == 1 else "bootstrap"
        verb2 = "clears" if len(stuck) == 1 else "clear"
        out.append(f"{_seeds(reached)} {verb} past {BOOTSTRAP_ACC:.0%}; "
                   f"{_seeds(stuck).lower()} never {verb2} the")
        out.append("threshold where round-to-round acceptance starts compounding. So the mean "
                   "table above is a")
        out.append("mixture of two outcomes, not one noisy outcome — reporting only the mean "
                   "would hide which")
        out.append("seeds fail.")
    out.append(f"Round-0 accuracy per seed (seeds {seed_list}) is "
               f"{' / '.join(_pct(x) for x in main['start_per_seed'])}, a "
               f"**{(max(main['start_per_seed']) - min(main['start_per_seed'])) * 100:.1f}-pt** "
               "range before the loop runs at all.")
    out.append("")

    star = curve
    ctrl = data["control_matched_compute"]["curve"]
    out.append("### STaR vs matched-compute control (frozen seed, same gradient steps)\n")
    out.append("| round | STaR answer acc | control answer acc |")
    out.append("|------:|----------------:|-------------------:|")
    for i in range(min(len(ctrl), len(star))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(ctrl[i]['answer_mean'])} |")
    out.append("")
    ratio = star[-1]["answer_mean"] / ctrl[-1]["answer_mean"]
    out.append("Both start from the same seed-only baseline; the control keeps training on the")
    out.append("*frozen* seed for the *same number of steps* and finishes at "
               f"{_pct(ctrl[-1]['answer_mean'])},")
    out.append(f"against {_pct(star[-1]['answer_mean'])} for STaR — **{ratio:.2f}x** the accuracy for "
               "the same compute.")
    out.append("The gap is the self-generated verified data, not extra gradient steps.")
    out.append("")

    out.append("### Keep-criterion ablation (final-answer vs full carry chain)\n")
    out.append("| round | keep=answer | keep=cot |")
    out.append("|------:|------------:|---------:|")
    cot = data["ablation_keep_cot"]["curve"]
    for i in range(min(len(star), len(cot))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(cot[i]['answer_mean'])} |")
    out.append("")
    cot_last, star_last = cot[-1], star[-1]
    gap_pts = (cot_last["answer_mean"] - star_last["answer_mean"]) * 100
    noise_pts = max(cot_last["answer_std"], star_last["answer_std"]) * 100
    verdict = (f"within the {noise_pts:.1f}-pt seed-to-seed spread, so the strict "
               "filter costs nothing"
               if abs(gap_pts) <= noise_pts else
               f"beyond the {noise_pts:.1f}-pt seed-to-seed spread, so on this data the "
               f"strict filter is a real {'gain' if gap_pts > 0 else 'loss'}")
    out.append("Accepting only fully carry-correct chains (scarcer but higher quality) finishes at "
               f"{_pct(cot_last['answer_mean'])}")
    out.append(f"against {_pct(star_last['answer_mean'])} for the answer-only filter — "
               f"{gap_pts:+.1f} pts, which is {verdict}")
    out.append("when the task has a verifiable full trace.")
    out.append("")

    out.append("### Draft temperature ablation (answer accuracy at final round)\n")
    temps = data["ablation_temperature"]
    out.append("| temperature | final answer acc | seed std |")
    out.append("|------------:|-----------------:|---------:|")
    for t in sorted(temps, key=float):
        last = temps[t][-1]
        out.append(f"| {t} | {_pct(last['answer_mean'])} | {last['answer_std']:.3f} |")
    out.append("")
    subset = "/".join(str(s) for s in cfg["seeds"])
    finals = {t: temps[t][-1] for t in temps}
    hi_t = max(finals, key=lambda t: finals[t]["answer_mean"])
    lo_t = min(finals, key=lambda t: finals[t]["answer_mean"])
    spread = (finals[hi_t]["answer_mean"] - finals[lo_t]["answer_mean"]) * 100
    noise = sum(v["answer_std"] for v in finals.values()) / len(finals) * 100
    out.append(f"Run on the same seeds as the headline ({subset}), so its `1.0` row *is* the main "
               "curve; a test")
    out.append("asserts that identity against `main.per_seed` in the JSON, which is only true "
               "because every")
    out.append("run initialises from its own seed rather than from wherever the script had got to.")
    out.append(f"Final accuracy ranges from {_pct(finals[lo_t]['answer_mean'])} (T={lo_t}) to "
               f"{_pct(finals[hi_t]['answer_mean'])} (T={hi_t}), a {spread:.1f}-pt spread.")
    if noise >= spread:
        out.append(f"That sits inside the {noise:.1f}-pt mean seed-to-seed std of these rows, so "
                   "this study does not")
        out.append("resolve the draft temperature: every setting bootstraps a curve of the same "
                   "shape.")
    else:
        out.append(f"That is wider than the {noise:.1f}-pt mean seed-to-seed std of these rows, so "
                   f"the T={hi_t} row")
        out.append(f"separating from T={lo_t} is more than one unlucky initialisation.")
    out.append("")

    sweep = data["seed_set_sweep"]
    sweep_rounds = cfg["sweep_rounds"]
    out.append(f"### Gold-seed sweep (nested seed sets, {sweep_rounds} rounds, "
               f"mean over {n_seeds} seeds)\n")
    out.append(f"| gold seed examples | round-0 acc | acc after {sweep_rounds} rounds | std | "
               f"seeds past {BOOTSTRAP_ACC:.0%} |")
    out.append("|-------------------:|------------:|--------------------------------:|----:|"
               "-----------------:|")
    for row in sweep:
        n_ok = sum(1 for v in row["per_seed"] if v >= BOOTSTRAP_ACC)
        out.append(f"| {row['n_seed']} | {_pct(row['start_mean'])} | "
                   f"{_pct(row['final_mean'])} | {row['final_std']:.3f} | "
                   f"{n_ok}/{len(row['per_seed'])} |")
    out.append("")
    out.append("Every row trains on a *prefix* of the same gold seed and is scored on the same "
               "held-out set,")
    out.append("so the only thing that changes down the table is how much verified supervision the "
               "loop starts with.")
    out.append(f"The {cfg['n_seed']}-seed row is the headline curve cut short at "
               f"{sweep_rounds} rounds — a test asserts")
    out.append("that identity cell for cell, which also proves the sweep was not run on different "
               "data.")
    below = [r for r in sweep if max(r["per_seed"]) < BOOTSTRAP_ACC]
    above = [r for r in sweep if max(r["per_seed"]) >= BOOTSTRAP_ACC]
    if below and above:
        out.append(f"No seed reaches {BOOTSTRAP_ACC:.0%} in {sweep_rounds} rounds at "
                   f"{max(r['n_seed'] for r in below)} gold examples or fewer; from "
                   f"{min(r['n_seed'] for r in above)} up at least one does. That is the "
                   "threshold this repo")
        out.append(f"trains at ({cfg['n_seed']}), and it is a *soft* one: mean accuracy still "
                   f"climbs from {_pct(sweep[0]['final_mean'])} to")
        out.append(f"{_pct(sweep[-1]['final_mean'])} across the sweep, so a smaller seed buys a "
                   "coin-flip rather than")
        out.append("a guaranteed failure.")
    elif not above:
        out.append(f"No size swept here bootstraps past {BOOTSTRAP_ACC:.0%} in "
                   f"{sweep_rounds} rounds, so this study does not")
        out.append("support a threshold claim — only the graded rise from "
                   f"{_pct(sweep[0]['final_mean'])} to {_pct(sweep[-1]['final_mean'])}.")
    else:
        out.append(f"Every size swept here has at least one seed past {BOOTSTRAP_ACC:.0%} in "
                   f"{sweep_rounds} rounds, so on this split")
        out.append("the effect is graded rather than a cliff: mean final accuracy rises from "
                   f"{_pct(sweep[0]['final_mean'])} to {_pct(sweep[-1]['final_mean'])}.")
    jumps = [(sweep[i + 1]["n_seed"],
              (sweep[i + 1]["final_mean"] - sweep[i]["final_mean"]) * 100)
             for i in range(len(sweep) - 1)]
    if jumps:
        n, d = max(jumps, key=lambda j: abs(j[1]))
        out.append(f"The largest step is the {n}-gold row, {d:+.1f} pts over the row before it.")
    return "\n".join(out)


def _write(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the block into README.md instead of printing it")
    ap.add_argument("--results", default="results/star.json")
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rendered = build(data)
    if args.write:
        _write(Path("README.md"), rendered)
        print("README results block rewritten")
    else:
        print(rendered)
