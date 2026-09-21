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
    out.append("")

    out.append(f"### Held-out answer accuracy vs STaR round (mean over {n_seeds} seeds)\n")
    out.append("| round | train pairs | answer acc (mean) | std | carry-correct (mean) |")
    out.append("|------:|------------:|-------------------:|----:|---------------------:|")
    for row in curve:
        tp = round(row["train_pairs_mean"])
        out.append(f"| {row['round']} | {tp} | {_pct(row['answer_mean'])} | "
                   f"{row['answer_std']:.3f} | {_pct(row['cot_mean'])} |")
    out.append("")
    std_first = f"{curve[0]['answer_std']:.2f}"
    std_last = f"{curve[-1]['answer_std']:.2f}"
    out.append(f"Overall gain: **{_pct(main['start'])} → {_pct(main['end'])}** "
               f"(+{main['gain'] * 100:.1f} pts). Note the seed-to-seed std collapsing")
    out.append(f"from {std_first} to {std_last}: independent runs converge to the same "
               "near-ceiling behavior.")
    out.append("")

    star = curve
    ctrl = data["control_matched_compute"]["curve"]
    ctrl_end = f"{round(100 * ctrl[-1]['answer_mean'])}"
    out.append("### STaR vs matched-compute control (frozen seed, same gradient steps)\n")
    out.append("| round | STaR answer acc | control answer acc |")
    out.append("|------:|----------------:|-------------------:|")
    for i in range(min(len(ctrl), len(star))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(ctrl[i]['answer_mean'])} |")
    out.append("")
    out.append("Both start from the same seed-only baseline; the control keeps training on the")
    out.append("*frozen* seed for the *same number of steps* and creeps to ~"
               f"{ctrl_end}%. STaR more than")
    out.append("doubles it. The gap is the self-generated verified data, not extra compute.")
    out.append("")

    out.append("### Keep-criterion ablation (final-answer vs full carry chain)\n")
    out.append("| round | keep=answer | keep=cot |")
    out.append("|------:|------------:|---------:|")
    cot = data["ablation_keep_cot"]["curve"]
    for i in range(min(len(star), len(cot))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(cot[i]['answer_mean'])} |")
    out.append("")
    out.append("Accepting only fully carry-correct chains (scarcer but higher quality) is, if")
    out.append("anything, slightly *better* here — the strict signal is worth its cost when the")
    out.append("task has a verifiable full trace.")
    out.append("")

    out.append("### Draft temperature ablation (answer accuracy at final round, 2-seed subset)\n")
    temps = data["ablation_temperature"]
    out.append("| temperature | final answer acc |")
    out.append("|------------:|-----------------:|")
    for t in sorted(temps, key=float):
        out.append(f"| {t} | {_pct(temps[t][-1]['answer_mean'])} |")
    out.append("")
    out.append("Run on a 2-seed subset, so the `1.0` row reads a little differently from the")
    out.append("3-seed main curve. Lower-temperature drafts win once the model is competent (less")
    out.append("diversity needed to find a correct chain), but all three land near ceiling.")
    return "\n".join(out)


if __name__ == "__main__":
    data = json.loads(Path("results/star.json").read_text())
    print(build(data))
