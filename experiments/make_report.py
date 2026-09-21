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
    out: list[str] = []

    out.append(f"- model parameters: **{cfg['params']:,}**")
    out.append(f"- seed / pool / eval sizes: {cfg['n_seed']} / {cfg['n_pool']} / {cfg['n_eval']}")
    out.append(f"- drafts per prompt per round: {cfg['k']}; rounds: {cfg['rounds']}; "
               f"seeds: {cfg['seeds']}")
    out.append("")

    out.append("### Held-out answer accuracy vs STaR round (mean over seeds)\n")
    out.append("| round | train pairs | answer acc (mean) | std | carry-correct (mean) |")
    out.append("|------:|------------:|--------------------:|----:|---------------------:|")
    for row in data["main"]["curve"]:
        tp = row["train_pairs_mean"]
        out.append(f"| {row['round']} | {tp} | {_pct(row['answer_mean'])} | "
                   f"{row['answer_std']:.3f} | {_pct(row['cot_mean'])} |")
    out.append("")
    out.append(f"Overall gain: **{_pct(data['main']['start'])} → "
               f"{_pct(data['main']['end'])}** (+{data['main']['gain'] * 100:.1f} pts).")
    out.append("")

    out.append("### STaR vs matched-compute control (seed-only, same gradient steps)\n")
    ctrl = data["control_matched_compute"]["curve"]
    star = data["main"]["curve"]
    out.append("| round | STaR answer acc | control answer acc |")
    out.append("|------:|----------------:|-------------------:|")
    for i in range(min(len(ctrl), len(star))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(ctrl[i]['answer_mean'])} |")
    out.append("")

    out.append("### Keep-criterion ablation (answer vs full carry chain)\n")
    out.append("| round | keep=answer | keep=cot |")
    out.append("|------:|------------:|---------:|")
    cot = data["ablation_keep_cot"]["curve"]
    for i in range(min(len(star), len(cot))):
        out.append(f"| {i} | {_pct(star[i]['answer_mean'])} | {_pct(cot[i]['answer_mean'])} |")
    out.append("")

    out.append("### Draft temperature ablation (answer accuracy at final round)\n")
    temps = data["ablation_temperature"]
    out.append("| temperature | final answer acc |")
    out.append("|------------:|-----------------:|")
    for t in sorted(temps, key=float):
        out.append(f"| {t} | {_pct(temps[t][-1]['answer_mean'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    data = json.loads(Path("results/star.json").read_text())
    print(build(data))
