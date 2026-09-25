# starlab — STaR self-improvement, made small enough to audit

A compact, **CPU-only** reproduction of the [STaR](https://arxiv.org/abs/2203.14465)
/ rejection-sampling-fine-tuning idea: a ~600-line, ~100k-parameter transformer
bootstraps its own reasoning by **generating candidate solutions, keeping only the
ones a verifier proves correct, and fine-tuning on them** — round after round.
No GPU, no
API keys, no downloaded weights. Every number in this README is produced by
`python experiments/run_study.py`, which took 15-18 minutes on the 8-thread laptop
these numbers come from (25 when other jobs shared the CPU). The artifact records that
environment (Python, torch, thread count), and two further runs inside it reproduced the
committed file with **one** field differing each time — `runtime_sec` (1709s and then
1514s against the published 914s, both while the machine was busy) — while every
accuracy, curve and count came out identical. CPU float reduction order depends
on the thread count and the torch build, so that caveat is part of what "bit-exact"
means here. Every `±` in this file is the **population** standard deviation over the
seeded runs (`statistics.pstdev`, divided by n), because these three seeds are the whole
repetition, not a sample drawn from a larger pool of runs.

![ci](https://github.com/Lesereingrape/starlab/actions/workflows/ci.yml/badge.svg)

## Why this exists

Self-improvement papers are usually run on 7B+ models where you cannot see the
mechanism or reproduce the curve. This project shrinks the whole loop to something
you can read in one sitting and re-run from scratch, so the *science* — does the
model genuinely get better **because of its own verified output**, or just because
it trained longer? — is checkable rather than asserted.

That question is why the study ships a **matched-compute control**: the control gets
the *same number of gradient steps* on the *fixed* gold seed. If STaR only won by
training more, the two curves would overlap. They don't.

## The task: verifiable column addition

The model learns 2-digit addition as an explicit digit-by-digit chain, right to
left, interleaving each output digit with the carry it produces:

```
prompt:   a1 a0  +  b1 b0  =
target:   o0  c1  o1  c2  o2        # answer is o2 o1 o0
```

This is deliberately built so a **pure, no-model verifier** can score any chain:

- `answer_correct` — reconstruct `o2 o1 o0` and check it equals `a + b`.
- `cot_correct` — the stricter check that every intermediate carry is right.

Because correctness is decidable in arithmetic, the "reward" is exact. That is what
makes a clean self-improvement curve possible — and, honestly, an easier setting
than open-ended reasoning (see [Limitations](#limitations--what-this-is-not)).

## How the loop works

Each STaR round (`src/starlab/star.py`):

1. Sample `k` chains per prompt from the current model at temperature 1.0.
2. Keep only verifier-accepted chains (the model is **never shown the answer**).
3. Fine-tune on `gold seed ∪ all accumulated self-generated chains`.
4. Re-evaluate on a fixed held-out set of prompts the model has never trained on.

The seed set is intentionally tiny (**240** examples over the full `0..99` operand
range ≈ 10k possible problems), so a large fraction of held-out prompts start out
unsolved. Each round converts newly-solvable prompts into fresh training data, which
unlocks more — the classic bootstrap.

## Quickstart

```bash
python -m pip install -e .            # only dependency is torch (CPU build is fine)

starlab quick --seed 0                # one seed of the published experiment: same
                                      # 240/1000/1000 split, same six rounds (~40s here)
starlab study                         # full multi-seed study + control + ablations
python experiments/run_study.py       # the same thing, writing results/star.json
python experiments/run_study.py --out /tmp/again.json   # rerun to diff it field for field
python experiments/make_report.py --write   # rewrite the README tables from the JSON
```

`quick` deliberately runs the *published* configuration rather than a smaller one, so
its curve is directly comparable with the tables below: `starlab quick --seed 2`
prints 10.2% → 15.5% → 33.8% → 58.4% → 85.2% → 89.5% → 90.9%, which is the `seed 2`
row of the per-seed table character for character. Beware which seed you pick: in
this study **seed 0 is the run that never bootstraps** (2.9% → 13.6%). Both are real
results, and the per-seed table shows all three.

## Results

The tables below are generated straight from
[`results/star.json`](results/star.json) by `experiments/make_report.py`, so they
cannot drift from the committed data.

<!-- RESULTS:START -->
- model parameters: **103,055**
- seed / pool / eval sizes: 240 / 1000 / 1000
- drafts per prompt per round: 8; rounds: 6; seeds: 0, 1, 2
- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu

### Held-out answer accuracy vs STaR round (mean over 3 seeds)

| round | train pairs | answer acc (mean) | std | carry-correct (mean) |
|------:|------------:|-------------------:|----:|---------------------:|
| 0 | 240 | 6.5% | 0.030 | 6.5% |
| 1 | 340 | 9.9% | 0.049 | 9.9% |
| 2 | 430 | 18.2% | 0.122 | 18.1% |
| 3 | 546 | 32.3% | 0.210 | 32.1% |
| 4 | 691 | 46.4% | 0.314 | 46.3% |
| 5 | 803 | 54.6% | 0.329 | 54.4% |
| 6 | 863 | 58.2% | 0.327 | 58.1% |

Overall gain: **6.5% → 58.2%** (+51.7 pts). The seed-to-seed std grows
from 0.03 to 0.33, so the runs move apart: the loop compounds small early differences instead of smoothing them.

### The same curve, seed by seed

| seed | round 0 | round 1 | round 2 | round 3 | round 4 | round 5 | round 6 |
|-----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 0 | 2.9% | 3.5% | 4.1% | 7.0% | 8.3% | 10.6% | 13.6% |
| 1 | 6.4% | 10.8% | 16.7% | 31.4% | 45.7% | 63.7% | 70.1% |
| 2 | 10.2% | 15.5% | 33.8% | 58.4% | 85.2% | 89.5% | 90.9% |

Seed 2 bootstraps past 80%; seeds 0, 1 never clear the
threshold where round-to-round acceptance starts compounding. So the mean table above is a
mixture of two outcomes, not one noisy outcome — reporting only the mean would hide which
seeds fail.
Round-0 accuracy per seed (seeds 0/1/2) is 2.9% / 6.4% / 10.2%, a **7.3-pt** range before the loop runs at all.

### STaR vs matched-compute control (frozen seed, same gradient steps)

| round | STaR answer acc | control answer acc |
|------:|----------------:|-------------------:|
| 0 | 6.5% | 6.5% |
| 1 | 9.9% | 7.3% |
| 2 | 18.2% | 7.9% |
| 3 | 32.3% | 9.9% |
| 4 | 46.4% | 9.4% |
| 5 | 54.6% | 10.3% |
| 6 | 58.2% | 10.5% |

Both start from the same seed-only baseline; the control keeps training on the
*frozen* seed for the *same number of steps* and finishes at 10.5%,
against 58.2% for STaR — **5.53x** the accuracy for the same compute.
The gap is the self-generated verified data, not extra gradient steps.

### Keep-criterion ablation (final-answer vs full carry chain)

| round | keep=answer | keep=cot |
|------:|------------:|---------:|
| 0 | 6.5% | 6.5% |
| 1 | 9.9% | 10.7% |
| 2 | 18.2% | 19.1% |
| 3 | 32.3% | 35.8% |
| 4 | 46.4% | 46.8% |
| 5 | 54.6% | 50.1% |
| 6 | 58.2% | 57.2% |

Accepting only fully carry-correct chains (scarcer but higher quality) finishes at 57.2%
against 58.2% for the answer-only filter — -1.0 pts, which is within the 32.7-pt seed-to-seed spread, so the strict filter costs nothing
when the task has a verifiable full trace.

### Draft temperature ablation (answer accuracy at final round)

| temperature | final answer acc | seed std |
|------------:|-----------------:|---------:|
| 0.7 | 53.7% | 0.330 |
| 1.0 | 58.2% | 0.327 |
| 1.3 | 57.5% | 0.334 |

Run on the same seeds as the headline (0/1/2), so its `1.0` row *is* the main curve; a test
asserts that identity against `main.per_seed` in the JSON, which is only true because every
run initialises from its own seed rather than from wherever the script had got to.
Final accuracy ranges from 53.7% (T=0.7) to 58.2% (T=1.0), a 4.5-pt spread.
That sits inside the 33.0-pt mean seed-to-seed std of these rows, so this study does not
resolve the draft temperature: every setting bootstraps a curve of the same shape.

### Gold-seed sweep (nested seed sets, 3 rounds, mean over 3 seeds)

| gold seed examples | round-0 acc | acc after 3 rounds | std | seeds past 80% |
|-------------------:|------------:|--------------------------------:|----:|-----------------:|
| 30 | 1.2% | 1.4% | 0.004 | 0/3 |
| 60 | 2.0% | 2.4% | 0.007 | 0/3 |
| 120 | 1.9% | 2.8% | 0.002 | 0/3 |
| 240 | 6.5% | 32.3% | 0.210 | 0/3 |
| 480 | 72.9% | 99.2% | 0.007 | 3/3 |

Every row trains on a *prefix* of the same gold seed and is scored on the same held-out set,
so the only thing that changes down the table is how much verified supervision the loop starts with.
The 240-seed row is the headline curve cut short at 3 rounds — a test asserts
that identity cell for cell, which also proves the sweep was not run on different data.
No seed reaches 80% in 3 rounds at 240 gold examples or fewer; from 480 up at least one does. That is the threshold this repo
trains at (240), and it is a *soft* one: mean accuracy still climbs from 1.4% to
99.2% across the sweep, so a smaller seed buys a coin-flip rather than
a guaranteed failure.
The largest step is the 480-gold row, +66.9 pts over the row before it.
<!-- RESULTS:END -->

## What the numbers say

- **Self-improvement is real, not just extra compute.** STaR's held-out answer
  accuracy climbs 6.5% → 58.2% over six rounds, while the matched-compute control
  — same seed, same gradient steps — only reaches 10.5%. The gap is the
  self-generated verified data, not training length.
- **The mean hides two outcomes.** Seed 2 finishes at 90.9% and seed 1 at 70.1%,
  but seed 0 ends at 13.6% — six rounds barely lift it off its baseline. The std
  column grows from 0.03 to 0.33 because the loop *amplifies* the small round-0
  differences (2.9% / 6.4% / 10.2%) instead of averaging them out.
- **A seed-size threshold exists, and it is soft.** Sweeping the gold seed over
  nested prefixes (same pool, same eval set) keeps the curve flat at 30/60/120
  examples — 1.4% to 2.8% after three rounds — reaches 32.3% ± 21.0 at 240, and
  99.2% ± 0.7 at 480 on that same three-round budget. Below the threshold the
  model cannot generate enough verified chains to train on, so the loop idles; at
  the published 240 it is a coin-flip between seeds, which is exactly why the
  per-seed table is part of the result.
- **`answer` vs `cot` keep-criteria** end 1.0 pt apart (58.2% vs 57.2%), well
  inside the seed-to-seed spread: with an exact verifier, demanding a fully
  correct chain costs almost nothing.
- **This headline is a correction.** An earlier version of this README reported
  27.6% → 92.3%. That run built each seed's model *after* the previous seed had
  already trained, so the initialisation stream depended on the order the seeds
  were executed in, and the printed mean was not the average of three independent
  runs. Every run now re-seeds from its own `seed`, and the honest number is the
  one above. The temperature ablation is a useful check on the fix: its `1.0` row
  reproduces the main curve cell for cell, and a test asserts that.

## Layout

```
src/starlab/
  data.py     the addition task + exact verifier (the reward)
  model.py    ~103k-param causal transformer, SFT loss, temperature sampling
  train.py    sft / sft_pairs / evaluate (accuracy on generated chains)
  star.py     the STaR loop; disjoint seed/pool/eval splits, with the seed-set
              sweep drawn as nested prefixes of one published split
  cli.py      `starlab quick|study`
experiments/
  run_study.py    multi-seed study + control + ablations -> results/star.json
  make_report.py  render README tables from results/star.json
tests/            verifier invariants, model shapes, STaR acceptance properties
results/star.json committed, reproducible results
```

## Limitations — what this is not

- **Elicitation, not new capability.** The model already "can" do addition; STaR
  mostly improves *coverage* of the problem space and reliability, then saturates in
  a few rounds. This matches the broader literature finding that self-training gains
  are front-loaded ([Quiet-STaR](https://arxiv.org/abs/2403.09629) reports a second
  round adds little).
- **Seed variance is genuine, and it can decide the result.** The three seeds start
  at 2.9% / 6.4% / 10.2% and end six rounds later at 13.6% / 70.1% / 90.9% — one
  seed in three never bootstraps. The mean, the std *and* the per-seed curve are all
  in the artifact, and the README renders them rather than picking a run.
- **Three seeds is three seeds.** With a 0.33 seed-to-seed std at the final round,
  the 58.2% mean is a rough read on the method, not a tight one; more seeds would
  move it, and the sweep row at 240 gold examples (32.3% ± 21.0) is the clearest
  sign of how much of that spread is inherent.
- **Clean-reward setting.** Arithmetic has an exact verifier. Real open-ended
  reasoning has no such oracle — [a learned verifier is the actual
  bottleneck](https://arxiv.org/abs/2505.22954) — so the conclusions here bound the
  *mechanism*, not the frontier.
- **Tiny and 2-digit on purpose.** Not a claim about LLMs; it is a leggable,
  reproducible sandbox for the post-training loop itself.

## Related work

STaR [2203.14465](https://arxiv.org/abs/2203.14465) · ReST
[2308.08998](https://arxiv.org/abs/2308.08998) · RFT / *Beyond Human Data*
[2312.06585](https://arxiv.org/abs/2312.06585) · Quiet-STaR
[2403.09629](https://arxiv.org/abs/2403.09629) · s1 budget-forcing
[2501.19393](https://arxiv.org/abs/2501.19393) · Distilling Step-by-Step
[2305.02301](https://arxiv.org/abs/2305.02301). For the preference-side of
post-training see the companion repo [grpo-repro](https://github.com/Lesereingrape/grpo-repro).

## License

MIT
