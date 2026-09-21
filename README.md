# starlab — STaR self-improvement, made small enough to audit

A compact, **CPU-only** reproduction of the [STaR](https://arxiv.org/abs/2203.14465)
/ rejection-sampling-fine-tuning idea: a ~100k-parameter transformer bootstraps
its own reasoning by **generating candidate solutions, keeping only the ones a
verifier proves correct, and fine-tuning on them** — round after round. No GPU, no
API keys, no downloaded weights. Every number in this README is produced by
`python experiments/run_study.py` on a laptop in a few minutes.

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

starlab quick --seed 0 --rounds 4     # one ~60s run, prints the accuracy curve
starlab study                          # full multi-seed study -> results/star.json
python experiments/make_report.py      # rebuild README tables from the JSON
```

## Results

The tables below are generated straight from
[`results/star.json`](results/star.json) by `experiments/make_report.py`, so they
cannot drift from the committed data.

<!-- RESULTS:START -->
- model parameters: **103,055**
- seed / pool / eval sizes: 240 / 1000 / 1000
- drafts per prompt per round: 8; rounds: 6; seeds: 0, 1, 2

### Held-out answer accuracy vs STaR round (mean over 3 seeds)

| round | train pairs | answer acc (mean) | std | carry-correct (mean) |
|------:|------------:|-------------------:|----:|---------------------:|
| 0 | 240 | 27.6% | 0.206 | 27.6% |
| 1 | 604 | 47.1% | 0.281 | 47.0% |
| 2 | 822 | 64.4% | 0.199 | 64.4% |
| 3 | 998 | 81.9% | 0.096 | 81.9% |
| 4 | 1109 | 88.8% | 0.046 | 88.8% |
| 5 | 1140 | 90.7% | 0.029 | 90.7% |
| 6 | 1155 | 92.3% | 0.016 | 92.3% |

Overall gain: **27.6% → 92.3%** (+64.7 pts). Note the seed-to-seed std collapsing
from 0.21 to 0.02: independent runs converge to the same near-ceiling behavior.

### STaR vs matched-compute control (frozen seed, same gradient steps)

| round | STaR answer acc | control answer acc |
|------:|----------------:|-------------------:|
| 0 | 27.6% | 29.2% |
| 1 | 47.1% | 38.3% |
| 2 | 64.4% | 44.7% |
| 3 | 81.9% | 46.9% |
| 4 | 88.8% | 47.6% |
| 5 | 90.7% | 48.2% |
| 6 | 92.3% | 51.8% |

Both start from the same seed-only baseline; the control keeps training on the
*frozen* seed for the *same number of steps* and creeps to ~52%. STaR more than
doubles it. The gap is the self-generated verified data, not extra compute.

### Keep-criterion ablation (final-answer vs full carry chain)

| round | keep=answer | keep=cot |
|------:|------------:|---------:|
| 0 | 27.6% | 29.2% |
| 1 | 47.1% | 56.9% |
| 2 | 64.4% | 78.3% |
| 3 | 81.9% | 88.7% |
| 4 | 88.8% | 92.1% |
| 5 | 90.7% | 92.9% |
| 6 | 92.3% | 93.5% |

Accepting only fully carry-correct chains (scarcer but higher quality) is, if
anything, slightly *better* here — the strict signal is worth its cost when the
task has a verifiable full trace.

### Draft temperature ablation (answer accuracy at final round, 2-seed subset)

| temperature | final answer acc |
|------------:|-----------------:|
| 0.7 | 96.4% |
| 1.0 | 91.5% |
| 1.3 | 93.4% |

Run on a 2-seed subset, so the `1.0` row reads a little differently from the
3-seed main curve. Lower-temperature drafts win once the model is competent (less
diversity needed to find a correct chain), but all three land near ceiling.
<!-- RESULTS:END -->

## What the numbers say

- **Self-improvement is real, not just extra compute.** STaR's held-out answer
  accuracy climbs 27.6% → 92.3% over six rounds, while the matched-compute control
  — same seed, same gradient steps — only creeps to 51.8%. The gap is the
  self-generated verified data, not training length.
- **A bootstrap threshold exists.** Below roughly 240 seed examples the model never
  learns the carry algorithm, so it self-generates almost nothing correct and STaR
  stays flat; at/above it the loop takes off. That cliff — not a smooth dial — is an
  honest, reproducible property of STaR-style methods.
- **`answer` vs `cot` keep-criteria** track each other closely here (a correct sum
  almost always has a correct chain in base-10 addition), but `cot` is scarcer.

## Layout

```
src/starlab/
  data.py     the addition task + exact verifier (the reward)
  model.py    ~103k-param causal transformer, SFT loss, temperature sampling
  train.py    sft / sft_pairs / evaluate (accuracy on generated chains)
  star.py     the STaR loop and disjoint seed/pool/eval splits
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
- **Seed variance is genuine.** Different seeds start at very different baseline
  accuracy (e.g. ~12% vs ~57% at 240 seeds); we report mean **and** std and do not
  cherry-pick the prettiest run.
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
