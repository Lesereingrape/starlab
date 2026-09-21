"""Tests for the STaR self-improvement loop (fast, invariant-based)."""

from __future__ import annotations

from starlab.data import score_solution, verify_cot
from starlab.model import TinyTransformer
from starlab.star import make_split, sample_verified, star_run


def test_make_split_disjoint():
    seed, pool, ev = make_split(0, 20, 30, 25)
    keys = [{(e.a, e.b) for e in xs} for xs in (seed, pool, ev)]
    assert all(len(k) == n for k, n in zip(keys, (20, 30, 25), strict=True))
    assert not (keys[0] & keys[1])
    assert not (keys[0] & keys[2])
    assert not (keys[1] & keys[2])


def test_sample_verified_only_keeps_correct():
    m = TinyTransformer()
    _, pool, _ = make_split(1, 20, 100, 20)
    kept = sample_verified(m, pool, k=4, temperature=1.0, keep="answer", seed=3)
    for ex, cot in kept:
        assert score_solution(ex, cot)["answer_correct"]


def test_cot_criterion_is_strict_subset():
    m = TinyTransformer()
    _, pool, _ = make_split(2, 20, 200, 20)
    ans = sample_verified(m, pool, k=6, temperature=1.0, keep="answer", seed=5)
    cot = sample_verified(m, pool, k=6, temperature=1.0, keep="cot", seed=5)
    for ex, c in cot:
        assert verify_cot(ex, c)
    # every carry-correct chain is also answer-correct, so cot keeps <= answer keeps
    assert len(cot) <= len(ans)


def test_star_run_curve_structure():
    seed, pool, ev = make_split(0, 60, 120, 120)
    kw = {"rounds": 2, "k": 6, "temperature": 1.0, "keep": "answer",
          "ft_steps": 120, "init_steps": 150, "seed": 0, "verbose": False}
    a = star_run(seed_examples=seed, pool=pool, eval_examples=ev, **kw)
    assert len(a["curve"]) == 3
    assert a["curve"][0]["train_pairs"] == len(seed)
    assert a["final"]["train_pairs"] >= a["curve"][0]["train_pairs"]
    assert set(a) >= {"curve", "final", "delta"}
