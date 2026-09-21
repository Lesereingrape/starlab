"""Tests for the verifiable arithmetic data layer."""

from __future__ import annotations

import random

from starlab.data import (
    Example,
    parse_answer,
    sample_examples,
    score_solution,
    verify_cot,
)


def test_prompt_and_cot_shapes():
    ex = Example(a=27, b=15)
    assert len(ex.prompt()) == 6
    assert len(ex.cot()) == 5


def test_gold_cot_reconstructs_answer():
    for a, b in [(27, 15), (99, 99), (5, 4), (0, 0), (98, 7)]:
        ex = Example(a=a, b=b)
        assert parse_answer(ex.cot()) == a + b
        assert verify_cot(ex, ex.cot())


def test_carry_chain_is_correct():
    # 68 + 47 -> o0=5 c1=1 o1=1 c2=1 o2=1 (answer 115)
    ex = Example(a=68, b=47)
    from starlab.data import token_to_digit

    digits = [token_to_digit(t) for t in ex.cot()]
    assert digits == [5, 1, 1, 1, 1]
    assert parse_answer(ex.cot()) == 115


def test_wrong_answer_fails_verification():
    ex = Example(a=27, b=15)
    bad = ex.cot()[:]
    bad[4] = bad[4] + 1  # bump the final carry digit -> wrong answer
    assert not verify_cot(ex, bad)
    sc = score_solution(ex, bad)
    assert not sc["answer_correct"]


def test_malformed_answer_is_none():
    assert parse_answer([1, 2, 3]) is None


def test_sample_examples_in_range():
    rng = random.Random(0)
    exs = sample_examples(50, rng, lo=0, hi=99)
    assert len(exs) == 50
    assert all(0 <= e.a <= 99 and 0 <= e.b <= 99 for e in exs)
