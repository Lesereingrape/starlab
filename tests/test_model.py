"""Tests for the tiny transformer and supervised training primitives."""

from __future__ import annotations

import torch

from starlab.data import Example
from starlab.model import TinyTransformer, count_parameters
from starlab.train import evaluate, make_eval_set, sft


def test_param_count_small():
    n = count_parameters(TinyTransformer())
    assert 50_000 < n < 200_000


def test_forward_shape():
    m = TinyTransformer()
    ids = torch.tensor([Example(12, 34).prompt()])
    logits = m(ids)
    assert logits.shape == (1, 6, 15)


def test_generate_length_and_greedy_determinism():
    m = TinyTransformer().eval()
    prompts = torch.tensor([Example(12, 34).prompt(), Example(5, 6).prompt()])
    a = m.generate_cot(prompts, n_tokens=5, greedy=True)
    b = m.generate_cot(prompts, n_tokens=5, greedy=True)
    assert a.shape == (2, 5)
    assert torch.equal(a, b)


def test_sft_reduces_loss_on_tiny_set():
    exs = [Example(a, b) for a, b in [(12, 5), (30, 4), (22, 11), (7, 3)]]
    m = TinyTransformer()
    prompts, cots = ([e.prompt() for e in exs], [e.cot() for e in exs])
    p = torch.tensor(prompts)
    c = torch.tensor(cots)
    start = float(m.sft_loss(p, c).detach())
    sft(m, exs, steps=200, batch=8, lr=5e-3, seed=0)
    end = float(m.sft_loss(p, c).detach())
    assert end < start


def test_evaluate_returns_valid_rates():
    m = TinyTransformer()
    ev = make_eval_set(16, seed=1, lo=0, hi=99)
    r = evaluate(m, ev)
    assert r["n"] == 16
    assert 0.0 <= r["answer_acc"] <= 1.0
