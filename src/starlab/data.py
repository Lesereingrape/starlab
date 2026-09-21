"""Grade-school addition as a verifiable chain-of-thought task.

The model is shown two 2-digit operands and must emit the digit-by-digit
column addition, right-to-left, interleaving each result digit with the carry
it produces:

    prompt:  a1 a0 + b1 b0 =
    target:  o0 c1 o1 c2 o2

where (units column) a0+b0 = o0 + 10*c1, (tens column) a1+b1+c1 = o1 + 10*c2,
and o2 = c2 is the final carry. Reading o2 o1 o0 gives the answer.

The whole point is that the *answer* (o2 o1 o0) can be reconstructed and
checked exactly against a+b without ever trusting the model's own arithmetic —
this is the verifiable reward that powers the self-improvement loop in
``star.py``. Difficulty is controlled by how large a+b is allowed to get, so we
can train on easy sums and probe generalization to carry-heavy ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Token ids: specials then the ten digits.
PAD, BOS, EOS, PLUS, EQ = 0, 1, 2, 3, 4
DIGIT0 = 5
VOCAB = ["<pad>", "<bos>", "<eos>", "+", "="] + [str(d) for d in range(10)]
NVOCAB = len(VOCAB)


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_to_digit(tok: int) -> int:
    return tok - DIGIT0


@dataclass(frozen=True)
class Example:
    a: int
    b: int

    @property
    def target(self) -> int:
        return self.a + self.b

    def prompt(self) -> list[int]:
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        return [digit_token(a1), digit_token(a0), PLUS,
                digit_token(b1), digit_token(b0), EQ]

    def cot(self) -> list[int]:
        """Ground-truth o0 c1 o1 c2 o2 (5 tokens, no EOS)."""
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        s0 = a0 + b0
        o0, c1 = s0 % 10, s0 // 10
        s1 = a1 + b1 + c1
        o1, c2 = s1 % 10, s1 // 10
        o2 = c2
        return [digit_token(x) for x in (o0, c1, o1, c2, o2)]


def sample_examples(n: int, rng: random.Random, lo: int, hi: int) -> list[Example]:
    """Draw ``n`` examples with a, b in [lo, hi] (both 0..99)."""
    hi = min(hi, 99)
    lo = max(lo, 0)
    return [Example(a=rng.randint(lo, hi), b=rng.randint(lo, hi)) for _ in range(n)]


def parse_answer(cot_tokens: list[int]) -> int | None:
    """Reconstruct o2 o1 o0 from a generated 5-token CoT, or None if malformed."""
    if len(cot_tokens) < 5:
        return None
    if any(not (DIGIT0 <= t <= DIGIT0 + 9) for t in cot_tokens[:5]):
        return None
    o0, _c1, o1, _c2, o2 = (token_to_digit(t) for t in cot_tokens[:5])
    return 100 * o2 + 10 * o1 + o0


def verify_cot(ex: Example, cot_tokens: list[int]) -> bool:
    """True only if every emitted digit AND carry is internally consistent."""
    gold = ex.cot()
    return len(cot_tokens) >= 5 and cot_tokens[:5] == gold


def score_solution(ex: Example, cot_tokens: list[int]) -> dict:
    """Verifiable scorecard for one generated solution.

    - ``answer_correct`` : reconstructed sum == a+b (the STaR/RFT reward).
    - ``cot_correct``    : the full carry chain is right (stricter).
    """
    ans = parse_answer(cot_tokens)
    return {
        "answer_correct": ans is not None and ans == ex.target,
        "cot_correct": verify_cot(ex, cot_tokens),
        "pred_sum": ans,
        "true_sum": ex.target,
    }


def render(ex: Example, cot_tokens: list[int]) -> str:
    toks = ex.prompt() + list(cot_tokens)
    return " ".join(VOCAB[t] for t in toks)
