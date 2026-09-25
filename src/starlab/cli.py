"""Command line entry point: ``starlab``.

    starlab quick        # one single-seed STaR run on the published split
    starlab study        # full multi-seed study -> results/star.json

``quick`` is a single seed of the experiment the README table averages over, on
the same 240/1000/1000 split and the same six rounds: it used to run 150 seeds
for four rounds, which is below the bootstrap threshold the README itself
documents, so the curve it printed was flat while sitting under a table that
climbed.
"""

from __future__ import annotations

import argparse

from .model import TinyTransformer, count_parameters
from .star import PUBLISHED_ROUNDS, published_split, star_run


def _quick(seed: int, rounds: int) -> None:
    print(f"tiny model params: {count_parameters(TinyTransformer()):,}")
    seed_ex, pool, ev = published_split(seed)
    print(f"seed={seed}  n_seed={len(seed_ex)}  n_pool={len(pool)}  n_eval={len(ev)}")
    star_run(seed_examples=seed_ex, pool=pool, eval_examples=ev,
             rounds=rounds, k=8, temperature=1.0, keep="answer", seed=seed)


def _study(args: argparse.Namespace) -> None:
    # Imported lazily so the heavy study stays out of the light CLI path.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    from run_study import run_study

    run_study(seeds=tuple(args.seeds), rounds=args.rounds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starlab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quick", help="single-seed STaR run on the published split")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--rounds", type=int, default=PUBLISHED_ROUNDS)

    s = sub.add_parser("study", help="full multi-seed study -> results/star.json")
    s.add_argument("--seed", dest="seeds", type=int, action="append")
    s.add_argument("--rounds", type=int, default=PUBLISHED_ROUNDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "quick":
        _quick(args.seed, args.rounds)
    else:
        if not getattr(args, "seeds", None):
            args.seeds = [0, 1, 2]
        _study(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
