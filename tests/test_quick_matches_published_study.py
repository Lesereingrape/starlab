"""`starlab quick --seed 2` is advertised as printing the published seed-2 row.

Three things have to agree about that curve: what the CLI prints, what the README's table
renders from `results/star.json`, and what the prose recites just under the install
commands. Only the table was pinned, so "character for character" was a promise about two
numbers nobody compared. The cheap guards below cover the documentation and the
configuration; the live run is gated on the artifact's own `environment` fields, because
off that machine a mismatch would come from float reduction order rather than from
anything the README got wrong.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from make_report import BOOTSTRAP_ACC, _pct  # noqa: E402

from starlab.cli import _parser, main  # noqa: E402
from starlab.star import PUBLISHED_ROUNDS, published_split  # noqa: E402

DATA = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
CFG = DATA["config"]
SEED = 2
ROW = [_pct(v) for v in DATA["main"]["per_seed"][str(SEED)]]


_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _prose() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return readme.split("<!-- RESULTS:START -->", 1)[0]


def _handwritten() -> str:
    """Everything the renderer does not write, folded onto one line per paragraph."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stripped = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", "",
                      readme, flags=re.DOTALL)
    return re.sub(r"\s+", " ", stripped)


def test_the_prose_recites_the_published_seed_row():
    m = re.search(r"prints ([0-9.% →]+), which is the `seed (\d+)`", _prose())
    assert m, (
        "the README no longer recites the quick curve next to the seed it quotes; if the "
        "sentence moved, move this guard with it rather than deleting it")
    assert m.group(2) == str(SEED), f"the prose now quotes seed {m.group(2)}, not {SEED}"
    quoted = [x.strip() for x in m.group(1).split("→")]
    assert quoted == ROW, f"the prose says {quoted}, the artifact's seed-{SEED} row is {ROW}"


def test_the_published_row_has_one_point_per_round_of_the_recorded_run():
    assert len(ROW) == CFG["rounds"] + 1, (
        f"seed {SEED} printed {len(ROW)} points for a {CFG['rounds']}-round study")


def test_the_seed_the_prose_warns_about_is_the_lowest_ending_row():
    """"Beware which seed you pick" is advice about one specific row, so pin it to that row."""
    m = re.search(
        r"seed (\d+) is the run that never gets going\*\* \(([\d.]+)% → ([\d.]+)%"
        r"([^)]*)\)",
        _handwritten())
    assert m, "the README no longer names the seed it warns about, or changed its shape"
    stuck = m.group(1)
    row = DATA["main"]["per_seed"][stuck]
    assert (float(m.group(2)), float(m.group(3))) == (
        round(row[0] * 100, 1), round(row[-1] * 100, 1)), (
        f"the prose quotes {m.group(2)}% → {m.group(3)}% for seed {stuck}, "
        f"the table row is {row[0]:.3f} → {row[-1]:.3f}")
    assert "lowest ending" in m.group(4), (
        f"the parenthetical now says {m.group(4)!r}, which no longer claims the minimum")
    ends = {k: v[-1] for k, v in DATA["main"]["per_seed"].items()}
    assert ends[stuck] == min(ends.values()), (
        f"seed {stuck} is called the lowest but the endings are {ends}")


def test_the_limitations_note_counts_the_rows_below_the_bootstrap_threshold():
    """The warning is a count over the same three rows, so check it against them.

    It once named a single failing seed while the table showed two under the threshold,
    which is the kind of claim only a test comparing prose to data can keep honest.
    """
    m = re.search(r"(\w+) of the (\w+) never clear the ([\d.]+)% threshold", _handwritten())
    assert m, "the limitations section no longer counts the seeds that miss the threshold"
    below, total = _WORD_NUM[m.group(1)], _WORD_NUM[m.group(2)]
    ends = [v[-1] for v in DATA["main"]["per_seed"].values()]
    assert float(m.group(3)) == BOOTSTRAP_ACC * 100, (
        f"the prose quotes a {m.group(3)}% threshold, make_report uses {BOOTSTRAP_ACC:.0%}")
    assert (below, total) == (sum(e < BOOTSTRAP_ACC for e in ends), len(ends)), (
        f"prose says {m.group(0)} for endings {ends} against {BOOTSTRAP_ACC:.0%}")


def test_quick_defaults_to_the_published_round_budget():
    """`starlab quick --seed 2` is run without --rounds in the README, so the default is it."""
    args = _parser().parse_args(["quick"])
    assert args.rounds == PUBLISHED_ROUNDS == CFG["rounds"], (
        args.rounds, PUBLISHED_ROUNDS, CFG["rounds"])


def test_quick_uses_the_split_sizes_the_artifact_records():
    seed_ex, pool, ev = published_split(SEED)
    assert (len(seed_ex), len(pool), len(ev)) == (
        CFG["n_seed"], CFG["n_pool"], CFG["n_eval"]), (
        f"quick draws {len(seed_ex)}/{len(pool)}/{len(ev)}, the artifact recorded "
        f"{CFG['n_seed']}/{CFG['n_pool']}/{CFG['n_eval']}")


def _same_environment() -> bool:
    from run_study import _environment
    return _environment() == DATA["environment"]


@pytest.mark.skipif(not _same_environment(),
                    reason="bit-exactness is only promised inside the recorded environment")
def test_the_live_quick_run_prints_the_published_row():
    """Also the README's wall-clock budget: `quick` advertises ~40 s on this machine."""
    started = time.perf_counter()
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["quick", "--seed", str(SEED)]) == 0
    elapsed = time.perf_counter() - started
    printed = [_pct(float(v)) for v in re.findall(r"answer=([\d.]+)", buf.getvalue())]
    assert printed == ROW, f"`quick --seed {SEED}` printed {printed}, the table says {ROW}"
    advertised = re.search(r"\((~(\d+)s here)\)", _prose())
    assert advertised, "the README no longer budgets a quick run in its own fence"
    seconds = int(advertised.group(2))
    assert elapsed < 2 * seconds, (
        f"the README budgets {advertised.group(1)} for one seed and it took {elapsed:.0f}s; "
        "either the configuration grew or the budget is wrong")
