"""The README results block must be exactly what make_report renders from the committed JSON.

Guards the repo's central honesty claim: every number is machine-generated from
``results/star.json`` and nothing is hand-copied into prose.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_renderer():
    path = ROOT / "experiments" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(text: str) -> str:
    m = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", text, re.DOTALL)
    assert m, "README is missing the RESULTS:START/END block"
    return m.group(1).strip()


def test_readme_matches_committed_results():
    data = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
    rendered = _load_renderer().build(data).strip()
    readme_block = _block((ROOT / "README.md").read_text(encoding="utf-8"))
    assert readme_block == rendered, (
        "README results drift: run `python experiments/make_report.py --write` to "
        "splice the rendered block back into README.md."
    )


def test_quickstart_runs_the_experiment_the_table_reports():
    """``starlab quick`` must not train a smaller world than the published curve.

    It used to draw 150 seed examples over 4 rounds while the study ran 240 over
    6. 240 is the bootstrap threshold the README documents, so the command the
    README points a reader at sat under its own cliff claim and printed a flat
    curve below a table that climbs. The demo was not buggy; it was measuring an
    unpublished regime and being read as if it were the result.
    """
    data = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
    config = data["config"]
    from starlab import cli
    from starlab.star import PUBLISHED_SPLIT

    assert (config["n_seed"], config["n_pool"], config["n_eval"]) == PUBLISHED_SPLIT
    assert config["rounds"] == cli.PUBLISHED_ROUNDS
    assert config["rounds"] == cli._parser().parse_args(["quick"]).rounds
    source = inspect.getsource(cli._quick)
    assert "lr=" not in source and "temperature=1.0" in source, (
        "quick must not retune what the study fixed, and the table's runs are at "
        "temperature 1.0")
    assert f"k={config['k']}" in source, (
        f"the table was measured at k={config['k']} drafts per prompt; the demo must "
        "draw the same number")
    assert 'keep="answer"' in source, (
        "the headline curve is the answer-only keep criterion, so the demo must not "
        "quietly run the stricter cot filter")


def test_quickstart_draws_the_same_questions_as_the_table():
    """Sizes in the artifact are not enough — the demo must use *this* split builder.

    ``make_split`` reshuffles pool and held-out set whenever the seed size changes,
    so a demo that called it directly could match the published numbers while
    training on different questions. Both paths go through ``published_split``.
    """
    data = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
    config = data["config"]
    from starlab import cli
    from starlab.star import published_split

    assert "published_split(" in inspect.getsource(cli._quick)
    seed_ex, pool, ev = published_split(0)
    assert (len(seed_ex), len(pool), len(ev)) == (config["n_seed"], config["n_pool"],
                                                 config["n_eval"])
    keys = {(e.a, e.b) for e in seed_ex}
    assert not keys & {(e.a, e.b) for e in pool + ev}
