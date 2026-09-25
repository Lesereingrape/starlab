"""Guard the README's hand-written size claim.

Everything inside the RESULTS markers is byte-pinned against the committed
artifact; the "~N-line" figure in the first paragraph is the number a reader takes
on trust, so it gets checked against ``src/`` too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOLERANCE = 50


def _source_lines() -> int:
    return sum(len(p.read_text(encoding="utf-8").splitlines())
               for p in sorted((ROOT / "src").rglob("*.py")))


def test_readme_line_count_claim_matches_the_source():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"~(\d+)-line", readme)
    assert m, "README no longer states its source size; drop or restore the claim"
    claimed = int(m.group(1))
    actual = _source_lines()
    assert abs(claimed - actual) <= TOLERANCE, (
        f"README says ~{claimed} lines, src/ has {actual}; update the claim")


def test_readme_parameter_claim_matches_the_artifact():
    """The stated parameter count is a measurement, not a vibe.

    It has to track the model the committed artifact was produced with, otherwise
    "~103k" and a 118k model can coexist in the README and the JSON unnoticed.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claimed = {int(m.group(1)) for m in re.finditer(r"~?(\d+)k[- ]param", readme)}
    assert claimed, "the README no longer states a parameter count"
    artifact = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
    actual = artifact["config"]["params"]
    for k in claimed:
        assert abs(k * 1000 - actual) / actual <= 0.05, (
            f"README says ~{k}k parameters, star.json records {actual:,}")


def test_readme_names_the_std_convention_the_tables_use():
    """`+/-` is ambiguous unless the file says which divisor produced it.

    The published spreads are the population standard deviation over seeds, so the
    README has to use that word: a reader who recomputed the other convention would
    land on a different number and conclude the tables were wrong.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search("population[^.]{0,60}standard\\s+deviation", readme), (
        "the README no longer states which standard-deviation convention its "
        "`+/-` columns use")


def test_the_published_wall_clock_is_the_one_the_artifact_records():
    """The rerun note names three runtimes; only the committed one is still checkable.

    This README rounds to whole seconds, so the test rounds the same way rather than
    asking the prose to carry a decimal it does not need.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"published\s+(\d+(?:\.\d+)?)s(?![\d])", readme)
    assert named, "the rerun note no longer names the published runtime"
    assert len(named) == 1, f"the README names the published runtime more than once: {named}"
    artifact = json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))
    assert abs(float(named[0]) - artifact["runtime_sec"]) <= 0.5, (
        f"README says the published run took {named[0]}s, "
        f"results/star.json records {artifact['runtime_sec']}s")


def test_the_documented_rerun_writes_a_relative_scratch_file():
    """`--out /tmp/...` is not one path across shells, so the recipe must not use it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--out /tmp/" not in readme, (
        "the rerun recipe is back to a /tmp path; Git-Bash rewrites it before the script "
        "sees it, so use a relative scratch file")
    assert "--out again-check.json" in readme, (
        "the rerun recipe no longer names the relative scratch file it documents")
