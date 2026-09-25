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
