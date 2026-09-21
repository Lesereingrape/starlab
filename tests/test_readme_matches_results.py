"""The README results block must be exactly what make_report renders from the committed JSON.

Guards the repo's central honesty claim: every number is machine-generated from
``results/star.json`` and nothing is hand-copied into prose.
"""

from __future__ import annotations

import importlib.util
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
        "README results drift: run `python experiments/make_report.py` and paste the "
        "output into the RESULTS block."
    )
