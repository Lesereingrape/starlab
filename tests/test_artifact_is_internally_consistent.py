"""The committed artifact must contradict no other part of itself.

The temperature ablation re-runs the *main* configuration at three draft temperatures,
one of which (1.0) is the headline setting on the headline seeds, so that row has to
equal the main curve cell for cell. That identity is what broke when model
initialisation drew from torch's global RNG in call order: the same configuration
scored differently depending on where in the script the run happened to sit, and
nothing in the tables looked wrong.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _data() -> dict:
    return json.loads((ROOT / "results" / "star.json").read_text(encoding="utf-8"))


def test_temperature_one_row_is_the_main_curve():
    """The T=1.0, keep=answer ablation *is* the headline experiment, so it must agree."""
    data = _data()
    row = data["ablation_temperature"]["1.0"]
    for i, entry in enumerate(data["main"]["curve"]):
        assert abs(entry["answer_mean"] - row[i]["answer_mean"]) < 1e-4, f"round {i} mean"
        assert abs(entry["answer_std"] - row[i]["answer_std"]) < 1e-4, f"round {i} std"


def test_per_seed_values_average_to_the_published_curve():
    data = _data()
    main = data["main"]
    for i, entry in enumerate(main["curve"]):
        across = statistics.fmean(v[i] for v in main["per_seed"].values())
        assert abs(across - entry["answer_mean"]) < 1e-4, f"round {i}"
    assert abs(statistics.fmean(main["start_per_seed"]) - main["start"]) < 1e-4
    assert abs(statistics.fmean(main["end_per_seed"]) - main["end"]) < 1e-4


def test_the_run_records_the_environment_it_is_bit_exact_under():
    env = _data()["environment"]
    assert env["device"] == "cpu"
    assert env["threads"] >= 1
    assert env["torch"].startswith("2.")
    assert env["python"] and env["platform"]


def test_sweep_row_at_the_published_size_is_the_truncated_headline():
    """One sweep row must *be* the headline curve stopped early.

    The sweep is the evidence behind the README's bootstrap-threshold sentence, and
    that sentence is only worth anything if the swept runs share the headline's
    pool and held-out set. Nested splits make that true by construction; this test
    makes it checkable, because a row that drifted to a different draw would no
    longer reproduce the headline number at the same seed and round count.
    """
    data = _data()
    cfg = data["config"]
    assert cfg["sweep_seed_sizes"] == sorted(cfg["sweep_seed_sizes"])
    assert cfg["n_seed"] in cfg["sweep_seed_sizes"]
    rows = data["seed_set_sweep"]
    assert [r["n_seed"] for r in rows] == cfg["sweep_seed_sizes"]
    row = next(r for r in rows if r["n_seed"] == cfg["n_seed"])
    assert row["rounds"] == cfg["sweep_rounds"]
    for seed, n in zip(cfg["seeds"], row["per_seed"], strict=True):
        headline = data["main"]["per_seed"][str(seed)]
        assert len(headline) > cfg["sweep_rounds"]
        assert abs(headline[cfg["sweep_rounds"]] - n) < 1e-4, f"seed {seed}"


def test_sweep_rows_start_weak_and_end_higher():
    """Rows at or below the published seed size must start near chance.

    The 480-gold row is allowed to start high — that is the finding rather than a
    bug: enough verified supervision lifts the round-0 baseline itself, which is
    why the sweep is read as a *change* over rounds and not as an endpoint.
    """
    data = _data()
    published = data["config"]["n_seed"]
    for row in data["seed_set_sweep"]:
        if row["n_seed"] <= published:
            assert row["start_mean"] <= 0.5, (
                f"n_seed={row['n_seed']} starts at {row['start_mean']}, so the sweep "
                "is no longer measuring a bootstrap threshold")
        assert row["final_mean"] >= row["start_mean"], (
            f"n_seed={row['n_seed']} went backwards ({row['start_mean']} -> "
            f"{row['final_mean']}) while the README says accuracy climbs")
