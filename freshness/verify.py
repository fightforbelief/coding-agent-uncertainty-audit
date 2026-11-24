"""Check the published denominators, reviewer extrema and runner controls."""
import csv
import json
import argparse
from pathlib import Path

from review_bounds import bounds

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def verify(census):
    result = json.loads(census.read_text())
    assert result["trajectories_scanned"] == 207489 and result["shards_verified"] == 84
    assert len(result["cells"]) == 10
    for k, (cache, no_match, timed) in {15: (295, 1569, 4061), 30: (5465, 2643, 27163)}.items():
        at_k = [r for r in result["cells"] if r["checkpoint"] == k]
        total, = [r for r in at_k if r["group"] == "all"]
        groups = [r for r in at_k if r["group"] != "all"]
        assert total["go_trajectories"] == 46834
        for field, value in (("cached", cache), ("no_match", no_match), ("timed", timed)):
            assert total[field] == value == sum(r.get(field, 0) for r in groups)
            assert total.get("tab_"+field, 0) <= value
        n = cache + no_match + timed
        share = cache/n
        assert total["delete_one_repo_min"] <= share <= total["delete_one_repo_max"]
        print(f"Round {k}: {cache}/{n} cached, {no_match} no match, {timed/n:.2%} ordinary timed")
    extremes = bounds(HERE / "paired_review_cells.csv")
    assert [round(extremes[k], 2) for k in ("whole_low","whole_high","field_low","field_high")] == [
        15.61, 29.25, 13.39, 31.47]
    print("Reviewer-choice endpoints verified")
    control_rows = rows("controls.csv")
    controls = {(r["runner"],r["condition"]): r for r in control_rows}
    assert len(controls) == len(control_rows) == 12
    for runner, a, b in (("go","cold","warm"), ("pytest","failure","setup_error"),
                         ("unittest","pass","skip")):
        x, y = controls[runner,a], controls[runner,b]
        assert (x["body_entries"], y["body_entries"]) == ("1","0")
        assert all(x[key] == y[key] for key in ("command","exit_code","old_parser"))
    std = rows("stdlib_controls.csv")
    assert len(std) == 24 and len({r["package"] for r in std}) == 8
    assert len({(r["package"],r["condition"]) for r in std}) == 24
    for r in std:
        assert r["condition"] in {"cold","warm","forced"}
        assert (r["exit_code"], r["verbose_case_pass"]) == ("0","1")
        assert int(r["body_entries"]) == (r["condition"] != "warm")
        assert int(r["cache_marker"]) == (r["condition"] == "warm")
    print("12 fixture conditions and 24 upstream test executions or replays verified")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("census", type=Path, nargs="?", default=HERE / "census.json")
    verify(ap.parse_args().census)
