"""Check manuscript quantities that can be reconstructed from aggregate tables."""
import csv
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output/final_paper_20261103/tables"


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    expected = {
        "reviewer_a": (93.72, 70.34), "reviewer_b": (97.22, 75.74),
        "agreement_only": (93.72, 65.75), "reconciled": (97.22, 74.05),
    }
    audit = records(ROOT / "replication/r15_weighted_totals.csv")
    assert len(audit) == 4 and {r["version"] for r in audit} == set(expected)
    for row in audit:
        assert int(row["sample_memberships"]) == 324
        total = float(row["weight_total"])
        command = 100 * float(row["weight_definite_command"]) / total
        execution = 100 * float(row["weight_confirmed_execution"]) / total
        assert (round(command, 2), round(execution, 2)) == expected[row["version"]]
        print(f'{row["version"]}: {command:.2f}% definite commands, {execution:.2f}% confirmed execution')

    checkpoints = records(TABLES / "checkpoint_metrics.csv")
    primary = {r["model"]: r for r in checkpoints
               if (r["kind"], r["population"], r["k"]) == ("logistic", "main", "15")}
    augmented = primary["M2"]
    assert int(augmented["n_test"]) == 20012 and augmented["primary_exploratory_endpoint"] == "True"
    delta = float(augmented["ap"]) - float(primary["M0"]["ap"])
    assert isclose(delta, float(augmented["delta_ap"]), abs_tol=1e-14)
    limits = [float(augmented[k]) for k in ("delta_ap_low", "delta_ap_high")]
    assert [round(v, 6) for v in (delta, *limits)] == [0.000907, -0.002150, 0.004284]
    print(f"Primary AP difference {delta:.9f}; archived interval [{limits[0]:.9f}, {limits[1]:.9f}]")

    stability = records(TABLES / "table_stability.csv")
    assert len(stability) == 14
    assert all(int(r["Valid refits"]) + int(r["Failed refits"]) == 500 for r in stability)
    failed = sum(int(r["Failed refits"]) for r in stability)
    assert failed == 860
    print(f"{failed} failed within-issue bootstrap refits out of {500 * len(stability)}")


if __name__ == "__main__":
    verify()
