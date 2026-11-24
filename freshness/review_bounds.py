"""Export anonymous grouped sufficient statistics and calculate review-choice endpoints."""
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
STRATA = {"A_r5_positive", "B_r15_positive_r5_known_negative"}


def export(source, dest):
    pairs = defaultdict(dict)
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            version = row["review_version"]
            if row["window"] != "r15" or row["stratum"] not in STRATA or version not in {"reviewer_a","reviewer_b"}:
                continue
            key = (row["review_id"], row["stratum"], row["scaffold"], row["model"], float(row["weight"]))
            if version in pairs[key]:
                raise ValueError("Repeated reviewer judgment")
            pairs[key][version] = (int(row["test_command_status"] == "definite_test_command"),
                                   int(row["execution_status"] == "execution_confirmed"))
    assert len(pairs) == len({k[0] for k in pairs}) == 324
    # Pool across identities and configurations. Only joint binary decisions and total weight leave the host.
    counts = Counter()
    weights = Counter()
    for key, pair in pairs.items():
        status = pair["reviewer_a"] + pair["reviewer_b"]
        counts[status] += 1
        weights[status] += key[-1]
    with dest.open("x", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["command_a","execution_a","command_b","execution_b","memberships","weight"])
        for key in sorted(counts):
            out.writerow([*key,counts[key],format(weights[key],".12g")])


def bounds(path):
    totals = Counter()
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a,b,c,d = [int(r[col]) for col in ("command_a","execution_a","command_b","execution_b")]
            weight = float(r["weight"])
            values = dict(a=a-b,b=c-d,whole_low=min(a-b,c-d),whole_high=max(a-b,c-d),
                          field_low=min(a,c)-max(b,d),field_high=max(a,c)-min(b,d))
            totals.update({k:weight*v for k,v in values.items()})
            totals["weight"] += weight
            totals["n"] += int(r["memberships"])
    assert totals["n"] == 324 and abs(totals["weight"]-44480) < 1e-6
    return {k:100*totals[k]/totals["weight"] for k in ("a","b","whole_low","whole_high","field_low","field_high")}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        export(Path(sys.argv[1]), Path(sys.argv[2]))
    print(bounds(Path(sys.argv[-1]) if len(sys.argv)>1 else ROOT/"paired_review_cells.csv"))
