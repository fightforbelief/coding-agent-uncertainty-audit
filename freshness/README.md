# Fresh execution evidence

The field endpoint counts printed Go package records in direct tool observations. A duration without a no-match annotation is compatible with a fresh matching test. Body entry is measured independently only in the controlled suites. The [protocol](protocol.md) records the original rules, the tab-separator correction and the post hoc analyses.

| Artifact | Unit |
|---|---|
| census.json | Pooled and configuration-specific counts at rounds 15 and 30. Omitted count keys are zero. Repository shares give each represented repository equal weight. Deletion extrema omit each represented repository in turn. |
| shards.json | Source revision and SHA-256 hashes of 84 public Parquet files |
| controls.csv | One row for each of 12 controlled runner conditions |
| stdlib_controls.csv | Cold, warm and forced runs of one existing upstream test in each of eight packages |
| paired_review_cells.csv | Pooled count and summed design weight by joint pair of binary judgments in the 324-member R15 positive frame |

From the repository root, check the archived outputs with Python 3.12.

```powershell
.\.venv\Scripts\python.exe freshness\verify.py
.\.venv\Scripts\python.exe replication\verify_summary.py
```

To reproduce from source, use the locked Python dependencies and Go 1.25.4 with its standard-library sources. Download the [pinned Open-SWE-Traces data](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/c144d9d160dcd9f5ebcd9520d2d54111dda8b681) with its original directory structure. Pass the parent of its `data` directory. The scripts create new output files, so select unused destination names. No trajectory command is executed.

```powershell
.\.venv\Scripts\python.exe freshness\census.py C:\datasets\Open-SWE-Traces freshness\shards.json census-recomputed.json --workers 6
.\.venv\Scripts\python.exe freshness\controls.py --go go --output controls-recomputed.csv
.\.venv\Scripts\python.exe freshness\stdlib_controls.py --go go --output stdlib-recomputed.csv
.\.venv\Scripts\python.exe freshness\review_bounds.py
```

The census verifies each shard before reading observations. Compare the numeric `cells` across platforms. File hashes of the executing script and manifest also record the original run's physical line endings. The controlled scripts use an isolated cache and a body-entry counter outside the test source directory. For the standard-library suite, an overlay redirects a source file to an instrumented temporary copy. The distribution on disk stays intact. Reviewer-choice endpoints can be recomputed from the pooled cells without individual reviews.
