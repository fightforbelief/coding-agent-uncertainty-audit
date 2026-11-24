# Did the Tests Run? Measuring Fresh Test Execution in Coding-Agent Traces

Yunjie Guo and Hanqing Yang · Independent Researcher, Suzhou, China

[Read the journal manuscript](paper.pdf) · [Download editable submission sources](journal/submission-source.zip) · [Inspect the aggregate checks](replication/README.md)

Of 5,925 recognized Go package successes in the first 15 rounds, 295 (4.98%) explicitly replayed cached results and 1,569 (26.48%) reported no matching tests. Across eight Go standard-library packages, the cold and warm invocations printed the same named test as passing. Only cold and forced invocations entered its body. [Field counts, controlled measurements and reproduction commands](freshness/README.md).

We audited test activity in a frozen Open-SWE-Traces snapshot of 207,489 trajectories. In the prespecified detector-positive audit frame at 15 rounds, the weighted shares labeled as definite commands and confirmed executions were 97.22% and 74.05%. Requiring agreement by both first-round reviewers yielded 93.72% and 65.75%. The labels began as AI-assisted drafts, underwent separate author review according to the project record, and used AI adjudication for disagreements.

Across every itemwise choice of a complete first-round judgment, the command-minus-execution evidence gap ranged from 15.61 to 29.25 percentage points.

The primary exploratory logistic comparison used 20,012 test trajectories from 347 held-out repositories. Adding four test-related features changed average precision by 0.000907 (paired 95% repository-bootstrap interval −0.002150 to 0.004284). A secondary gradient-boosting model gained 0.003888 (0.000539 to 0.007222). All main within-issue command-presence intervals crossed one. The paper reports all 12 prediction cells and accounts for 860 failed refits among 7,000 requested within-issue bootstrap fits.

## Manuscript and evidence

- [Journal source](journal/paper.tex) and [flat source archive](journal/submission-source.zip). Compile the three files together with Tectonic or PDFLaTeX. The November 24 revision and the [cover letter](journal/cover-letter.txt) are prepared for *Empirical Software Engineering*.
- [Review data and verification](replication/README.md). Check the reported audit shares against weighted totals, the primary change in average precision against the archived metrics, and the number of failed refits against the complete stability table.
- [Machine-readable results](output/final_paper_20261103/tables/) for all prediction cells and within-issue analyses.
- [Exploratory protocol](analysis_v2/PROTOCOL_v3.md), [protocol deviations](analysis_v2/DEVIATIONS.md) and [audit exclusion manifest](analysis_v2/audit/EXCLUSION_SET_MANIFEST.json).
- [NVIDIA Open-SWE-Traces, revision c144d9d](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/c144d9d160dcd9f5ebcd9520d2d54111dda8b681). The audit uses the four historical model/scaffold groups from this revision.

## Verification on the research host

From the P8 project root on hanqing-win in PowerShell, use the project's existing environment. The real-shard tests use the raw source mount. The archived research workspace also contains local-only historical fixtures, review returns, model fits and intermediate files.

```powershell
$env:PYTHONPATH = 'src'
$env:P8_DATA_ROOT = (Resolve-Path '..\_p8-v3_4-census-staging\data\data').Path
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe replication\verify_summary.py
```

## Earlier versions

The [October 27 historical study](paper_20261027.pdf) examined a single strict test-failure marker at five rounds. Its ΔAP was 0.0000659. The journal article uses a later exploratory extension of the same snapshot. [November 3 manuscript and tables](output/final_paper_20261103/) and [subsequent 13-page version](https://github.com/fightforbelief/coding-agent-uncertainty-audit/blob/118d72b107a8c1a044d0e563fa8ee89cc4df3ae3/paper.pdf).
