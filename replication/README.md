# Verify the manuscript from aggregate results

From the repository root run `python replication/verify_summary.py`. This check uses the Python standard library.

- **Measurement.** Each row in [the audit table](r15_weighted_totals.csv) summarizes the same 324 memberships in the 15-round detector-positive frame, combining strata A and B. Divide each weighted category total by the full weight total. A and B denote the two individual reviews where they appear as reviewer names. The agreement-only coding retains fieldwise disagreement as uncertainty. The final coding uses the declared AI adjudication. An unconfirmed execution can reflect missing or inconclusive evidence.
- **Prediction.** The script subtracts the two primary average precisions in [the checkpoint table](../output/final_paper_20261103/tables/checkpoint_metrics.csv) and checks the recorded interval. The authors' workspace contains the 20,012 scored trajectories from 347 repositories. During closeout, an independent calculation on those saved scores reproduced the point estimates and both endpoints from 1,000 paired repository draws with seed 20261092. The aggregate table alone supports verification of the subtraction and transcription of the interval.
- **Stability.** The script checks the denominator of every [within-issue refit cell](../output/final_paper_20261103/tables/table_stability.csv) and sums the failed fits.
- **Figures.** Run `python replication/draw_figures.py` with ReportLab installed to regenerate both vector figures from the published CSVs.

NVIDIA generated the historical trajectories. The [pinned dataset card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/c144d9d160dcd9f5ebcd9520d2d54111dda8b681/README.md) describes the source and [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/). The audit labels and prediction summaries are derived results.
