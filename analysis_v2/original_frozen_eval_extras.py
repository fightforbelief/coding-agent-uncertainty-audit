"""Original Prompt 7.1: extra metrics on REPRODUCED historical predictions.

No new model, feature selection or V3 test-set evaluation. Historical AP must
match the frozen result before producing additional descriptive metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def run(predictions: Path, output: Path):
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    frame = pd.read_parquet(predictions)
    if len(frame) != 20118 or frame.trajectory_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ValueError("Historical predictions failed identity/label checks")
    y = frame.label.to_numpy()
    baseline = float(y.mean())
    metrics, bins = [], []
    fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Ideal calibration")
    for name, column, expected in (("structural", "p_structural", .6409601282425617), ("augmented", "p_augmented", .6410260199693942)):
        p = frame[column].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid predicted probabilities")
        ap = float(average_precision_score(y, p))
        if abs(ap - expected) > 1e-9:
            raise ValueError("Historical AP does not match; stop before extra metrics")
        metrics.append(dict(model=name, n=len(y), positive_class="resolved==0", positive_prevalence=baseline,
                            average_precision=ap, roc_auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                            ap_minus_random_baseline=ap-baseline, ap_divided_by_random_baseline=ap/baseline))
        index = np.minimum((p * 10).astype(int), 9)
        points = []
        for number in range(10):
            selected = index == number
            count = int(selected.sum())
            mean_p = float(p[selected].mean()) if count else None
            observed = float(y[selected].mean()) if count else None
            bins.append(dict(model=name, bin=number, lower=number/10, upper=(number+1)/10, n=count,
                             mean_predicted_probability=mean_p, observed_failure_rate=observed))
            if count:
                points.append((mean_p, observed))
        ax.plot(*zip(*points), marker="o", label=name.capitalize())
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted failure probability", ylabel="Observed failure rate",
           title="Historical held-out predictions: 10 equal-width bins")
    ax.legend(frameon=False)
    output.mkdir(parents=True)
    pd.DataFrame(metrics).to_csv(output / "frozen_extra_metrics.csv", index=False)
    pd.DataFrame(bins).to_csv(output / "frozen_calibration_bins.csv", index=False)
    fig.savefig(output / "fig_calibration.png", dpi=180)
    plt.close(fig)
    manifest = dict(status="ORIGINAL PROMPT 7.1 COMPLETE; V3 MODEL ANALYSES NOT RUN", source_prediction_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                    metrics=metrics, new_model_fitted=False, new_v3_test_set_scored=False)
    (output / "FROZEN_EXTRA_METRICS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = ["| Model | AP | AUROC | Brier | Random AP baseline | AP above baseline |", "|---|---:|---:|---:|---:|---:|"]
    rows.extend(f"| {r['model']} | {r['average_precision']:.9f} | {r['roc_auc']:.6f} | {r['brier']:.6f} | {baseline:.6f} | {r['ap_minus_random_baseline']:.6f} |" for r in metrics)
    (output / "07_1_frozen_eval_extras.md").write_text("# 原方案 Prompt 7.1：历史预测补充指标\n\n仅使用本轮精确复现的原主分析预测，未拟合任何新模型。阳性类别为 resolved==0。\n\n" + "\n".join(rows) + "\n\n![10 个等宽分箱的历史校准曲线](fig_calibration.png)\n\n空分箱保留在 CSV 中，均值与观察率为空；绘图只画非空分箱。指标与图是历史研究的描述性补充，不能代替 V3 的独立测量核验。Prompt 7.2 非线性基线需等待人工审计与 V3 协议门槛，不在此提前运行。\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.predictions, args.output), indent=2))
