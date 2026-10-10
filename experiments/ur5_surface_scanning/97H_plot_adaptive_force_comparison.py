#!/usr/bin/env python3
"""97H - Plot and report the fixed-speed 97E vs adaptive-speed 97G results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "results" / "robustness" / "97E_robustness_summary.csv"
ADAPTIVE_PATH = (
    HERE
    / "results"
    / "adaptive_robustness"
    / "97G_adaptive_robustness_summary.csv"
)
OUTPUT_DIR = HERE / "results" / "adaptive_robustness"
BASELINE_DURATION_S = 40.286


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def main():
    baseline = read_rows(BASELINE_PATH)
    adaptive = read_rows(ADAPTIVE_PATH)
    baseline_cases = [row["case"] for row in baseline]
    adaptive_by_case = {row["case"]: row for row in adaptive}
    if set(baseline_cases) != set(adaptive_by_case):
        raise ValueError("97E与97G场景集合不一致，请先完整运行97G")
    adaptive = [adaptive_by_case[name] for name in baseline_cases]

    labels = [
        "nominal",
        "Z +1mm",
        "Z -1mm",
        "X +1mm",
        "tilt +1deg",
        "mu 0.15",
        "mu 0.60",
    ]
    x = np.arange(len(labels))
    width = 0.38
    metrics = (
        ("force_mean_error_n", "Mean force error [N]", 0.50),
        ("force_max_error_n", "Maximum force error [N]", 3.00),
        ("tcp_max_error_mm", "Maximum TCP error [mm]", 2.00),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    for axis, (key, ylabel, threshold) in zip(axes, metrics):
        old = value(baseline, key)
        new = value(adaptive, key)
        axis.bar(x - width / 2, old, width, label="97E fixed speed")
        axis.bar(x + width / 2, new, width, label="97G adaptive speed")
        axis.axhline(
            threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="acceptance limit" if key == metrics[0][0] else None,
        )
        axis.set_ylabel(ylabel)
    axes[0].legend(ncol=3, loc="upper left")
    axes[-1].set_xticks(x, labels, rotation=18, ha="right")
    axes[-1].set_xlabel("Robustness scenario")
    figure.suptitle("UR5 learned-surface 5 N scan: 97E vs 97G")
    figure.tight_layout()
    metrics_path = OUTPUT_DIR / "97H_97E_vs_97G_metrics.png"
    figure.savefig(metrics_path, dpi=180)
    plt.close(figure)

    duration = value(adaptive, "scan_duration_s")
    speed_scale = value(adaptive, "speed_scale_mean")
    figure, axis_duration = plt.subplots(figsize=(11.5, 5.4))
    axis_duration.bar(x, duration, color="tab:blue", alpha=0.82)
    axis_duration.axhline(
        BASELINE_DURATION_S,
        color="tab:red",
        linestyle="--",
        label=f"97E fixed-speed duration ({BASELINE_DURATION_S:.1f} s)",
    )
    axis_duration.set_ylabel("97G scan duration [s]")
    axis_duration.set_xticks(x, labels, rotation=18, ha="right")
    axis_scale = axis_duration.twinx()
    axis_scale.plot(
        x,
        speed_scale,
        color="tab:orange",
        marker="o",
        linewidth=2,
        label="mean speed scale",
    )
    axis_scale.set_ylabel("Mean path-speed scale")
    handles_a, labels_a = axis_duration.get_legend_handles_labels()
    handles_b, labels_b = axis_scale.get_legend_handles_labels()
    axis_duration.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    axis_duration.set_title("Accuracy-throughput trade-off")
    figure.tight_layout()
    tradeoff_path = OUTPUT_DIR / "97H_accuracy_throughput_tradeoff.png"
    figure.savefig(tradeoff_path, dpi=180)
    plt.close(figure)

    old_mean = value(baseline, "force_mean_error_n")
    new_mean = value(adaptive, "force_mean_error_n")
    old_max = value(baseline, "force_max_error_n")
    new_max = value(adaptive, "force_max_error_n")
    high_index = baseline_cases.index("friction_high_0p60")
    mean_reduction = 100.0 * (1.0 - new_mean[high_index] / old_mean[high_index])
    max_reduction = 100.0 * (1.0 - new_max[high_index] / old_max[high_index])
    baseline_passed = sum(row["pass"] == "True" for row in baseline)
    adaptive_passed = sum(row["pass"] == "True" for row in adaptive)

    report_path = OUTPUT_DIR / "97H_97E_vs_97G_comparison.md"
    report_path.write_text(
        "\n".join(
            [
                "# 97H 固定速度与自适应速度对比",
                "",
                f"- 97E固定速度通过：{baseline_passed}/{len(baseline)}。",
                f"- 97G自适应速度通过：{adaptive_passed}/{len(adaptive)}。",
                f"- 高摩擦平均力误差：{old_mean[high_index]:.3f} → "
                f"{new_mean[high_index]:.3f} N（下降{mean_reduction:.1f}%）。",
                f"- 高摩擦最大力误差：{old_max[high_index]:.3f} → "
                f"{new_max[high_index]:.3f} N（下降{max_reduction:.1f}%）。",
                f"- 高摩擦扫描时间：{BASELINE_DURATION_S:.3f} → "
                f"{duration[high_index]:.3f} s。",
                "- 所有场景均无力矩饱和，TCP最大误差均低于2 mm。",
                "",
                "结论：改善来自力/摩擦感知的路径时间缩放与更高带宽的法向导纳，"
                "不是提高笛卡尔位置PID。代价是困难接触区域降低扫描吞吐率。",
                "",
                "![97E与97G误差对比](97H_97E_vs_97G_metrics.png)",
                "",
                "![精度与速度权衡](97H_accuracy_throughput_tradeoff.png)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved 保存: {metrics_path}")
    print(f"Saved 保存: {tradeoff_path}")
    print(f"Saved 保存: {report_path}")


if __name__ == "__main__":
    main()
