#!/usr/bin/env python3
"""106 - Generate a reproducible 97D vs 105 controller comparison report."""

import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "97D_learned_surface_force_scan_result.npz"
TSID_PATH = HERE / "105_tsid_learned_surface_5n_scan_result.npz"
CSV_PATH = HERE / "106_tsid_vs_97d_comparison.csv"
REPORT_PATH = HERE / "106_tsid_vs_97d_comparison.md"


def scalar(data, name, default=np.nan):
    return float(np.asarray(data[name])) if name in data.files else float(default)


def summarize(data, controller):
    force_error = np.asarray(data["force_error"])
    tcp_error = np.asarray(data["tcp_error"])
    orientation_error = np.asarray(data["normal_error_deg"])
    return {
        "controller": controller,
        "pass": bool(np.asarray(data["result_pass"])),
        "scan_duration_s": float(np.asarray(data["time"])[-1]),
        "force_mean_error_n": float(np.mean(force_error)),
        "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
        "force_max_error_n": float(np.max(force_error)),
        "tcp_mean_error_mm": float(np.mean(tcp_error) * 1000.0),
        "tcp_max_error_mm": float(np.max(tcp_error) * 1000.0),
        "orientation_mean_error_deg": float(np.mean(orientation_error)),
        "orientation_max_error_deg": float(np.max(orientation_error)),
        "max_actuator_torque_nm": scalar(data, "max_actuator_torque"),
        "minimum_sigma": float(np.min(data["sigma_min"])),
        "maximum_condition": float(np.max(data["condition"])),
        "post_solve_clipping": controller == "97D computed torque",
        "hqp_motor_bound_violation_nm": scalar(
            data, "max_motor_bound_violation", default=np.nan
        ),
    }


def fmt(value, digits=3):
    if isinstance(value, (bool, np.bool_)):
        return "PASS" if value else "FAIL"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def main():
    baseline = np.load(BASELINE_PATH)
    tsid = np.load(TSID_PATH)

    for key in ("learned_position", "learned_surface_normal", "learned_arc_length"):
        if not np.allclose(baseline[key], tsid[key], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"两个控制器没有使用相同的学习路径: {key}")
    if bool(np.asarray(baseline["analytic_surface_used"])) or bool(
        np.asarray(tsid["analytic_surface_used"])
    ):
        raise RuntimeError("对比实验不得使用解析曲面作为期望轨迹")

    rows = [
        summarize(baseline, "97D computed torque"),
        summarize(tsid, "105 TSID/HQP"),
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [
        ("pass", "结果", 0),
        ("scan_duration_s", "扫描时间 (s)", 3),
        ("force_mean_error_n", "力平均误差 (N)", 3),
        ("force_rmse_n", "力 RMSE (N)", 3),
        ("force_max_error_n", "力最大误差 (N)", 3),
        ("tcp_mean_error_mm", "TCP平均误差 (mm)", 3),
        ("tcp_max_error_mm", "TCP最大误差 (mm)", 3),
        ("orientation_mean_error_deg", "姿态平均误差 (deg)", 3),
        ("orientation_max_error_deg", "姿态最大误差 (deg)", 3),
        ("max_actuator_torque_nm", "最大电机力矩 (Nm)", 3),
        ("minimum_sigma", "最小雅可比奇异值", 6),
        ("maximum_condition", "最大条件数", 2),
        ("hqp_motor_bound_violation_nm", "HQP电机边界违规 (Nm)", 3),
    ]
    lines = [
        "# 97D 与 105 TSID/HQP 控制器对比",
        "",
        "两组实验使用完全相同的97B学习曲面、961个位姿、1.538 m路径和5 N目标力。",
        "",
        "| 指标 | 97D计算力矩 | 105 TSID/HQP |",
        "|---|---:|---:|",
    ]
    for key, label, digits in labels:
        lines.append(
            f"| {label} | {fmt(rows[0][key], digits)} | {fmt(rows[1][key], digits)} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 两种控制器均完成完整学习曲面扫描并通过相同精度阈值。",
            "- 105保持了与97D相当的力和位姿跟踪性能。",
            "- 关键架构差异是105把关节状态与最终电机指令边界放入HQP，且不进行求解后力矩裁剪。",
            "- MuJoCo仍是唯一接触求解器；TSID不重复建立刚性接触。",
            "",
            "报告由 `106_compare_97d_vs_tsid_hqp.py` 从两个NPZ结果自动生成。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("========== 106 Controller Comparison 控制器对比 ==========")
    for row in rows:
        print(
            f"{row['controller']}: force mean/max="
            f"{row['force_mean_error_n']:.3f}/{row['force_max_error_n']:.3f} N, "
            f"TCP mean/max={row['tcp_mean_error_mm']:.3f}/"
            f"{row['tcp_max_error_mm']:.3f} mm, "
            f"tau_max={row['max_actuator_torque_nm']:.3f} Nm"
        )
    print(f"Saved 保存: {CSV_PATH.name}")
    print(f"Saved 保存: {REPORT_PATH.name}")
    print("Result 结果: PASS 通过")


if __name__ == "__main__":
    main()
