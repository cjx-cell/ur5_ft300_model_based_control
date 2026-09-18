#!/usr/bin/env python3
"""107 - Seven-case robustness suite for the 105 TSID/HQP scan controller."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROLLER_PATH = HERE / "105_ur5_tsid_learned_surface_5n_scan.py"
SCENARIO_PATH = HERE / "97E_ur5_learned_surface_robustness.py"
OUTPUT_DIR = HERE / "results" / "tsid_robustness"
BASELINE_CSV = HERE / "results" / "robustness" / "97E_robustness_summary.csv"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scalar(result, key, default=np.nan):
    return float(np.asarray(result[key])) if key in result.files else float(default)


def summarize(scenario, result):
    return {
        "case": scenario["name"],
        "pass": bool(np.asarray(result["result_pass"])),
        "offset_x_mm": 1000.0 * scenario["offset_x"],
        "offset_z_mm": 1000.0 * scenario["offset_z"],
        "tilt_y_deg": scenario["tilt_y_deg"],
        "friction": scenario["friction"],
        "force_mean_error_n": float(np.mean(result["force_error"])),
        "force_max_error_n": float(np.max(result["force_error"])),
        "tcp_mean_error_mm": float(1000.0 * np.mean(result["tcp_error"])),
        "tcp_max_error_mm": float(1000.0 * np.max(result["tcp_error"])),
        "orientation_max_error_deg": float(np.max(result["normal_error_deg"])),
        "learned_normal_max_deviation_deg": float(
            np.max(result["learned_normal_deviation_deg"])
        ),
        "tilt_correction_max_deg": float(np.max(result["tilt_angle_deg"])),
        "normal_correction_min_mm": float(
            1000.0 * np.min(result["normal_correction"])
        ),
        "normal_correction_max_mm": float(
            1000.0 * np.max(result["normal_correction"])
        ),
        "sigma_min": float(np.min(result["sigma_min"])),
        "condition_max": float(np.max(result["condition"])),
        "max_actuator_torque_nm": scalar(result, "max_actuator_torque"),
        "max_motor_bound_violation_nm": scalar(
            result, "max_motor_bound_violation"
        ),
    }


def write_outputs(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "107_tsid_robustness_summary.csv"
    npz_path = OUTPUT_DIR / "107_tsid_robustness_summary.npz"
    report_path = OUTPUT_DIR / "107_tsid_vs_97d_robustness.md"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        **{key: np.asarray([row[key] for row in rows]) for key in rows[0]},
    )

    baseline = {}
    if BASELINE_CSV.exists():
        with BASELINE_CSV.open(encoding="utf-8") as stream:
            baseline = {row["case"]: row for row in csv.DictReader(stream)}

    lines = [
        "# 97D 与 105 TSID/HQP 鲁棒性对比",
        "",
        "控制器始终使用未修改的97B学习曲面；工件位姿和摩擦扰动仅存在于MuJoCo环境中。",
        "",
        "| 场景 | 97D | TSID/HQP | 97D力均值/最大 (N) | TSID力均值/最大 (N) | TSID TCP最大 (mm) | HQP边界违规 (Nm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = baseline.get(row["case"])
        old_pass = (
            ("PASS" if old["pass"].lower() == "true" else "FAIL")
            if old
            else "N/A"
        )
        new_pass = "PASS" if row["pass"] else "FAIL"
        old_force = (
            f"{float(old['force_mean_error_n']):.3f}/{float(old['force_max_error_n']):.3f}"
            if old
            else "N/A"
        )
        lines.append(
            f"| {row['case']} | {old_pass} | {new_pass} | {old_force} | "
            f"{row['force_mean_error_n']:.3f}/{row['force_max_error_n']:.3f} | "
            f"{row['tcp_max_error_mm']:.3f} | "
            f"{row['max_motor_bound_violation_nm']:.3e} |"
        )
    passed = sum(row["pass"] for row in rows)
    lines.extend(
        [
            "",
            f"TSID/HQP通过场景：**{passed}/{len(rows)}**。",
            "",
            "所有场景均使用HQP内部关节/电机边界，控制输出不进行事后裁剪。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, npz_path, report_path


def main():
    scenario_module = load_module("scenarios_97e", SCENARIO_PATH)
    controller = load_module("controller_105", CONTROLLER_PATH)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case["name"] for case in scenario_module.SCENARIOS],
        help="只运行指定场景，可重复传入",
    )
    parser.add_argument("--keep-traces", action="store_true", help="保留完整时间序列")
    args = parser.parse_args()
    selected = [
        case
        for case in scenario_module.SCENARIOS
        if not args.case or case["name"] in args.case
    ]

    controller.PRINT_INTERVAL = 1000.0
    rows = []
    print("========== 107 TSID/HQP Robustness Suite 鲁棒性验证 ==========")
    print("Controller knowledge 控制器已知: unchanged 97B learned path")

    with tempfile.TemporaryDirectory(prefix="ur5_107_") as temporary:
        temporary = Path(temporary)
        for index, scenario in enumerate(selected, start=1):
            print(f"\n--- Case {index}/{len(selected)}: {scenario['name']} ---")
            model_path = temporary / f"{scenario['name']}.xml"
            result_path = temporary / f"{scenario['name']}.npz"
            scenario_module.build_perturbed_model(scenario, model_path)
            controller.MJCF_PATH = model_path
            controller.OUTPUT_PATH = result_path
            controller.main([], raise_on_failure=False)
            with np.load(result_path) as result:
                row = summarize(scenario, result)
            rows.append(row)
            print(
                f"Case result: {'PASS' if row['pass'] else 'FAIL'} | "
                f"Fmean/max={row['force_mean_error_n']:.3f}/"
                f"{row['force_max_error_n']:.3f} N | "
                f"TCPmax={row['tcp_max_error_mm']:.3f} mm | "
                f"bound={row['max_motor_bound_violation_nm']:.3e} Nm"
            )
            if args.keep_traces:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUTPUT_DIR / f"107_{scenario['name']}_trace.npz").write_bytes(
                    result_path.read_bytes()
                )

    csv_path, npz_path, report_path = write_outputs(rows)
    passed = sum(row["pass"] for row in rows)
    print("\n========== 107 Summary 汇总 ==========")
    print(f"Passed 通过: {passed}/{len(rows)}")
    print(f"Saved 保存: {csv_path}")
    print(f"Saved 保存: {npz_path}")
    print(f"Saved 保存: {report_path}")


if __name__ == "__main__":
    main()
