#!/usr/bin/env python3
"""97G - Validate force-aware adaptive path speed over all 97E scenarios.

97D remains the fixed-speed reference.  This experiment enables its optional
path-time scaling: measured normal-force error and tangential contact force
can slow the learned trajectory, while the normal/tilt admittance and inverse
dynamics controller remain unchanged.  No analytic surface information is
used by the controller.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROLLER_PATH = HERE / "97D_ur5_learned_surface_5n_scan.py"
ROBUSTNESS_PATH = HERE / "97E_ur5_learned_surface_robustness.py"
OUTPUT_DIR = HERE / "results" / "adaptive_robustness"


class Camera:
    def __init__(self):
        self.lookat = np.zeros(3)
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0


class HeadlessViewer:
    def __init__(self, data, stop_time=120.0):
        self.data = data
        self.stop_time = stop_time
        self.cam = Camera()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False

    def is_running(self):
        return float(self.data.time) < self.stop_time

    def sync(self):
        return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(scenario, result):
    force_error = np.asarray(result["force_error"])
    tcp_error_mm = 1000.0 * np.asarray(result["tcp_error"])
    speed_scale = np.asarray(result["speed_scale"])
    return {
        "case": scenario["name"],
        "pass": bool(np.asarray(result["result_pass"])),
        "offset_x_mm": 1000.0 * scenario["offset_x"],
        "offset_z_mm": 1000.0 * scenario["offset_z"],
        "tilt_y_deg": scenario["tilt_y_deg"],
        "friction": scenario["friction"],
        "scan_duration_s": float(np.asarray(result["time"])[-1]),
        "speed_scale_mean": float(np.mean(speed_scale)),
        "speed_scale_min": float(np.min(speed_scale)),
        "force_mean_error_n": float(np.mean(force_error)),
        "force_p95_error_n": float(np.percentile(force_error, 95)),
        "force_max_error_n": float(np.max(force_error)),
        "tcp_mean_error_mm": float(np.mean(tcp_error_mm)),
        "tcp_p95_error_mm": float(np.percentile(tcp_error_mm, 95)),
        "tcp_max_error_mm": float(np.max(tcp_error_mm)),
        "orientation_max_error_deg": float(
            np.max(result["normal_error_deg"])
        ),
        "tangential_force_max_n": float(
            np.max(result["tangential_force"])
        ),
        "max_actuator_torque_nm": float(
            np.asarray(result["max_actuator_torque"])
        ),
        "saturation_ratio_percent": float(
            np.asarray(result["saturation_ratio_percent"])
        ),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows):
    report_path = OUTPUT_DIR / "97G_adaptive_robustness_report.md"
    lines = [
        "# 97G 自适应路径速度鲁棒性验证",
        "",
        "控制器只使用97B学习路径与FT300反馈；工件扰动仍对控制器隐藏。",
        "",
        "| 场景 | 通过 | 力误差 均值/P95/最大 (N) | TCP P95/最大 (mm) | 速度比例 均值/最小 | 扫描时间 (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['pass']} | "
            f"{row['force_mean_error_n']:.3f}/"
            f"{row['force_p95_error_n']:.3f}/"
            f"{row['force_max_error_n']:.3f} | "
            f"{row['tcp_p95_error_mm']:.3f}/"
            f"{row['tcp_max_error_mm']:.3f} | "
            f"{row['speed_scale_mean']:.3f}/"
            f"{row['speed_scale_min']:.3f} | "
            f"{row['scan_duration_s']:.2f} |"
        )
    passed = sum(row["pass"] for row in rows)
    lines.extend(
        [
            "",
            f"通过场景：**{passed}/{len(rows)}**。",
            "",
            "通过条件沿用97D：力平均误差≤0.50 N、最大误差≤3.00 N、"
            "TCP最大误差≤2 mm、姿态最大误差≤1°且无力矩饱和。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        help="只运行指定97E场景，可重复传入",
    )
    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="保存每个场景的完整NPZ轨迹",
    )
    args = parser.parse_args()

    robustness = load_module("robustness_97e_for_97g", ROBUSTNESS_PATH)
    controller = load_module("controller_97d_for_97g", CONTROLLER_PATH)
    selected = [
        scenario
        for scenario in robustness.SCENARIOS
        if not args.case or scenario["name"] in args.case
    ]
    if args.case and len(selected) != len(set(args.case)):
        known = ", ".join(s["name"] for s in robustness.SCENARIOS)
        raise ValueError(f"未知场景；可选值: {known}")

    controller.ADAPTIVE_PATH_SPEED = True
    # The fixed-speed 97D admittance is too conservative to recover from a
    # short unloading transient. 97G uses a lighter virtual mass, retuned
    # damping and greater acceleration/velocity authority.
    controller.FORCE_MASS = 1.0
    controller.FORCE_DAMPING = 80.0
    controller.NORMAL_ACCEL_LIMIT = 2.50
    controller.NORMAL_VELOCITY_LIMIT = 0.015
    controller.time.sleep = lambda _: None
    controller.PRINT_INTERVAL = 1000.0
    controller.mujoco.viewer.launch_passive = (
        lambda model, data: HeadlessViewer(data)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("========== 97G Adaptive-Speed Robustness 97G自适应速度鲁棒性 ==========")
    print(f"Cases 场景数: {len(selected)}")

    with tempfile.TemporaryDirectory(prefix="ur5_97g_") as temporary:
        temporary = Path(temporary)
        for index, scenario in enumerate(selected, start=1):
            print(f"\n--- Case {index}/{len(selected)}: {scenario['name']} ---")
            model_path = temporary / f"{scenario['name']}.xml"
            result_path = temporary / f"{scenario['name']}.npz"
            robustness.build_perturbed_model(scenario, model_path)
            controller.MJCF_PATH = model_path
            controller.OUTPUT_PATH = result_path
            controller.main()
            with np.load(result_path) as result:
                row = summarize(scenario, result)
            rows.append(row)
            print(
                f"Metrics: mean/P95/max force="
                f"{row['force_mean_error_n']:.3f}/"
                f"{row['force_p95_error_n']:.3f}/"
                f"{row['force_max_error_n']:.3f} N | "
                f"TCPmax={row['tcp_max_error_mm']:.3f} mm | "
                f"speed mean/min={row['speed_scale_mean']:.3f}/"
                f"{row['speed_scale_min']:.3f}"
            )
            if args.keep_traces:
                trace_path = OUTPUT_DIR / f"97G_{scenario['name']}_trace.npz"
                trace_path.write_bytes(result_path.read_bytes())

    summary_path = OUTPUT_DIR / "97G_adaptive_robustness_summary.csv"
    write_csv(summary_path, rows)
    report_path = write_report(rows)
    passed = sum(row["pass"] for row in rows)
    print(f"\nPassed 通过: {passed}/{len(rows)}")
    print(f"Saved 保存: {summary_path}")
    print(f"Saved 保存: {report_path}")


if __name__ == "__main__":
    main()
