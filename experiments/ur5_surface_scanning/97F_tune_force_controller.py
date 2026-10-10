#!/usr/bin/env python3
"""97F - Tune the 97D force controller on nominal/high-friction scans.

The baseline controller file is not modified.  Each candidate is injected into
an imported 97D module and evaluated over the complete 1.538 m learned path in
both nominal and high-friction environments.  Selection considers force mean,
P95/max force error, TCP max error and scan time; failed/saturated candidates
receive a large penalty.

This script is intentionally a small, auditable design-of-experiments sweep,
not an opaque optimizer.  The remaining five 97E perturbations are reserved
for the subsequent 97G validation to reduce parameter overfitting.
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
OUTPUT_DIR = HERE / "results" / "controller_tuning"


# Candidate values are centered on the verified 97D baseline.  The sweep first
# separates trajectory timing from force-loop bandwidth, then combines the
# most plausible changes.  Names are stable identifiers used by 97G.
CANDIDATES = (
    {
        "name": "baseline",
        "scan_speed": 0.040,
        "connector_speed": 0.025,
        "minimum_segment_duration": 0.40,
        "force_mass": 2.0,
        "force_damping": 120.0,
        "normal_velocity_limit": 0.004,
        "normal_acceleration_limit": 0.15,
    },
    {
        "name": "slower_35_20",
        "scan_speed": 0.035,
        "connector_speed": 0.020,
        "minimum_segment_duration": 0.50,
        "force_mass": 2.0,
        "force_damping": 120.0,
        "normal_velocity_limit": 0.004,
        "normal_acceleration_limit": 0.15,
    },
    {
        "name": "slower_30_15",
        "scan_speed": 0.030,
        "connector_speed": 0.015,
        "minimum_segment_duration": 0.60,
        "force_mass": 2.0,
        "force_damping": 120.0,
        "normal_velocity_limit": 0.004,
        "normal_acceleration_limit": 0.15,
    },
    {
        "name": "faster_admittance",
        "scan_speed": 0.040,
        "connector_speed": 0.025,
        "minimum_segment_duration": 0.40,
        "force_mass": 1.5,
        "force_damping": 100.0,
        "normal_velocity_limit": 0.005,
        "normal_acceleration_limit": 0.25,
    },
    {
        "name": "balanced_35_20",
        "scan_speed": 0.035,
        "connector_speed": 0.020,
        "minimum_segment_duration": 0.50,
        "force_mass": 1.5,
        "force_damping": 100.0,
        "normal_velocity_limit": 0.005,
        "normal_acceleration_limit": 0.25,
    },
    {
        "name": "robust_30_15",
        "scan_speed": 0.030,
        "connector_speed": 0.015,
        "minimum_segment_duration": 0.60,
        "force_mass": 1.5,
        "force_damping": 110.0,
        "normal_velocity_limit": 0.005,
        "normal_acceleration_limit": 0.25,
    },
)

TRAIN_CASES = ("nominal", "friction_high_0p60")


class Camera:
    def __init__(self):
        self.lookat = np.zeros(3)
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0


class HeadlessViewer:
    def __init__(self, data, stop_time=100.0):
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


def apply_candidate(controller, candidate):
    controller.SCAN_AVG_SPEED = candidate["scan_speed"]
    controller.CONNECTOR_AVG_SPEED = candidate["connector_speed"]
    controller.MIN_SEGMENT_DURATION = candidate["minimum_segment_duration"]
    controller.FORCE_MASS = candidate["force_mass"]
    controller.FORCE_DAMPING = candidate["force_damping"]
    controller.NORMAL_VELOCITY_LIMIT = candidate["normal_velocity_limit"]
    controller.NORMAL_ACCEL_LIMIT = candidate["normal_acceleration_limit"]


def summarize(candidate, case_name, result):
    force_error = np.asarray(result["force_error"])
    tcp_error_mm = 1000.0 * np.asarray(result["tcp_error"])
    normal_error = np.asarray(result["normal_error_deg"])
    duration = float(np.asarray(result["time"])[-1])
    saturation = float(np.asarray(result["saturation_ratio_percent"]))
    passed = bool(np.asarray(result["result_pass"]))

    # Dimensionless score relative to meaningful engineering scales.  Maximum
    # error matters, but does not completely dominate P95/mean behavior.
    score = (
        float(np.mean(force_error)) / 0.35
        + float(np.percentile(force_error, 95)) / 1.0
        + 0.35 * float(np.max(force_error)) / 3.0
        + 0.35 * float(np.max(tcp_error_mm)) / 1.2
        + 0.15 * duration / 40.3
    )
    if saturation > 0.0:
        score += 100.0 + saturation

    return {
        "candidate": candidate["name"],
        "case": case_name,
        "pass": passed,
        "score": score,
        "scan_speed_mm_s": 1000.0 * candidate["scan_speed"],
        "connector_speed_mm_s": 1000.0 * candidate["connector_speed"],
        "force_mass": candidate["force_mass"],
        "force_damping": candidate["force_damping"],
        "normal_velocity_limit_mm_s": 1000.0
        * candidate["normal_velocity_limit"],
        "normal_acceleration_limit": candidate["normal_acceleration_limit"],
        "duration_s": duration,
        "force_mean_error_n": float(np.mean(force_error)),
        "force_rmse_n": float(np.sqrt(np.mean(force_error**2))),
        "force_p95_error_n": float(np.percentile(force_error, 95)),
        "force_max_error_n": float(np.max(force_error)),
        "tcp_mean_error_mm": float(np.mean(tcp_error_mm)),
        "tcp_p95_error_mm": float(np.percentile(tcp_error_mm, 95)),
        "tcp_max_error_mm": float(np.max(tcp_error_mm)),
        "orientation_max_error_deg": float(np.max(normal_error)),
        "max_actuator_torque_nm": float(
            np.asarray(result["max_actuator_torque"])
        ),
        "saturation_ratio_percent": saturation,
    }


def candidate_ranking(rows):
    ranking = []
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == candidate["name"]]
        if len(selected) != len(TRAIN_CASES):
            continue
        high = next(row for row in selected if row["case"] == "friction_high_0p60")
        nominal = next(row for row in selected if row["case"] == "nominal")
        # Robustness is weighted more heavily than nominal because it is the
        # only known failing boundary.  Time remains a term in each score.
        aggregate = 0.4 * nominal["score"] + 0.6 * high["score"]
        # Use the complete 97D acceptance decision for both tuning cases.  In
        # particular, high-friction peak force must not be hidden by good TCP
        # tracking or a low aggregate score.
        safety_feasible = nominal["pass"] and high["pass"]
        ranking.append(
            {
                "candidate": candidate["name"],
                "aggregate_score": aggregate,
                "safety_feasible": safety_feasible,
                "nominal_force_mean_n": nominal["force_mean_error_n"],
                "nominal_force_p95_n": nominal["force_p95_error_n"],
                "nominal_force_max_n": nominal["force_max_error_n"],
                "high_friction_force_mean_n": high["force_mean_error_n"],
                "high_friction_force_p95_n": high["force_p95_error_n"],
                "high_friction_force_max_n": high["force_max_error_n"],
                "nominal_tcp_max_mm": nominal["tcp_max_error_mm"],
                "high_friction_tcp_max_mm": high["tcp_max_error_mm"],
                "duration_s": nominal["duration_s"],
            }
        )
    ranking.sort(key=lambda row: (not row["safety_feasible"], row["aggregate_score"]))
    return ranking


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(ranking):
    report = OUTPUT_DIR / "97F_tuning_report.md"
    lines = [
        "# 97F 力控制器参数筛选",
        "",
        "训练场景仅使用名义工况和高摩擦工况，其余五个扰动留给97G独立验证。",
        "",
        "| 排名 | 参数组 | 可行 | 综合分数 | 名义力均值/P95/最大 (N) | 高摩擦力均值/P95/最大 (N) | TCP最大 名义/高摩擦 (mm) | 时间 (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranking, start=1):
        lines.append(
            f"| {index} | {row['candidate']} | {row['safety_feasible']} | "
            f"{row['aggregate_score']:.3f} | "
            f"{row['nominal_force_mean_n']:.3f}/{row['nominal_force_p95_n']:.3f}/"
            f"{row['nominal_force_max_n']:.3f} | "
            f"{row['high_friction_force_mean_n']:.3f}/"
            f"{row['high_friction_force_p95_n']:.3f}/"
            f"{row['high_friction_force_max_n']:.3f} | "
            f"{row['nominal_tcp_max_mm']:.3f}/"
            f"{row['high_friction_tcp_max_mm']:.3f} | {row['duration_s']:.3f} |"
        )
    winner = ranking[0]
    lines.append("")
    if winner["safety_feasible"]:
        lines.append(f"推荐候选：**{winner['candidate']}**。")
    else:
        lines.extend(
            [
                f"固定参数中得分最优的是 **{winner['candidate']}**，"
                "但没有候选同时通过名义与高摩擦工况。",
                "",
                "结论：不能仅靠固定减速或增大导纳带宽解决高摩擦峰值，"
                "后续97G改用力/摩擦感知的在线路径速度缩放。",
            ]
        )
    lines.extend(
        [
            "",
            "参数筛选仅使用两个训练场景；97G使用全部七场景独立复验。",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=[candidate["name"] for candidate in CANDIDATES],
        help="只运行指定参数组，可重复传入",
    )
    args = parser.parse_args()
    candidates = [
        candidate
        for candidate in CANDIDATES
        if not args.candidate or candidate["name"] in args.candidate
    ]

    robustness = load_module("robustness_97e", ROBUSTNESS_PATH)
    controller = load_module("controller_97d_tuning", CONTROLLER_PATH)
    scenarios = {
        scenario["name"]: scenario for scenario in robustness.SCENARIOS
    }
    controller.time.sleep = lambda _: None
    controller.PRINT_INTERVAL = 1000.0
    controller.mujoco.viewer.launch_passive = (
        lambda model, data: HeadlessViewer(data, stop_time=100.0)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print("========== 97F Force-Controller Tuning 97F力控制器调参 ==========")
    print(f"Candidates 参数组: {len(candidates)}")
    print(f"Training cases 训练场景: {', '.join(TRAIN_CASES)}")

    with tempfile.TemporaryDirectory(prefix="ur5_97f_") as temporary:
        temporary = Path(temporary)
        for candidate_index, candidate in enumerate(candidates, start=1):
            apply_candidate(controller, candidate)
            for case_index, case_name in enumerate(TRAIN_CASES, start=1):
                print(
                    f"\n--- Candidate {candidate_index}/{len(candidates)} "
                    f"{candidate['name']} | case {case_index}/{len(TRAIN_CASES)} "
                    f"{case_name} ---"
                )
                model_path = temporary / f"{candidate['name']}_{case_name}.xml"
                result_path = temporary / f"{candidate['name']}_{case_name}.npz"
                robustness.build_perturbed_model(scenarios[case_name], model_path)
                controller.MJCF_PATH = model_path
                controller.OUTPUT_PATH = result_path
                controller.main()
                with np.load(result_path) as result:
                    row = summarize(candidate, case_name, result)
                rows.append(row)
                print(
                    f"Metrics: mean/P95/max force="
                    f"{row['force_mean_error_n']:.3f}/"
                    f"{row['force_p95_error_n']:.3f}/"
                    f"{row['force_max_error_n']:.3f} N | "
                    f"TCPmax={row['tcp_max_error_mm']:.3f} mm | "
                    f"time={row['duration_s']:.2f}s"
                )

    detail_path = OUTPUT_DIR / "97F_tuning_runs.csv"
    write_csv(detail_path, rows)
    ranking = candidate_ranking(rows)
    ranking_path = OUTPUT_DIR / "97F_tuning_ranking.csv"
    write_csv(ranking_path, ranking)
    report_path = write_report(ranking)

    print("\n========== 97F Ranking 排名 ==========")
    for index, row in enumerate(ranking, start=1):
        print(
            f"{index}. {row['candidate']}: score={row['aggregate_score']:.3f}, "
            f"feasible={row['safety_feasible']}, "
            f"Fhigh mean/max={row['high_friction_force_mean_n']:.3f}/"
            f"{row['high_friction_force_max_n']:.3f} N"
        )
    print(f"Saved 保存: {detail_path}")
    print(f"Saved 保存: {ranking_path}")
    print(f"Saved 保存: {report_path}")


if __name__ == "__main__":
    main()
