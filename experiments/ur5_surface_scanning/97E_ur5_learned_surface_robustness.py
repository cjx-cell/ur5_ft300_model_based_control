#!/usr/bin/env python3
"""97E - UR5 learned-surface force-scan robustness suite.

Each case executes the complete 1.538 m step-97D scan.  The controller keeps
using the unchanged 97B learned path; only the hidden MuJoCo workpiece is
perturbed.  This mirrors the earlier step-75 robustness methodology.

Default cases
-------------
- nominal
- workpiece Z offset: +1.0 mm and -1.0 mm
- workpiece X offset: +1.0 mm
- workpiece Y-axis tilt: +1.0 degree
- contact friction: 0.15 and 0.60 (nominal 0.30)

Outputs
-------
results/robustness/97E_robustness_summary.csv
results/robustness/97E_robustness_summary.npz

Use --keep-traces to retain each case's full step-97D NPZ trace.  Without it,
only the compact summary is retained.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
CONTROLLER_PATH = HERE / "97D_ur5_learned_surface_5n_scan.py"
SOURCE_MJCF = HERE / "ur5_ft300_inspection_workcell.xml"
OUTPUT_DIR = HERE / "results" / "robustness"


SCENARIOS = (
    {
        "name": "nominal",
        "offset_x": 0.0,
        "offset_z": 0.0,
        "tilt_y_deg": 0.0,
        "friction": 0.30,
    },
    {
        "name": "offset_z_plus_1mm",
        "offset_x": 0.0,
        "offset_z": +0.001,
        "tilt_y_deg": 0.0,
        "friction": 0.30,
    },
    {
        "name": "offset_z_minus_1mm",
        "offset_x": 0.0,
        "offset_z": -0.001,
        "tilt_y_deg": 0.0,
        "friction": 0.30,
    },
    {
        "name": "offset_x_plus_1mm",
        "offset_x": +0.001,
        "offset_z": 0.0,
        "tilt_y_deg": 0.0,
        "friction": 0.30,
    },
    {
        "name": "tilt_y_plus_1deg",
        "offset_x": 0.0,
        "offset_z": 0.0,
        "tilt_y_deg": +1.0,
        "friction": 0.30,
    },
    {
        "name": "friction_low_0p15",
        "offset_x": 0.0,
        "offset_z": 0.0,
        "tilt_y_deg": 0.0,
        "friction": 0.15,
    },
    {
        "name": "friction_high_0p60",
        "offset_x": 0.0,
        "offset_z": 0.0,
        "tilt_y_deg": 0.0,
        "friction": 0.60,
    },
)


class _Camera:
    def __init__(self):
        self.lookat = np.zeros(3)
        self.distance = 0.0
        self.azimuth = 0.0
        self.elevation = 0.0


class _HeadlessViewer:
    def __init__(
        self,
        model,
        data,
        stop_time,
    ):
        del model
        self.data = data
        self.stop_time = stop_time
        self.cam = _Camera()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False

    def is_running(self):
        return float(self.data.time) < self.stop_time

    def sync(self):
        return None


def load_controller():
    spec = importlib.util.spec_from_file_location(
        "ur5_scan_97d",
        CONTROLLER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载控制器: {CONTROLLER_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def parse_vector(text):
    return np.asarray(
        [float(value) for value in text.split()],
        dtype=float,
    )


def format_vector(vector):
    return " ".join(
        f"{value:.9g}"
        for value in vector
    )


def build_perturbed_model(
    scenario,
    output_path,
):
    tree = ET.parse(
        SOURCE_MJCF
    )

    root = tree.getroot()
    workpiece = root.find(
        ".//geom[@name='inspection_workpiece']"
    )

    if workpiece is None:
        raise RuntimeError(
            "MJCF中找不到inspection_workpiece"
        )

    position = parse_vector(
        workpiece.attrib["pos"]
    )

    position[0] += scenario[
        "offset_x"
    ]

    position[2] += scenario[
        "offset_z"
    ]

    workpiece.set(
        "pos",
        format_vector(position),
    )

    tilt = np.deg2rad(
        scenario["tilt_y_deg"]
    )

    workpiece.set(
        "quat",
        format_vector(
            np.array(
                [
                    np.cos(0.5 * tilt),
                    0.0,
                    np.sin(0.5 * tilt),
                    0.0,
                ]
            )
        ),
    )

    nominal_friction = parse_vector(
        workpiece.attrib.get(
            "friction",
            "0.30 0.01 0.001",
        )
    )

    nominal_friction[0] = scenario[
        "friction"
    ]

    workpiece.set(
        "friction",
        format_vector(
            nominal_friction
        ),
    )

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )


def scalar(data, key):
    return float(
        np.asarray(data[key])
    )


def summarize(
    scenario,
    result,
):
    force_error = result[
        "force_error"
    ]

    tcp_error = result[
        "tcp_error"
    ]

    orientation_error = result[
        "normal_error_deg"
    ]

    learned_deviation = result[
        "learned_normal_deviation_deg"
    ]

    tilt_moment = result[
        "tilt_moment_norm"
    ]

    tilt_angle = result[
        "tilt_angle_deg"
    ]

    normal_correction = result[
        "normal_correction"
    ]

    return {
        "case": scenario["name"],
        "pass": bool(
            result["result_pass"]
        ),
        "offset_x_mm": 1000.0
        * scenario["offset_x"],
        "offset_z_mm": 1000.0
        * scenario["offset_z"],
        "tilt_y_deg": scenario[
            "tilt_y_deg"
        ],
        "friction": scenario[
            "friction"
        ],
        "force_mean_error_n": float(
            np.mean(force_error)
        ),
        "force_max_error_n": float(
            np.max(force_error)
        ),
        "tcp_mean_error_mm": float(
            1000.0 * np.mean(tcp_error)
        ),
        "tcp_max_error_mm": float(
            1000.0 * np.max(tcp_error)
        ),
        "orientation_max_error_deg": float(
            np.max(orientation_error)
        ),
        "learned_normal_max_deviation_deg": float(
            np.max(learned_deviation)
        ),
        "tcp_tilt_moment_max_nm": float(
            np.max(tilt_moment)
        ),
        "tilt_correction_max_deg": float(
            np.max(tilt_angle)
        ),
        "normal_correction_min_mm": float(
            1000.0 * np.min(normal_correction)
        ),
        "normal_correction_max_mm": float(
            1000.0 * np.max(normal_correction)
        ),
        "sigma_min": float(
            np.min(result["sigma_min"])
        ),
        "condition_max": float(
            np.max(result["condition"])
        ),
        "max_actuator_torque_nm": scalar(
            result,
            "max_actuator_torque",
        ),
        "saturation_ratio_percent": scalar(
            result,
            "saturation_ratio_percent",
        ),
    }


def write_summary(rows):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "97E_robustness_summary.csv"
    )

    npz_path = (
        OUTPUT_DIR
        / "97E_robustness_summary.npz"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        npz_path,
        **{
            key: np.asarray(
                [row[key] for row in rows]
            )
            for key in rows[0]
        },
    )

    return csv_path, npz_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "运行97D完整路径工件偏差/摩擦鲁棒性实验"
        )
    )

    parser.add_argument(
        "--case",
        action="append",
        choices=[
            scenario["name"]
            for scenario in SCENARIOS
        ],
        help="只运行指定场景，可重复传入",
    )

    parser.add_argument(
        "--keep-traces",
        action="store_true",
        help="保留每个场景的完整NPZ时间序列",
    )

    args = parser.parse_args()

    selected = [
        scenario
        for scenario in SCENARIOS
        if (
            not args.case
            or scenario["name"]
            in args.case
        )
    ]

    if not selected:
        raise RuntimeError(
            "没有选择任何鲁棒性场景"
        )

    if not SOURCE_MJCF.exists():
        raise FileNotFoundError(
            SOURCE_MJCF
        )

    controller = load_controller()
    controller.time.sleep = lambda _: None
    controller.PRINT_INTERVAL = 1000.0

    rows = []

    print(
        "========== 97E UR5 Robustness Suite "
        "97E UR5鲁棒性验证 =========="
    )

    print(
        "Controller knowledge 控制器已知信息: "
        "97B learned path only 仅97B学习路径"
    )

    print(
        "Hidden perturbations 隐藏扰动: "
        "workpiece pose/friction 工件位姿与摩擦"
    )

    with tempfile.TemporaryDirectory(
        prefix="ur5_97e_"
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        for index, scenario in enumerate(
            selected,
            start=1,
        ):
            print(
                f"\n--- Case 场景 {index}/{len(selected)}: "
                f"{scenario['name']} ---"
            )

            model_path = (
                temporary_directory
                / f"{scenario['name']}.xml"
            )

            temporary_result = (
                temporary_directory
                / f"{scenario['name']}.npz"
            )

            build_perturbed_model(
                scenario,
                model_path,
            )

            # Stop after the normal post-scan hold and summary save.
            controller.mujoco.viewer.launch_passive = (
                lambda model, data: _HeadlessViewer(
                    model,
                    data,
                    stop_time=51.0,
                )
            )

            controller.MJCF_PATH = model_path
            controller.OUTPUT_PATH = (
                temporary_result
            )

            controller.main()

            with np.load(
                temporary_result
            ) as result:
                row = summarize(
                    scenario,
                    result,
                )

            rows.append(row)

            print(
                "Case result 场景结果: "
                + (
                    "PASS 通过"
                    if row["pass"]
                    else "FAIL 未通过"
                )
                + f" | Fmean={row['force_mean_error_n']:.3f} N"
                + f" | Fmax={row['force_max_error_n']:.3f} N"
                + f" | TCPmax={row['tcp_max_error_mm']:.3f} mm"
                + f" | tilt={row['tilt_correction_max_deg']:.3f} deg"
            )

            if args.keep_traces:
                OUTPUT_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                trace_path = (
                    OUTPUT_DIR
                    / (
                        "97E_"
                        + scenario["name"]
                        + "_trace.npz"
                    )
                )

                # A byte-for-byte copy is intentional for binary NPZ output.
                trace_path.write_bytes(
                    temporary_result.read_bytes()
                )

    csv_path, npz_path = write_summary(
        rows
    )

    passed = sum(
        row["pass"]
        for row in rows
    )

    print(
        "\n========== 97E Summary 97E汇总 =========="
    )

    print(
        f"Passed 通过: {passed}/{len(rows)}"
    )

    print(
        f"Saved 保存: {csv_path}"
    )

    print(
        f"Saved 保存: {npz_path}"
    )


if __name__ == "__main__":
    main()
