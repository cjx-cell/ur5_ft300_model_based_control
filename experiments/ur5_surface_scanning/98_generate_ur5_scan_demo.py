#!/usr/bin/env python3
"""Render the recorded 97D scan as README-ready MP4 and GIF media."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

# Select a headless backend before importing MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MODEL_PATH = HERE / "ur5_ft300_inspection_workcell.xml"
RESULT_PATH = HERE / "97D_learned_surface_force_scan_result.npz"
MEDIA_DIR = ROOT / "docs" / "media"

MP4_PATH = MEDIA_DIR / "ur5_learned_surface_5n_scan.mp4"
GIF_PATH = MEDIA_DIR / "ur5_learned_surface_5n_scan.gif"

WIDTH = 640
HEIGHT = 360
FPS = 16
FRAME_COUNT = 240


def overlay_status(
    frame_bgr,
    progress,
    desired_force,
    measured_force,
):
    cv2.rectangle(
        frame_bgr,
        (12, 12),
        (330, 93),
        (20, 24, 30),
        thickness=-1,
    )

    cv2.putText(
        frame_bgr,
        "UR5 learned-surface 5 N scan",
        (24, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame_bgr,
        (
            f"progress {progress:5.1f}%   "
            f"force {measured_force:4.2f}/{desired_force:4.2f} N"
        ),
        (24, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (225, 232, 240),
        1,
        cv2.LINE_AA,
    )

    bar_left = 24
    bar_top = 74
    bar_width = 285

    cv2.rectangle(
        frame_bgr,
        (bar_left, bar_top),
        (bar_left + bar_width, bar_top + 8),
        (75, 82, 90),
        thickness=-1,
    )

    cv2.rectangle(
        frame_bgr,
        (bar_left, bar_top),
        (
            bar_left
            + int(
                bar_width
                * np.clip(progress, 0.0, 100.0)
                / 100.0
            ),
            bar_top + 8,
        ),
        (55, 205, 110),
        thickness=-1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="生成97D扫描演示MP4/GIF"
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=FRAME_COUNT,
        help="输出帧数，默认240",
    )

    args = parser.parse_args()

    for path in [
        MODEL_PATH,
        RESULT_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(
        RESULT_PATH
    ) as result:
        required = {
            "qpos",
            "progress_percent",
            "desired_force",
            "measured_normal_force",
        }

        missing = required - set(
            result.files
        )

        if missing:
            raise RuntimeError(
                "97D结果缺少渲染字段，请重新运行97D: "
                f"{sorted(missing)}"
            )

        qpos = result["qpos"].copy()
        progress = result[
            "progress_percent"
        ].copy()
        desired_force = result[
            "desired_force"
        ].copy()
        measured_force = result[
            "measured_normal_force"
        ].copy()

    indices = np.linspace(
        0,
        len(qpos) - 1,
        args.frames,
        dtype=int,
    )

    MEDIA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model,
        height=HEIGHT,
        width=WIDTH,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array(
        [0.02, -0.18, 0.91]
    )
    camera.distance = 1.78
    camera.azimuth = 142.0
    camera.elevation = -24.0

    writer = cv2.VideoWriter(
        str(MP4_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV无法创建MP4输出"
        )

    try:
        for index in indices:
            data.qpos[:] = qpos[index]
            data.qvel[:] = 0.0

            mujoco.mj_forward(
                model,
                data,
            )

            renderer.update_scene(
                data,
                camera=camera,
            )

            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(
                frame_rgb,
                cv2.COLOR_RGB2BGR,
            )

            overlay_status(
                frame_bgr,
                float(progress[index]),
                float(desired_force[index]),
                float(measured_force[index]),
            )

            writer.write(
                frame_bgr
            )

    finally:
        writer.release()
        renderer.close()

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(MP4_PATH),
            "-filter_complex",
            (
                "fps=12,scale=480:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=96[p];"
                "[s1][p]paletteuse=dither=bayer"
            ),
            str(GIF_PATH),
        ],
        check=True,
    )

    print(
        f"Saved 保存: {MP4_PATH}"
    )

    print(
        f"Saved 保存: {GIF_PATH}"
    )


if __name__ == "__main__":
    main()
