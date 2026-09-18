#!/usr/bin/env python3
"""89 - Visualize the generated 3D raster path and surface normals."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent

MODEL_PATH = HERE / "ur3_ft300_inspection_scan_preview.xml"
PATH_DATA = HERE / "89_raster_scan_path.npz"

PRINT_INTERVAL = 1.0

KP = np.array([100.0, 100.0, 80.0, 45.0, 35.0, 25.0])
KD = 2.0 * np.sqrt(KP)


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing model object 模型对象不存在: {name}")
    return idx


def main():
    if not MODEL_PATH.exists() or not PATH_DATA.exists():
        raise FileNotFoundError(
            "Run 89_generate_raster_scan_path.py first."
        )

    path = np.load(PATH_DATA)

    positions = path["position"]
    normals = path["surface_normal"]
    rotations = path["rotation"]
    arc_length = path["arc_length"]
    row_ids = path["row_id"]

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_id = get_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    tip_site_id = get_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "inspection_tip_site",
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    q_ref = data.qpos[:6].copy()

    print("========== 89 Path Viewer 89轨迹预览 ==========")
    print(f"Rows 扫描行数: {len(np.unique(row_ids))}")
    print(f"Points 路径点数: {len(positions)}")
    print(f"Length 路径长度: {arc_length[-1] * 1000.0:.1f} mm")
    print("Green path 绿色轨迹: raster scan 往复式扫描路径")
    print("Blue lines 蓝色短线: outward normals 曲面外法向")
    print("Tool +Z 工具+Z: opposite to blue normal 与蓝色法向相反")
    print("Green sphere 绿色球: start 起点")
    print("Red sphere 红色球: end 终点")
    print("Close viewer to exit 关闭窗口结束")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.16, -0.24, 0.76])
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -28.0

        last_print = -PRINT_INTERVAL

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_forward(model, data)

            q = data.qpos[:6].copy()
            dq = data.qvel[:6].copy()

            tau = (
                data.qfrc_bias[:6]
                - data.qfrc_passive[:6]
                + KP * (q_ref - q)
                - KD * dq
            )

            for i in range(6):
                if model.actuator_ctrllimited[i]:
                    low, high = model.actuator_ctrlrange[i]
                    tau[i] = np.clip(tau[i], low, high)

            data.ctrl[:6] = tau
            mujoco.mj_step(model, data)

            if data.time - last_print >= PRINT_INTERVAL:
                last_print = data.time
                tcp = data.site_xpos[tip_site_id].copy()

                print(
                    f"Time 时间: {data.time:5.1f} s | "
                    f"TCP Z: {tcp[2]:.3f} m"
                )

            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
