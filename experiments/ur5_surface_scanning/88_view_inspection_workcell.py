#!/usr/bin/env python3
"""88 - Visual inspection of the industrial surface-scanning workcell."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ur3_ft300_inspection_workcell.xml"

PRINT_INTERVAL = 1.0

KP = np.array([100.0, 100.0, 80.0, 45.0, 35.0, 25.0])
KD = 2.0 * np.sqrt(KP)


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing model object 模型对象不存在: {name}")
    return idx


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing ur3_ft300_inspection_workcell.xml. "
            "Run 88_build_inspection_workcell.py first."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_id = get_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    tip_site_id = get_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    workpiece_geom_id = get_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "inspection_workpiece"
    )

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    q_ref = data.qpos[:6].copy()

    tip_position = data.site_xpos[tip_site_id].copy()
    workpiece_center = data.geom_xpos[workpiece_geom_id].copy()
    workpiece_top_z = workpiece_center[2] + model.geom_size[workpiece_geom_id, 2]
    clearance = tip_position[2] - workpiece_top_z

    print("========== 88 Workcell Viewer 88场景可视化 ==========")
    print("Robot 机器人: UR3 + FT300 + inspection probe")
    print("Workpiece 工件: double-curvature ellipsoid 双曲率椭球曲面")
    print(
        f"Initial TCP clearance 初始TCP离工件顶部: "
        f"{clearance * 1000.0:.1f} mm"
    )
    print("Green points 绿色点: planned scan-area corners 计划扫描区域四角")
    print("Close viewer to exit 关闭窗口结束")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = np.array([0.08, -0.20, 0.66])
        viewer.cam.distance = 1.75
        viewer.cam.azimuth = 142.0
        viewer.cam.elevation = -24.0

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
                tip_position = data.site_xpos[tip_site_id].copy()
                print(
                    f"Time 时间: {data.time:5.1f} s | "
                    f"TCP Z: {tip_position[2]:.3f} m"
                )

            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
