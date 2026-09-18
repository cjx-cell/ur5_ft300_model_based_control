#!/usr/bin/env python3
"""87 - Visual check for the UR3 + FT300 inspection-probe model.

Purpose
-------
1. Load the validated inspection model.
2. Hold the robot at the home configuration with torque control.
3. Visually confirm that the Robotiq gripper is gone.
4. Confirm that the inspection probe extends along tool +Z.
5. Confirm that inspection_tip is 120 mm from the FT300 tool-side frame.

Close the MuJoCo viewer window to exit.
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ur3_ft300_inspection.xml"

PRINT_INTERVAL = 1.0

# Simulation-only holding gains for visual inspection.
KP = np.array([100.0, 100.0, 80.0, 45.0, 35.0, 25.0])
KD = 2.0 * np.sqrt(KP)


def get_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Model object not found 模型对象不存在: {name}")
    return idx


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing ur3_ft300_inspection.xml. "
            "Run 86_build_inspection_probe_model.py first."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_id = get_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    ft_site_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, "ft300_site")
    tip_site_id = get_id(
        model, mujoco.mjtObj.mjOBJ_SITE, "inspection_tip_site"
    )
    tip_body_id = get_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "inspection_tip"
    )

    # Explicitly verify that the gripper is absent.
    gripper_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robotiq_85_base_link",
    )
    if gripper_id >= 0:
        raise RuntimeError("Robotiq gripper still exists 夹爪仍存在")

    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    q_ref = data.qpos[:6].copy()

    ft_position = data.site_xpos[ft_site_id].copy()
    tip_position = data.site_xpos[tip_site_id].copy()
    tip_rotation = data.xmat[tip_body_id].reshape(3, 3).copy()

    probe_vector = tip_position - ft_position
    probe_length = float(np.linalg.norm(probe_vector))
    probe_direction = probe_vector / probe_length

    # Local +Z of inspection_tip expressed in world coordinates.
    tool_z_world = tip_rotation[:, 2]
    axis_alignment = float(np.dot(probe_direction, tool_z_world))

    print("========== 87 Inspection Viewer 87检测工具可视化 ==========")
    print(
        f"Robot dimensions 机器人维度: "
        f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
    )
    print("Gripper removed 夹爪删除: PASS 通过")
    print("TCP frame TCP坐标系: inspection_tip")
    print(
        f"FT300-to-TCP distance FT300到TCP距离: "
        f"{probe_length * 1000.0:.3f} mm"
    )
    print(
        f"Tool +Z alignment 工具+Z轴一致性: "
        f"{axis_alignment:.9f}"
    )
    print(
        "Visual check 目视检查: "
        "confirm probe direction and red TCP site 确认探头方向与红色TCP点"
    )
    print("Close viewer to exit 关闭窗口结束")

    if abs(probe_length - 0.120) > 1e-6:
        raise RuntimeError(
            f"Unexpected TCP distance TCP距离错误: {probe_length:.9f} m"
        )
    if axis_alignment < 0.999999:
        raise RuntimeError(
            f"Probe is not aligned with tool +Z 探头未与工具+Z对齐: "
            f"{axis_alignment:.9f}"
        )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # A stable camera view for the complete UR3 + tool.
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.72])
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        last_print = -PRINT_INTERVAL

        while viewer.is_running():
            step_start = time.time()

            # Update bias/passive terms at the current state.
            mujoco.mj_forward(model, data)

            q = data.qpos[:6].copy()
            dq = data.qvel[:6].copy()

            # MuJoCo equation:
            #   M qdd + qfrc_bias = qfrc_actuator + qfrc_passive + ...
            #
            # Therefore the commanded actuator torque includes
            # gravity/Coriolis compensation and subtracts passive forces.
            tau = (
                data.qfrc_bias[:6]
                - data.qfrc_passive[:6]
                + KP * (q_ref - q)
                - KD * dq
            )

            if model.actuator_ctrllimited is not None:
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
                    f"TCP XYZ TCP位置(m): "
                    f"[{tip_position[0]:+.3f}, "
                    f"{tip_position[1]:+.3f}, "
                    f"{tip_position[2]:+.3f}]"
                )

            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
