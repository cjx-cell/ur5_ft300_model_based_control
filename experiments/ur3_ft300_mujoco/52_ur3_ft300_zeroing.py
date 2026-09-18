#!/usr/bin/env python3

"""
52_ur3_ft300_zeroing.py

实验目的：
------------------------------------------------------------
验证：

Tare / Zeroing 清零
≠
Payload Gravity Compensation 负载重力补偿


实验流程：

0 ~ 1 s
    姿态 A 保持不动
    对 FT300 raw wrench 求平均
    得到 tare wrench

1 ~ 2 s
    继续保持姿态 A
    zeroed wrench 应该接近 0

2 ~ 6 s
    平滑改变 wrist_1 / wrist_2
    不发生任何环境接触

6 s 以后
    姿态 B 保持不动

如果 Zeroing 可以完全消除 payload gravity，
那么姿态 B 时 zeroed wrench 应该仍然为 0。

但实际不会。

因为：
    重力方向固定在 WORLD
    FT300 坐标系随机器人旋转

所以 payload gravity 在 sensor frame 中的分量会变化。
"""

import time

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 导入我们已经验证好的模型与映射工具
# ============================================================

from ur3_ft300_robotiq_pinocchio import (
    load_models,
    mujoco_to_pinocchio_q,
    pinocchio_to_mujoco_tau,
)


# ============================================================
# 2. 加载 MuJoCo + Pinocchio
# ============================================================

mj_model, pin_model = load_models()

mj_data = mujoco.MjData(
    mj_model
)

pin_data = pin_model.createData()


# ============================================================
# 3. UR3 六个关节
# ============================================================

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ============================================================
# 4. UR3 六个 motor actuator
# ============================================================

ARM_ACTUATOR_NAMES = [
    "shoulder_pan_motor",
    "shoulder_lift_motor",
    "elbow_motor",
    "wrist1_motor",
    "wrist2_motor",
    "wrist3_motor",
]


# ============================================================
# 5. 获取 joint qpos / dof index
# ============================================================

arm_qpos_indices = []

arm_dof_indices = []


for name in ARM_JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"找不到关节: {name}"
        )

    arm_qpos_indices.append(
        int(
            mj_model.jnt_qposadr[
                joint_id
            ]
        )
    )

    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                joint_id
            ]
        )
    )


# ============================================================
# 6. 获取 actuator ID
# ============================================================

arm_actuator_ids = []


for name in ARM_ACTUATOR_NAMES:

    actuator_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"找不到 actuator: {name}"
        )

    arm_actuator_ids.append(
        actuator_id
    )


# Robotiq actuator

gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)


# ============================================================
# 7. 获取 FT300 sensor
# ============================================================

force_sensor_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SENSOR,
    "ft300_force",
)


torque_sensor_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SENSOR,
    "ft300_torque",
)


if (
    force_sensor_id < 0
    or torque_sensor_id < 0
):
    raise RuntimeError(
        "找不到 FT300 force / torque sensor"
    )


# ============================================================
# 8. sensordata 起始地址
# ============================================================

force_sensor_adr = int(
    mj_model.sensor_adr[
        force_sensor_id
    ]
)


torque_sensor_adr = int(
    mj_model.sensor_adr[
        torque_sensor_id
    ]
)


# ============================================================
# 9. 重置到 home
# ============================================================

home_key_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)


if home_key_id < 0:
    raise RuntimeError(
        "找不到 home keyframe"
    )


mujoco.mj_resetDataKeyframe(
    mj_model,
    mj_data,
    home_key_id,
)


mj_data.qvel[:] = 0.0


mujoco.mj_forward(
    mj_model,
    mj_data,
)


# ============================================================
# 10. 保存姿态 A
# ============================================================

q_arm_start = np.array(
    [
        mj_data.qpos[index]
        for index
        in arm_qpos_indices
    ]
)


# ============================================================
# 11. 姿态 B
# ============================================================
#
# 这里只改变两个腕部关节：
#
# wrist_1 +30°
# wrist_2 +45°
#
# 目的不是研究关节运动本身，
#
# 而是让：
#
# FT300 sensor frame
#
# 相对于 WORLD gravity
#
# 明显改变方向。
# ============================================================

q_arm_final = (
    q_arm_start.copy()
)


q_arm_final[3] += np.deg2rad(
    30.0
)

q_arm_final[4] += np.deg2rad(
    45.0
)


# ============================================================
# 12. 时间安排
# ============================================================

# 0 ~ 1 秒：
# 收集 Tare 数据

tare_end_time = 1.0


# 1 ~ 2 秒：
# 姿态 A 清零后保持

motion_start_time = 2.0


# 2 ~ 6 秒：
# 平滑运动

motion_duration = 4.0


# ============================================================
# 13. Smoothstep
# ============================================================
#
# s:
#
# 0 -> 1
#
# s_dot:
#
# 轨迹速度
#
# 开始和结束时速度均为0。
# ============================================================

def smoothstep(t):

    if t <= motion_start_time:

        return 0.0, 0.0


    if t >= (
        motion_start_time
        +
        motion_duration
    ):

        return 1.0, 0.0


    r = (
        (t - motion_start_time)
        /
        motion_duration
    )


    s = (
        3.0 * r**2
        -
        2.0 * r**3
    )


    s_dot = (
        (6.0 * r - 6.0 * r**2)
        /
        motion_duration
    )


    return s, s_dot


# ============================================================
# 14. Desired joint trajectory
# ============================================================

def desired_arm_state(t):

    s, s_dot = smoothstep(
        t
    )


    delta_q = (
        q_arm_final
        -
        q_arm_start
    )


    q_des = (
        q_arm_start
        +
        s
        *
        delta_q
    )


    dq_des = (
        s_dot
        *
        delta_q
    )


    return (
        q_des,
        dq_des,
        s,
    )


# ============================================================
# 15. Joint controller gains
# ============================================================

Kp_arm = np.array([
    100.0,
    100.0,
    80.0,
    50.0,
    35.0,
    20.0,
])


Kd_arm = np.array([
    18.0,
    18.0,
    14.0,
    9.0,
    7.0,
    4.0,
])


# ============================================================
# 16. Angle error
# ============================================================

def angle_error(
    desired,
    current,
):

    error = (
        desired
        -
        current
    )


    return np.arctan2(
        np.sin(error),
        np.cos(error),
    )


# ============================================================
# 17. Arm trajectory controller
# ============================================================

def arm_controller():

    # --------------------------------------------------------
    # Desired q / dq
    # --------------------------------------------------------

    (
        q_des,
        dq_des,
        progress,
    ) = desired_arm_state(
        mj_data.time
    )


    # --------------------------------------------------------
    # Current q / dq
    # --------------------------------------------------------

    q_arm = np.array(
        [
            mj_data.qpos[index]
            for index
            in arm_qpos_indices
        ]
    )


    dq_arm = np.array(
        [
            mj_data.qvel[index]
            for index
            in arm_dof_indices
        ]
    )


    # --------------------------------------------------------
    # Pinocchio configuration
    # --------------------------------------------------------

    q_pin = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )


    # --------------------------------------------------------
    # Gravity compensation
    # --------------------------------------------------------
    #
    # 完整 Pinocchio 模型包含：
    #
    # UR3
    # FT300
    # Robotiq
    #
    # 因此这里已经考虑完整末端负载。
    # --------------------------------------------------------

    gravity_pin = (
        pin.computeGeneralizedGravity(
            pin_model,
            pin_data,
            q_pin,
        )
    )


    gravity_mj = (
        pinocchio_to_mujoco_tau(
            gravity_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # Joint PD
    # --------------------------------------------------------

    position_error = angle_error(
        q_des,
        q_arm,
    )


    velocity_error = (
        dq_des
        -
        dq_arm
    )


    tau_pd = (

        Kp_arm
        *
        position_error

        +

        Kd_arm
        *
        velocity_error

    )


    # --------------------------------------------------------
    # UR3 motor commands
    # --------------------------------------------------------

    for i in range(6):

        actuator_id = (
            arm_actuator_ids[i]
        )


        dof_id = (
            arm_dof_indices[i]
        )


        torque = (
            gravity_mj[dof_id]
            +
            tau_pd[i]
        )


        ctrl_min = (
            mj_model.actuator_ctrlrange[
                actuator_id,
                0
            ]
        )


        ctrl_max = (
            mj_model.actuator_ctrlrange[
                actuator_id,
                1
            ]
        )


        mj_data.ctrl[
            actuator_id
        ] = np.clip(
            torque,
            ctrl_min,
            ctrl_max,
        )


    # --------------------------------------------------------
    # Gripper 保持 home
    # --------------------------------------------------------

    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = 0.0


    # 最大关节跟踪误差
    #
    # 仅用于确认机器人确实跟上轨迹。

    max_error_deg = np.rad2deg(
        np.max(
            np.abs(
                position_error
            )
        )
    )


    return (
        progress,
        max_error_deg,
    )


# ============================================================
# 18. 读取 FT300 RAW wrench
# ============================================================

def read_ft300_raw():

    force = (
        mj_data.sensordata[
            force_sensor_adr
            :
            force_sensor_adr + 3
        ]
        .copy()
    )


    torque = (
        mj_data.sensordata[
            torque_sensor_adr
            :
            torque_sensor_adr + 3
        ]
        .copy()
    )


    wrench = np.concatenate([
        force,
        torque,
    ])


    return wrench


# ============================================================
# 19. Tare 数据
# ============================================================
#
# 我们不是只取一个瞬间值。
#
# 而是在 0.2 ~ 1.0 秒之间取平均。
#
# 现实传感器清零时也通常会：
#
# 连续采样
#     ↓
# 求平均
#     ↓
# 作为零偏
# ============================================================

tare_start_time = 0.2


tare_sum = np.zeros(
    6
)

tare_count = 0

tare_wrench = None


# ============================================================
# 20. 阶段名称
# ============================================================

def phase_name(t):

    if t < tare_end_time:

        return (
            "Taring 清零采样"
        )


    if t < motion_start_time:

        return (
            "Pose A 姿态A保持"
        )


    if t < (
        motion_start_time
        +
        motion_duration
    ):

        return (
            "Moving 换姿态中"
        )


    return (
        "Pose B 姿态B保持"
    )


# ============================================================
# 21. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        # ====================================================
        # A. 控制机器人
        # ====================================================

        (
            motion_progress,
            arm_tracking_error_deg,
        ) = arm_controller()


        # ====================================================
        # B. MuJoCo physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        # ====================================================
        # C. 读取 FT300 raw
        # ====================================================

        raw_wrench = (
            read_ft300_raw()
        )


        current_time = (
            mj_data.time
        )


        # ====================================================
        # D. 采集 Tare
        # ====================================================

        if (
            tare_start_time
            <= current_time
            <
            tare_end_time
        ):

            tare_sum += (
                raw_wrench
            )

            tare_count += 1


        # ====================================================
        # E. 完成清零
        # ====================================================

        if (
            tare_wrench is None
            and
            current_time
            >=
            tare_end_time
        ):

            if tare_count == 0:

                raise RuntimeError(
                    "没有采集到 Tare 数据"
                )


            tare_wrench = (
                tare_sum
                /
                tare_count
            )


            print(
                "\n"
                "Tare completed 清零完成"
            )


            print(
                "Tare force 清零参考力 "
                "[Fx Fy Fz] N:",
                np.round(
                    tare_wrench[:3],
                    3,
                ),
            )


            print(
                "Tare torque 清零参考力矩 "
                "[Mx My Mz] N·m:",
                np.round(
                    tare_wrench[3:],
                    4,
                ),
            )


        # ====================================================
        # F. Zeroed wrench
        # ====================================================

        if tare_wrench is None:

            zeroed_wrench = (
                np.full(
                    6,
                    np.nan,
                )
            )

        else:

            zeroed_wrench = (
                raw_wrench
                -
                tare_wrench
            )


        viewer.sync()


        # ====================================================
        # G. 精简打印
        # ============================================================
        #
        # timestep = 0.001
        #
        # 每1000步约1秒。
        #
        # 只打印：
        #
        # 时间
        # 当前阶段
        # 运动进度
        # Raw force
        # Zeroed force
        # Raw torque
        # Zeroed torque
        #
        # 不再打印一大堆控制器内部数据。
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            print(
                "\n"
                "========== FT300 Zeroing FT300清零 =========="
            )


            print(
                f"Time 时间: "
                f"{current_time:.2f} s"
            )


            print(
                f"Phase 阶段: "
                f"{phase_name(current_time)}"
            )


            print(
                f"Motion progress 换姿态进度: "
                f"{motion_progress:.2f}"
            )


            print(
                f"Arm tracking error 机械臂跟踪误差: "
                f"{arm_tracking_error_deg:.3f} deg"
            )


            print(
                "Raw force 原始力 "
                "[Fx Fy Fz] N:",
                np.round(
                    raw_wrench[:3],
                    3,
                ),
            )


            if tare_wrench is not None:

                print(
                    "Zeroed force 清零后力 "
                    "[Fx Fy Fz] N:",
                    np.round(
                        zeroed_wrench[:3],
                        3,
                    ),
                )


                print(
                    f"Zeroed force magnitude "
                    f"清零后合力: "
                    f"{np.linalg.norm(zeroed_wrench[:3]):.3f} N"
                )


                print(
                    "Zeroed torque 清零后力矩 "
                    "[Mx My Mz] N·m:",
                    np.round(
                        zeroed_wrench[3:],
                        4,
                    ),
                )


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )