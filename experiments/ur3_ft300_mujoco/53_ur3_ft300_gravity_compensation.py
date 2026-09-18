#!/usr/bin/env python3

"""
53_ur3_ft300_gravity_compensation.py

目标：
------------------------------------------------------------
验证：

Zeroing / Tare 清零
≠
Payload Gravity Compensation 负载重力补偿


实验过程：

0 ~ 1 s
    姿态 A 静止
    根据模型计算 payload gravity wrench
    用：

        Raw - Payload Gravity

    标定传感器 bias

1 ~ 2 s
    姿态 A 保持

2 ~ 6 s
    改变 wrist_1 / wrist_2 姿态

6 s 以后
    姿态 B 保持


理论上，无环境接触且机器人静止时：

External Wrench
=
Raw Wrench
-
Payload Gravity Wrench
-
Sensor Bias

应该约等于 0。


注意：
------------------------------------------------------------
本实验只补偿“负载重力”。

机器人运动过程中还有惯性力，因此 2~6 s 时：

External Wrench

不一定严格为 0。

真正重点看：

姿态 A 静止
姿态 B 静止
"""

import time

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 导入已经验证好的模型与状态映射工具
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
# 4. UR3 六个力矩执行器
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
# 5. 获取 UR3 joint qpos / dof index
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
# 6. 获取 UR3 actuator ID
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


# ============================================================
# 7. Robotiq gripper actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)


# ============================================================
# 8. 获取 FT300 force / torque sensor
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
# 9. 获取 ft300_site
# ============================================================
#
# 这就是你刚才指出缺少的部分。
#
# ft300_site：
#
# 定义 FT300 的：
#
# - 测量截面
# - 测量坐标系
# ============================================================

ft300_site_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ft300_site",
)


if ft300_site_id < 0:
    raise RuntimeError(
        "找不到 ft300_site"
    )


# ============================================================
# 10. 找到 ft300_site 所属 body
# ============================================================
#
# 当前模型应该得到：
#
# robotiq_ft_frame_id
#
#
# 也就是说：
#
# ft300_sensor
#       ↓
# robotiq_ft_frame_id
#       ↑
#     site 在这里
#       ↓
# Robotiq
# ============================================================

sensor_body_id = int(
    mj_model.site_bodyid[
        ft300_site_id
    ]
)


sensor_body_name = mujoco.mj_id2name(
    mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    sensor_body_id,
)


# ============================================================
# 11. sensordata 地址
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
# 12. 判断 body 是否属于 FT300 下游
# ============================================================
#
# 例如：
#
# robotiq_ft_frame_id
#     ↓
# robotiq_85_base_link
#     ↓
# fingers
#     ↓
# fingertips
#
# 全部属于 FT300 后面的 payload。
# ============================================================

def is_descendant(
    body_id,
    root_body_id,
):

    current = body_id

    while current > 0:

        if current == root_body_id:
            return True

        current = int(
            mj_model.body_parentid[
                current
            ]
        )

    return False


# ============================================================
# 13. 找出 FT300 下游所有 body
# ============================================================

payload_body_ids = []


for body_id in range(
    mj_model.nbody
):

    if is_descendant(
        body_id,
        sensor_body_id,
    ):

        payload_body_ids.append(
            body_id
        )


# ============================================================
# 14. 计算 payload 总质量
# ============================================================

payload_mass = 0.0


for body_id in payload_body_ids:

    payload_mass += (
        mj_model.body_mass[
            body_id
        ]
    )


print(
    "FT300 measurement body "
    "FT300测量体:",
    sensor_body_name,
)


print(
    f"Payload mass 下游负载质量: "
    f"{payload_mass:.6f} kg"
)


print(
    f"Payload weight 下游负载重量: "
    f"{payload_mass * 9.81:.3f} N"
)


# ============================================================
# 15. Reset 到 home
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
# 16. 保存姿态 A
# ============================================================

q_arm_start = np.array(
    [
        mj_data.qpos[index]
        for index
        in arm_qpos_indices
    ]
)


# ============================================================
# 17. 定义姿态 B
# ============================================================
#
# 与 52 保持一样：
#
# wrist_1 +30°
# wrist_2 +45°
#
# 目的：
#
# 改变 FT300 sensor frame
# 相对于世界重力方向的姿态。
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
# 18. 时间设置
# ============================================================

bias_start_time = 0.2

bias_end_time = 1.0


motion_start_time = 2.0

motion_duration = 4.0


# ============================================================
# 19. Smoothstep
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
# 20. Desired joint trajectory
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
# 21. Joint PD gains
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
# 22. Angle error
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
# 23. UR3 trajectory controller
# ============================================================

def arm_controller():

    # --------------------------------------------------------
    # Desired
    # --------------------------------------------------------

    (
        q_des,
        dq_des,
        progress,
    ) = desired_arm_state(
        mj_data.time
    )


    # --------------------------------------------------------
    # Current
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
    # MuJoCo q -> Pinocchio q
    # --------------------------------------------------------

    q_pin = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )


    # --------------------------------------------------------
    # 完整机器人 Gravity Compensation
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
    # 写入六个 UR3 motor
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
# 24. 读取 FT300 Raw Wrench
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


    return np.concatenate([
        force,
        torque,
    ])


# ============================================================
# 25. 计算当前姿态的 Payload Gravity Wrench
# ============================================================
#
# 这是本课核心。
#
#
# 第一步：
#
# 找到整个 Robotiq payload 的总质心
#
#
# 第二步：
#
# 根据：
#
#     F = -m g
#
# 算出 FT300 对 payload 的支撑力
#
#
# 第三步：
#
# 根据：
#
#     M = r × F
#
# 算出重力力矩
#
#
# 第四步：
#
# WORLD frame
#
#       ↓
#
# FT300 sensor frame
# ============================================================

def payload_gravity_wrench():

    # --------------------------------------------------------
    # A. Payload 总质心 WORLD
    # --------------------------------------------------------

    weighted_com_world = np.zeros(
        3
    )


    for body_id in payload_body_ids:

        mass = (
            mj_model.body_mass[
                body_id
            ]
        )


        # xipos：
        #
        # 当前 body COM
        # 在 WORLD 中的位置

        weighted_com_world += (
            mass
            *
            mj_data.xipos[
                body_id
            ]
        )


    payload_com_world = (
        weighted_com_world
        /
        payload_mass
    )


    # --------------------------------------------------------
    # B. FT300 site WORLD position
    # --------------------------------------------------------

    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    # --------------------------------------------------------
    # C. 重力
    # --------------------------------------------------------
    #
    # MuJoCo：
    #
    # gravity =
    #
    # [0, 0, -9.81]
    #
    #
    # Payload 自己受到：
    #
    # m * g
    #
    #
    # FT300 测到的支撑作用方向
    # 根据 51 的实验验证为：
    #
    # -m * g
    # --------------------------------------------------------

    gravity_world = np.array(
        mj_model.opt.gravity
    )


    force_world = (
        -payload_mass
        *
        gravity_world
    )


    # --------------------------------------------------------
    # D. 重力力矩
    # --------------------------------------------------------

    r_world = (
        payload_com_world
        -
        sensor_position_world
    )


    torque_world = np.cross(
        r_world,
        force_world,
    )


    # --------------------------------------------------------
    # E. FT300 orientation
    # --------------------------------------------------------
    #
    # site_xmat：
    #
    # Sensor LOCAL
    #
    #       ↓
    #
    # WORLD
    #
    #
    # 所以 WORLD -> Sensor：
    #
    # R.T
    # --------------------------------------------------------

    R_sensor_world = (
        mj_data.site_xmat[
            ft300_site_id
        ]
        .reshape(
            3,
            3
        )
        .copy()
    )


    force_sensor = (
        R_sensor_world.T
        @
        force_world
    )


    torque_sensor = (
        R_sensor_world.T
        @
        torque_world
    )


    return np.concatenate([
        force_sensor,
        torque_sensor,
    ])


# ============================================================
# 26. Bias 标定变量
# ============================================================
#
# 现在不再把：
#
# Raw Wrench
#
# 直接当成 Tare。
#
#
# 而是：
#
# Bias =
#
# Raw
# -
# Payload Gravity
#
#
# 在理想 MuJoCo 中，
# 这个 Bias 应该非常接近 0。
# ============================================================

bias_sum = np.zeros(
    6
)

bias_count = 0

bias_wrench = None


# ============================================================
# 27. 当前实验阶段
# ============================================================

def phase_name(t):

    if t < bias_end_time:

        return (
            "Bias calibration 零偏标定"
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
# 28. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        # ====================================================
        # A. Robot controller
        # ====================================================

        (
            motion_progress,
            arm_tracking_error_deg,
        ) = arm_controller()


        # ====================================================
        # B. Physics step
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        current_time = (
            mj_data.time
        )


        # ====================================================
        # C. Raw FT300
        # ====================================================

        raw_wrench = (
            read_ft300_raw()
        )


        # ====================================================
        # D. 当前姿态理论 Payload Gravity
        # ====================================================

        gravity_wrench = (
            payload_gravity_wrench()
        )


        # ====================================================
        # E. Bias calibration
        # ====================================================

        if (
            bias_start_time
            <= current_time
            <
            bias_end_time
        ):

            bias_sample = (
                raw_wrench
                -
                gravity_wrench
            )


            bias_sum += (
                bias_sample
            )


            bias_count += 1


        # ====================================================
        # F. Bias calibration 完成
        # ====================================================

        if (
            bias_wrench is None
            and
            current_time
            >=
            bias_end_time
        ):

            if bias_count == 0:

                raise RuntimeError(
                    "没有采集到 Bias 数据"
                )


            bias_wrench = (
                bias_sum
                /
                bias_count
            )


            print(
                "\n"
                "Bias calibration completed "
                "零偏标定完成"
            )


            print(
                "Sensor bias force "
                "传感器力零偏 "
                "[Fx Fy Fz] N:",
                np.round(
                    bias_wrench[:3],
                    5,
                ),
            )


            print(
                "Sensor bias torque "
                "传感器力矩零偏 "
                "[Mx My Mz] N·m:",
                np.round(
                    bias_wrench[3:],
                    6,
                ),
            )


        # ====================================================
        # G. External Wrench
        # ====================================================
        #
        # 最关键公式：
        #
        # W_external
        #
        # =
        #
        # W_raw
        #
        # -
        #
        # W_payload_gravity
        #
        # -
        #
        # W_sensor_bias
        # ====================================================

        if bias_wrench is not None:

            external_wrench = (

                raw_wrench

                -

                gravity_wrench

                -

                bias_wrench

            )

        else:

            external_wrench = np.full(
                6,
                np.nan,
            )


        viewer.sync()


        # ====================================================
        # H. 精简打印
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            print(
                "\n"
                "========== FT300 Gravity Compensation "
                "FT300重力补偿 =========="
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
                f"Arm tracking error "
                f"机械臂跟踪误差: "
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


            print(
                "Payload gravity "
                "负载重力 "
                "[Fx Fy Fz] N:",
                np.round(
                    gravity_wrench[:3],
                    3,
                ),
            )


            if bias_wrench is not None:

                print(
                    "External force "
                    "补偿后外力 "
                    "[Fx Fy Fz] N:",
                    np.round(
                        external_wrench[:3],
                        3,
                    ),
                )


                print(
                    f"External force magnitude "
                    f"补偿后合力: "
                    f"{np.linalg.norm(external_wrench[:3]):.4f} N"
                )


                print(
                    "External torque "
                    "补偿后外力矩 "
                    "[Mx My Mz] N·m:",
                    np.round(
                        external_wrench[3:],
                        4,
                    ),
                )


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )