#!/usr/bin/env python3

"""
51_ur3_ft300_raw_sensor.py

目标：
------------------------------------------------------------
1. 使用正式模型：
   UR3 + FT300 + Robotiq 2F-85

2. 不与环境接触。

3. 使用 Pinocchio 重力补偿让 UR3 保持静止。

4. 读取 MuJoCo FT300：
   [Fx, Fy, Fz, Mx, My, Mz]

5. 根据 FT300 后方所有负载的：
   - 总质量
   - 总质心
   - 重力

   自动计算理论 FT300 wrench。

6. 比较：
   FT300 raw measurement
   vs
   theoretical payload wrench
"""

import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 导入我们已经验证过的 Pinocchio / MuJoCo 映射工具
# ============================================================

from ur3_ft300_robotiq_pinocchio import (
    load_models,
    mujoco_to_pinocchio_q,
    pinocchio_to_mujoco_tau,
)


# ============================================================
# 2. 文件路径
# ============================================================

HERE = Path(__file__).resolve().parent

MJCF_PATH = (
    HERE
    /
    "ur3_ft300_robotiq_force_control.xml"
)


# ============================================================
# 3. 加载模型
# ============================================================
#
# load_models() 会：
#
# - 加载 MuJoCo MJCF
# - 加载 Pinocchio URDF
# - 同步 armature
# - 同步 damping
# - 同步 friction
#
# 这套模型已经通过：
#
# M(q)
# h(q, dq)
#
# 的数值一致性测试。
# ============================================================

mj_model, pin_model = load_models()

mj_data = mujoco.MjData(
    mj_model
)

pin_data = pin_model.createData()


# ============================================================
# 4. UR3 六个关节与六个 actuator
# ============================================================

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


ARM_ACTUATOR_NAMES = [
    "shoulder_pan_motor",
    "shoulder_lift_motor",
    "elbow_motor",
    "wrist1_motor",
    "wrist2_motor",
    "wrist3_motor",
]


# ============================================================
# 5. 获取关节 ID
# ============================================================

arm_joint_ids = []

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

    arm_joint_ids.append(
        joint_id
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


# Robotiq 位置执行器

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


ft300_site_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ft300_site",
)


if (
    force_sensor_id < 0
    or torque_sensor_id < 0
    or ft300_site_id < 0
):
    raise RuntimeError(
        "找不到 FT300 sensor/site"
    )


# sensor 在 mj_data.sensordata 中的起始位置

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
# 8. FT300 site 所在 body
# ============================================================
#
# 当前应该是：
#
# robotiq_ft_frame_id
#
# 它是 FT300 工具侧的测量截面。
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


print(
    "FT300 measurement body FT300测量体:",
    sensor_body_name,
)


# ============================================================
# 9. 判断某 body 是否属于 FT300 下游
# ============================================================
#
# 我们需要找到：
#
# robotiq_ft_frame_id
#       ↓
# 所有子孙 body
#
# 这些 body 的质量总和就是：
#
# FT300 后面的 payload。
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
# 10. 找出 FT300 后面的所有 body
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
# 11. 计算下游 payload 总质量
# ============================================================

payload_mass = 0.0


for body_id in payload_body_ids:

    payload_mass += (
        mj_model.body_mass[
            body_id
        ]
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
# 12. 使用 MuJoCo home keyframe
# ============================================================
#
# 你的 MJCF 已经有：
#
# <key name="home" ... />
#
# 所以直接使用它。
# ============================================================

home_key_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)


if home_key_id < 0:
    raise RuntimeError(
        "模型中找不到 home keyframe"
    )


mujoco.mj_resetDataKeyframe(
    mj_model,
    mj_data,
    home_key_id,
)


# 初始速度清零

mj_data.qvel[:] = 0.0


# 先做一次 forward

mujoco.mj_forward(
    mj_model,
    mj_data,
)


# ============================================================
# 13. 保存当前 UR3 姿态作为目标姿态
# ============================================================

q_arm_des = np.array(
    [
        mj_data.qpos[index]
        for index
        in arm_qpos_indices
    ]
)


# ============================================================
# 14. Joint holding gains
# ============================================================
#
# 这里只是让机器人稳定保持 home。
#
# 不是这一课的研究重点。
#
# 核心仍然是：
#
# gravity compensation
# +
# 较温和的 joint PD
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
# 15. 角度误差
# ============================================================
#
# 避免 continuous joint 在 ±pi 附近出现：
#
# 2*pi
#
# 的假误差。
# ============================================================

def angle_error(
    desired,
    current,
):

    delta = (
        desired
        -
        current
    )

    return np.arctan2(
        np.sin(delta),
        np.cos(delta),
    )


# ============================================================
# 16. UR3 保持控制器
# ============================================================

def hold_arm_controller():

    # --------------------------------------------------------
    # MuJoCo -> Pinocchio q
    # --------------------------------------------------------

    q_pin = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )


    # --------------------------------------------------------
    # 完整机器人重力补偿
    # --------------------------------------------------------
    #
    # 注意：
    #
    # 这里的 Pinocchio 模型已经包含：
    #
    # UR3
    # FT300
    # Robotiq
    #
    # 所以算出的 gravity torque
    # 自动包含夹爪负载。
    # --------------------------------------------------------

    gravity_pin = (
        pin.computeGeneralizedGravity(
            pin_model,
            pin_data,
            q_pin,
        )
    )


    # 转换成 MuJoCo generalized-force 顺序

    gravity_mj = (
        pinocchio_to_mujoco_tau(
            gravity_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # 当前六个 UR3 joint
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
    # Joint PD
    # --------------------------------------------------------

    error = angle_error(
        q_arm_des,
        q_arm,
    )


    tau_pd = (
        Kp_arm
        *
        error

        -

        Kd_arm
        *
        dq_arm
    )


    # --------------------------------------------------------
    # 写 actuator control
    # --------------------------------------------------------

    for i in range(6):

        dof_id = (
            arm_dof_indices[i]
        )

        actuator_id = (
            arm_actuator_ids[i]
        )


        # gravity compensation
        #
        # +
        #
        # joint holding PD

        torque = (
            gravity_mj[dof_id]
            +
            tau_pd[i]
        )


        # actuator torque limit

        lower = (
            mj_model
            .actuator_ctrlrange[
                actuator_id,
                0
            ]
        )

        upper = (
            mj_model
            .actuator_ctrlrange[
                actuator_id,
                1
            ]
        )


        mj_data.ctrl[
            actuator_id
        ] = np.clip(
            torque,
            lower,
            upper,
        )


    # --------------------------------------------------------
    # Robotiq 保持 home
    # --------------------------------------------------------

    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = 0.0


    return error


# ============================================================
# 17. 读取 FT300 raw wrench
# ============================================================
#
# 输出坐标系：
#
# ft300_site LOCAL frame
#
#
# Force:
#
# [Fx Fy Fz]
#
#
# Torque:
#
# [Mx My Mz]
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


    return force, torque


# ============================================================
# 18. 计算理论 payload wrench
# ============================================================
#
# 静止状态：
#
# FT300 必须托住整个下游 payload。
#
#
# WORLD 中：
#
# gravity =
#
#     m * g
#
#
# FT300 对 payload 的支撑力：
#
#     F_sensor =
#
#     -m * g
#
#
# Torque:
#
#     M =
#
#     r_com × F_sensor
#
#
# 最后再把 WORLD wrench
# 转到 FT300 LOCAL frame。
# ============================================================

def theoretical_payload_wrench():

    # --------------------------------------------------------
    # A. payload COM in WORLD
    # --------------------------------------------------------

    weighted_position = np.zeros(
        3
    )


    for body_id in payload_body_ids:

        mass = (
            mj_model.body_mass[
                body_id
            ]
        )

        # xipos =
        #
        # body inertial-frame origin
        #
        # 即该 body COM 的 world position。

        weighted_position += (
            mass
            *
            mj_data.xipos[
                body_id
            ]
        )


    payload_com_world = (
        weighted_position
        /
        payload_mass
    )


    # --------------------------------------------------------
    # B. FT300 site position
    # --------------------------------------------------------

    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    # --------------------------------------------------------
    # C. 理论支撑力 WORLD
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
    # D. 理论支撑力矩 WORLD
    # --------------------------------------------------------

    r = (
        payload_com_world
        -
        sensor_position_world
    )


    torque_world = np.cross(
        r,
        force_world,
    )


    # --------------------------------------------------------
    # E. WORLD -> FT300 sensor frame
    # --------------------------------------------------------
    #
    # site_xmat：
    #
    # LOCAL -> WORLD
    #
    # 所以：
    #
    # WORLD -> LOCAL
    #
    # 用 transpose。
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


    force_local = (
        R_sensor_world.T
        @
        force_world
    )


    torque_local = (
        R_sensor_world.T
        @
        torque_world
    )


    return (
        force_local,
        torque_local,
    )


# ============================================================
# 19. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        # ----------------------------------------------------
        # 保持机器人静止
        # ----------------------------------------------------

        arm_error = (
            hold_arm_controller()
        )


        # ----------------------------------------------------
        # MuJoCo physics
        # ----------------------------------------------------

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        # ----------------------------------------------------
        # 读取真实 FT300
        # ----------------------------------------------------

        (
            force_raw,
            torque_raw,
        ) = read_ft300_raw()


        # ----------------------------------------------------
        # 计算理论 FT300
        # ----------------------------------------------------

        (
            force_expected,
            torque_expected,
        ) = theoretical_payload_wrench()


        viewer.sync()


        # ====================================================
        # 20. 精简打印
        # ============================================================
        #
        # timestep = 0.001 s
        #
        # 1000 steps ≈ 1 second
        #
        # 前1秒不打印，
        # 给系统一点稳定时间。
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            max_arm_error_deg = (
                np.rad2deg(
                    np.max(
                        np.abs(
                            arm_error
                        )
                    )
                )
            )


            force_error = (
                np.linalg.norm(
                    force_raw
                    -
                    force_expected
                )
            )


            torque_error = (
                np.linalg.norm(
                    torque_raw
                    -
                    torque_expected
                )
            )


            print(
                "\n"
                "========== FT300 Raw FT300原始读数 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Arm hold error 机械臂保持误差: "
                f"{max_arm_error_deg:.4f} deg"
            )


            print(
                "Raw force 原始力 "
                "[Fx Fy Fz] N:",
                np.round(
                    force_raw,
                    3,
                ),
            )


            print(
                "Expected force 理论力 "
                "[Fx Fy Fz] N:",
                np.round(
                    force_expected,
                    3,
                ),
            )


            print(
                f"Force error 力误差: "
                f"{force_error:.4f} N"
            )


            print(
                "Raw torque 原始力矩 "
                "[Mx My Mz] N·m:",
                np.round(
                    torque_raw,
                    4,
                ),
            )


            print(
                "Expected torque 理论力矩 "
                "[Mx My Mz] N·m:",
                np.round(
                    torque_expected,
                    4,
                ),
            )


            print(
                f"Torque error 力矩误差: "
                f"{torque_error:.5f} N·m"
            )


        step += 1


        # 只控制 viewer 的现实播放速度。
        #
        # 不影响 MuJoCo 数值积分。

        time.sleep(
            mj_model.opt.timestep
        )