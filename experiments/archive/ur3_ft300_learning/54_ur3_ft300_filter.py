#!/usr/bin/env python3

"""
54_ur3_ft300_filter.py

目标：
------------------------------------------------------------
建立完整的 FT300 信号处理链：

Raw FT300
    ↓
Payload Gravity Compensation
    ↓
Bias Compensation
    ↓
Compensated External Wrench
    ↓
模拟真实传感器测量噪声
    ↓
Low-Pass Filter
    ↓
Filtered External Wrench


本实验：

机器人始终保持在 home 姿态。

没有真实环境接触。

因此理论外力：

    W_external ≈ 0

我们人为加入测量噪声：

    W_noisy = W_external + noise

然后观察：

    noisy signal

和：

    filtered signal

之间的差异。


注意：
------------------------------------------------------------
人为加入噪声只用于学习滤波。

它不是 MuJoCo 动力学的一部分，
也不会作用到机器人上。
"""

import time

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 加载已经验证过的模型工具
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
# 3. UR3 joint / actuator names
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
# 4. 获取 UR3 joint indices
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
# 5. 获取 UR3 actuator IDs
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
# 6. 获取 FT300 sensor
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


# ============================================================
# 7. Sensor data addresses
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
# 8. FT300 measurement body
# ============================================================

sensor_body_id = int(
    mj_model.site_bodyid[
        ft300_site_id
    ]
)


# ============================================================
# 9. 判断一个 body 是否位于 FT300 下游
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
# 10. FT300 downstream payload bodies
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


payload_mass = sum(
    mj_model.body_mass[
        body_id
    ]
    for body_id
    in payload_body_ids
)


# ============================================================
# 11. Reset 到 home
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
# 12. Home position
# ============================================================

q_arm_des = np.array(
    [
        mj_data.qpos[index]
        for index
        in arm_qpos_indices
    ]
)


# ============================================================
# 13. Joint controller gains
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
# 14. Angle error
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
# 15. 保持机器人静止
# ============================================================

def hold_arm_controller():

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


    q_pin = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )


    # --------------------------------------------------------
    # Pinocchio：
    #
    # 已知机器人完整重力补偿
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


    position_error = angle_error(
        q_arm_des,
        q_arm,
    )


    tau_pd = (

        Kp_arm
        *
        position_error

        -

        Kd_arm
        *
        dq_arm

    )


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


    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = 0.0


# ============================================================
# 16. 读取 FT300 raw wrench
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
# 17. Payload gravity compensation
# ============================================================

def payload_gravity_wrench():

    # --------------------------------------------------------
    # Payload center of mass in WORLD
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
    # FT300 measurement origin
    # --------------------------------------------------------

    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Payload gravity force in WORLD
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
    # Payload gravity torque
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
    # WORLD -> FT300 frame
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
# 18. Bias calibration
# ============================================================

bias_start_time = 0.2
bias_end_time = 1.0


bias_sum = np.zeros(
    6
)

bias_count = 0

bias_wrench = None


# ============================================================
# 19. 模拟 FT300 测量噪声
# ============================================================
#
# MuJoCo 当前传感器过于理想。
#
# 为了学习滤波，我们人为模拟：
#
# Force:
#
# sigma = 0.15 N
#
#
# Torque:
#
# sigma = 0.01 N*m
#
#
# 这里使用固定随机种子，
# 所以每次实验都可以复现。
# ============================================================

rng = np.random.default_rng(
    42
)


force_noise_std = 0.15

torque_noise_std = 0.01


def add_measurement_noise(
    wrench,
):

    noise = np.concatenate([

        rng.normal(
            0.0,
            force_noise_std,
            3,
        ),

        rng.normal(
            0.0,
            torque_noise_std,
            3,
        ),

    ])


    return (
        wrench
        +
        noise
    )


# ============================================================
# 20. Low-pass filter parameters
# ============================================================
#
# MuJoCo timestep：
#
# dt = 0.001 s
#
#
# Sampling frequency：
#
# fs = 1000 Hz
#
#
# Cutoff frequency：
#
# fc = 10 Hz
#
#
# 一阶低通：
#
# tau = 1 / (2*pi*fc)
#
# alpha = tau / (tau + dt)
#
#
# y[k] =
#
# alpha * y[k-1]
#
# +
#
# (1-alpha) * x[k]
# ============================================================

dt = float(
    mj_model.opt.timestep
)


cutoff_frequency = 10.0


filter_time_constant = (
    1.0
    /
    (
        2.0
        *
        np.pi
        *
        cutoff_frequency
    )
)


alpha = (

    filter_time_constant

    /

    (
        filter_time_constant
        +
        dt
    )

)


# ============================================================
# 21. Filter state
# ============================================================

filtered_wrench = np.zeros(
    6
)

filter_initialized = False


def low_pass_filter(
    input_wrench,
):

    global filtered_wrench
    global filter_initialized


    # 第一帧：
    #
    # 直接令输出 = 输入
    #
    # 避免从全0开始造成启动瞬态。

    if not filter_initialized:

        filtered_wrench = (
            input_wrench.copy()
        )

        filter_initialized = True

        return filtered_wrench.copy()


    filtered_wrench = (

        alpha
        *
        filtered_wrench

        +

        (1.0 - alpha)
        *
        input_wrench

    )


    return filtered_wrench.copy()


# ============================================================
# 22. 统计噪声大小
# ============================================================
#
# 为了不用终端打印大量数据，
# 每秒统计一次：
#
# noisy Fx RMS
# filtered Fx RMS
#
# RMS 越小：
#
# 波动越小。
# ============================================================

noisy_fx_samples = []

filtered_fx_samples = []


# ============================================================
# 23. Simulation
# ============================================================

step = 0


print(
    f"Filter cutoff 截止频率: "
    f"{cutoff_frequency:.1f} Hz"
)


print(
    f"Filter alpha 滤波系数: "
    f"{alpha:.4f}"
)


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        # ====================================================
        # A. Hold robot
        # ====================================================

        hold_arm_controller()


        # ====================================================
        # B. Physics
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
        # D. Payload gravity
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

            bias_sum += (
                raw_wrench
                -
                gravity_wrench
            )

            bias_count += 1


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
                "Bias calibration "
                "零偏标定: completed 完成"
            )


        # ====================================================
        # F. Gravity compensated wrench
        # ====================================================

        if bias_wrench is not None:

            compensated_wrench = (

                raw_wrench

                -

                gravity_wrench

                -

                bias_wrench

            )

        else:

            compensated_wrench = np.zeros(
                6
            )


        # ====================================================
        # G. 模拟传感器噪声
        # ====================================================
        #
        # 注意：
        #
        # 这里只修改“软件测量值”。
        #
        # 不会给机械臂施加任何力。
        # ====================================================

        noisy_wrench = (
            add_measurement_noise(
                compensated_wrench
            )
        )


        # ====================================================
        # H. Low-pass filter
        # ====================================================

        filtered = (
            low_pass_filter(
                noisy_wrench
            )
        )


        # ====================================================
        # I. 收集每秒统计数据
        # ====================================================

        if current_time >= 1.0:

            noisy_fx_samples.append(
                noisy_wrench[0]
            )

            filtered_fx_samples.append(
                filtered[0]
            )


        viewer.sync()


        # ====================================================
        # J. 每秒打印一次
        # ====================================================

        if (
            step >= 2000
            and
            step % 1000 == 0
        ):

            noisy_fx_rms = np.sqrt(
                np.mean(
                    np.square(
                        noisy_fx_samples
                    )
                )
            )


            filtered_fx_rms = np.sqrt(
                np.mean(
                    np.square(
                        filtered_fx_samples
                    )
                )
            )


            print(
                "\n"
                "========== FT300 Filter FT300滤波 =========="
            )


            print(
                f"Time 时间: "
                f"{current_time:.2f} s"
            )


            print(
                f"Noisy Fx 带噪Fx: "
                f"{noisy_wrench[0]:+.3f} N"
            )


            print(
                f"Filtered Fx 滤波后Fx: "
                f"{filtered[0]:+.3f} N"
            )


            print(
                f"Noisy RMS 带噪波动: "
                f"{noisy_fx_rms:.3f} N"
            )


            print(
                f"Filtered RMS 滤波后波动: "
                f"{filtered_fx_rms:.3f} N"
            )


            # 清空统计窗口
            #
            # 下一秒重新统计。

            noisy_fx_samples.clear()

            filtered_fx_samples.clear()


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )