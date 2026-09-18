#!/usr/bin/env python3

"""
72_ur3_tcp_free_space_pivot_diagnostic.py


自由空间 TCP Pivot + Passive Force Compensation
自由空间 TCP 旋转 + MuJoCo 被动力补偿诊断


============================================================
实验目的
============================================================

当前我们已经排除了：

1. 曲面接触导致误差
2. FT300 力反馈导致误差
3. Jacobian 奇异
4. 电机力矩饱和
5. 线加速度限幅
6. 原先过小的姿态加速度限幅


现在重点验证：

    MuJoCo qfrc_passive

是否造成动态跟踪误差。


============================================================
动力学关系
============================================================

Pinocchio RNEA 给出：

    tau_rnea
        =
    M(q) qdd_des
    +
    h(q, qdot)


但是 MuJoCo 实际动力学中还有：

    qfrc_passive


例如：

    joint damping
    spring
    tendon passive force
    等


因此如果：

    qfrc_passive < 0

表示被动力正在阻碍当前运动。


为了让总广义力仍然达到：

    tau_rnea

电机应该输出：

    tau_motor
        =
    tau_rnea
        -
    qfrc_passive


例如：

    tau_rnea = 10 Nm
    qfrc_passive = -2 Nm

那么：

    tau_motor
        =
    10 - (-2)
        =
    12 Nm

MuJoCo 总效果：

    12 + (-2)
        =
    10 Nm


============================================================
实验运动
============================================================

没有墙。
没有接触。
没有外力控制。

绿色虚拟 TCP：

    固定在空间位置

120 mm 工具：

    围绕 TCP 转动


姿态轨迹与之前曲面实验完全相同：

5 ~ 13 s

    0 deg
        ->
    负角度
        ->
    0 deg
        ->
    正角度
        ->
    0 deg


重点观察：

    8 s
    9 s
    10 s
    11 s


尤其：

    Tool error
    TCP error
    Passive torque


============================================================
当前限幅
============================================================

linear acceleration:

    ±30 m/s²

angular acceleration:

    ±80 rad/s²


这两个值目前用于仿真诊断，
不是实际 UR3 的工程安全参数。
"""


import time

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


from ur3_ft300_robotiq_pinocchio import (
    load_models,
    mujoco_to_pinocchio_q,
    mujoco_to_pinocchio_v,
    pinocchio_to_mujoco_tau,
)


# ============================================================
# 1. 基础方向
# ============================================================

base_surface_angle_deg = (
    20.0
)


base_surface_angle = np.deg2rad(
    base_surface_angle_deg
)


base_normal_world = np.array([
    np.cos(base_surface_angle),
    np.sin(base_surface_angle),
    0.0,
])


# ============================================================
# 2. 虚拟曲面参数
#
# 注意：
#
# 当前没有真正创建曲面。
#
# 这里只使用之前曲面的数学公式生成
# 完全相同的姿态轨迹。
# ============================================================

curve_amplitude = (
    0.004
)


curve_length = (
    0.030
)


# ============================================================
# 3. 虚拟曲面高度 / 斜率
# ============================================================

def surface_height(
    s,
):

    if s <= 0.0:

        return (
            0.0,
            0.0,
        )


    if s >= curve_length:

        return (
            0.0,
            0.0,
        )


    phase = (
        2.0
        *
        np.pi
        *
        s
        /
        curve_length
    )


    height = (
        0.5
        *
        curve_amplitude
        *
        (
            1.0
            -
            np.cos(
                phase
            )
        )
    )


    slope = (
        curve_amplitude
        *
        np.pi
        /
        curve_length
        *
        np.sin(
            phase
        )
    )


    return (
        height,
        slope,
    )


# ============================================================
# 4. 工具目标偏转角
#
# 曲面法向：
#
# n(s)
# ∝
# n0 - h'(s)t0
#
#
# 因此相对初始方向：
#
# theta(s)
# =
# -atan(h'(s))
# ============================================================

def normal_angle(
    s,
):

    (
        _,
        slope,
    ) = surface_height(
        s
    )


    return (
        -np.arctan(
            slope
        )
    )


# ============================================================
# 5. theta(s) 的一阶 / 二阶导数
# ============================================================

def normal_angle_derivatives(
    s,
):

    if (
        s <= 0.0
        or
        s >= curve_length
    ):

        return (
            0.0,
            0.0,
            0.0,
        )


    # --------------------------------------------------------
    # phase = k*s
    # --------------------------------------------------------

    k = (
        2.0
        *
        np.pi
        /
        curve_length
    )


    # --------------------------------------------------------
    # h'(s)
    #
    # =
    #
    # c sin(k s)
    # --------------------------------------------------------

    c = (
        curve_amplitude
        *
        np.pi
        /
        curve_length
    )


    phase = (
        k
        *
        s
    )


    slope = (
        c
        *
        np.sin(
            phase
        )
    )


    slope_s = (
        c
        *
        k
        *
        np.cos(
            phase
        )
    )


    slope_ss = (
        -c
        *
        k
        *
        k
        *
        np.sin(
            phase
        )
    )


    # --------------------------------------------------------
    # theta
    # --------------------------------------------------------

    theta = (
        -np.arctan(
            slope
        )
    )


    # --------------------------------------------------------
    # dtheta/ds
    # --------------------------------------------------------

    theta_s = (
        -slope_s

        /

        (
            1.0
            +
            slope
            *
            slope
        )
    )


    # --------------------------------------------------------
    # d²theta/ds²
    # --------------------------------------------------------

    theta_ss = (
        -(
            slope_ss
            *
            (
                1.0
                +
                slope
                *
                slope
            )

            -

            2.0
            *
            slope
            *
            slope_s
            *
            slope_s
        )

        /

        (
            (
                1.0
                +
                slope
                *
                slope
            )
            **
            2
        )
    )


    return (
        theta,
        theta_s,
        theta_ss,
    )


# ============================================================
# 6. 数学工具
# ============================================================

def rotation_z(
    angle,
):

    c = np.cos(
        angle
    )


    s = np.sin(
        angle
    )


    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def skew(
    vector,
):

    return np.array([
        [
            0.0,
            -vector[2],
            vector[1],
        ],

        [
            vector[2],
            0.0,
            -vector[0],
        ],

        [
            -vector[1],
            vector[0],
            0.0,
        ],
    ])


# ============================================================
# 7. Smoothstep
# ============================================================

def smoothstep(
    t,
    start_time,
    duration,
):

    if t <= start_time:

        return (
            0.0,
            0.0,
            0.0,
        )


    if t >= (
        start_time
        +
        duration
    ):

        return (
            1.0,
            0.0,
            0.0,
        )


    ratio = (
        (t - start_time)
        /
        duration
    )


    progress = (
        3.0
        *
        ratio**2

        -

        2.0
        *
        ratio**3
    )


    progress_dot = (
        (
            6.0
            *
            ratio

            -

            6.0
            *
            ratio**2
        )

        /

        duration
    )


    progress_ddot = (
        (
            6.0

            -

            12.0
            *
            ratio
        )

        /

        duration**2
    )


    return (
        progress,
        progress_dot,
        progress_ddot,
    )


# ============================================================
# 8. 与曲面实验完全相同的时间轨迹
# ============================================================

motion_start_time = (
    5.0
)


motion_duration = (
    8.0
)


def desired_curve_parameter(
    t,
):

    (
        progress,
        progress_dot,
        progress_ddot,
    ) = smoothstep(
        t,
        motion_start_time,
        motion_duration,
    )


    curve_s = (
        curve_length
        *
        progress
    )


    curve_s_dot = (
        curve_length
        *
        progress_dot
    )


    curve_s_ddot = (
        curve_length
        *
        progress_ddot
    )


    return (
        curve_s,
        curve_s_dot,
        curve_s_ddot,
    )


# ============================================================
# 9. 加载 MuJoCo + Pinocchio 正式模型
# ============================================================

mj_model, pin_model = (
    load_models()
)


mj_data = mujoco.MjData(
    mj_model
)


pin_data = (
    pin_model.createData()
)


dt = float(
    mj_model.opt.timestep
)


# ============================================================
# 10. Reset
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


mj_data.qvel[:] = (
    0.0
)


mujoco.mj_forward(
    mj_model,
    mj_data,
)


# ============================================================
# 11. UR3 joint / actuator mapping
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


arm_dof_indices = []

arm_pin_v_indices = []

arm_actuator_ids = []


# ------------------------------------------------------------
# Joint mapping
# ------------------------------------------------------------

for joint_name in ARM_JOINT_NAMES:

    mj_joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )


    if mj_joint_id < 0:

        raise RuntimeError(
            f"MuJoCo 找不到关节: "
            f"{joint_name}"
        )


    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        )
    )


    pin_joint_id = (
        pin_model.getJointId(
            joint_name
        )
    )


    if pin_joint_id == 0:

        raise RuntimeError(
            f"Pinocchio 找不到关节: "
            f"{joint_name}"
        )


    arm_pin_v_indices.append(
        int(
            pin_model.idx_vs[
                pin_joint_id
            ]
        )
    )


# ------------------------------------------------------------
# Actuator mapping
# ------------------------------------------------------------

for actuator_name in ARM_ACTUATOR_NAMES:

    actuator_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )


    if actuator_id < 0:

        raise RuntimeError(
            f"找不到电机: "
            f"{actuator_name}"
        )


    arm_actuator_ids.append(
        actuator_id
    )


gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)


# ============================================================
# 12. Pinocchio task frame
# ============================================================

TASK_FRAME_NAME = (
    "robotiq_ft_frame_id"
)


task_frame_id = (
    pin_model.getFrameId(
        TASK_FRAME_NAME
    )
)


if task_frame_id >= pin_model.nframes:

    raise RuntimeError(
        f"Pinocchio 找不到 frame: "
        f"{TASK_FRAME_NAME}"
    )


# ============================================================
# 13. FT300 site
#
# 本实验不读取 FT300 力。
#
# 这里只用它确定虚拟工具的初始位置。
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
# 14. 初始 Pinocchio pose
# ============================================================

q_pin_initial = (
    mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )
)


pin.forwardKinematics(
    pin_model,
    pin_data,
    q_pin_initial,
)


pin.updateFramePlacements(
    pin_model,
    pin_data,
)


initial_pose = (
    pin_data.oMf[
        task_frame_id
    ]
)


p_task_initial = (
    initial_pose.translation.copy()
)


R_initial = (
    initial_pose.rotation.copy()
)


# ============================================================
# 15. 创建 120 mm 虚拟 TCP
# ============================================================

tool_length = (
    0.120
)


sensor_position_initial = (
    mj_data.site_xpos[
        ft300_site_id
    ]
    .copy()
)


tcp_position_initial = (
    sensor_position_initial

    +

    tool_length
    *
    base_normal_world
)


# ------------------------------------------------------------
# TCP offset expressed in task frame
#
# r_local
#
# =
#
# R_initial^T
# *
# (p_TCP - p_task)
# ------------------------------------------------------------

tcp_offset_task_local = (
    R_initial.T

    @

    (
        tcp_position_initial

        -

        p_task_initial
    )
)


# ------------------------------------------------------------
# 工具轴在 task frame 中的固定方向
# ------------------------------------------------------------

tool_axis_task_local = (
    R_initial.T

    @

    base_normal_world
)


tool_axis_task_local /= np.linalg.norm(
    tool_axis_task_local
)


# ============================================================
# 16. Cartesian controller gains
# ============================================================

Kp_position = np.array([
    225.0,
    225.0,
    225.0,
])


Kd_position = np.array([
    30.0,
    30.0,
    30.0,
])


Kp_orientation = np.array([
    225.0,
    225.0,
    225.0,
])


Kd_orientation = np.array([
    30.0,
    30.0,
    30.0,
])


# ============================================================
# 17. 当前诊断使用的任务空间加速度限制
# ============================================================

linear_acceleration_limit = (
    30.0
)


angular_acceleration_limit = (
    80.0
)


# ============================================================
# 18. TCP Controller
# ============================================================

def tcp_controller(
    desired_rotation,
    desired_angular_velocity,
    desired_angular_acceleration,
):

    # ========================================================
    # A. MuJoCo -> Pinocchio state
    # ========================================================

    q_pin = (
        mujoco_to_pinocchio_q(
            mj_data.qpos,
            mj_model,
            pin_model,
        )
    )


    v_pin = (
        mujoco_to_pinocchio_v(
            mj_data.qvel,
            mj_model,
            pin_model,
        )
    )


    # ========================================================
    # B. Forward kinematics
    # ========================================================

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
    )


    pin.computeJointJacobians(
        pin_model,
        pin_data,
        q_pin,
    )


    pin.computeJointJacobiansTimeVariation(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
    )


    pin.updateFramePlacements(
        pin_model,
        pin_data,
    )


    task_pose = (
        pin_data.oMf[
            task_frame_id
        ]
    )


    p_task = (
        task_pose.translation.copy()
    )


    R_task = (
        task_pose.rotation.copy()
    )


    # ========================================================
    # C. 当前 TCP 位置
    # ========================================================
    #
    # r_world
    #
    # =
    #
    # R_task * r_local
    # ========================================================

    r_world = (
        R_task

        @

        tcp_offset_task_local
    )


    current_tcp_position = (
        p_task

        +

        r_world
    )


    # ========================================================
    # D. Task-frame Jacobian
    # ========================================================

    J_task = (
        pin.getFrameJacobian(
            pin_model,
            pin_data,
            task_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
    )


    dJ_task = (
        pin.getFrameJacobianTimeVariation(
            pin_model,
            pin_data,
            task_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
    )


    Jv_task = (
        J_task[
            :3,
            :
        ]
    )


    Jw_task = (
        J_task[
            3:,
            :
        ]
    )


    # ========================================================
    # E. Shift Jacobian from task origin to TCP
    # ========================================================
    #
    # v_TCP
    #
    # =
    #
    # v_task
    #
    # +
    #
    # omega x r
    #
    #
    # omega x r
    #
    # =
    #
    # -[r]x omega
    #
    #
    # 所以：
    #
    # Jv_TCP
    #
    # =
    #
    # Jv_task
    #
    # -
    #
    # [r]x Jw
    # ========================================================

    Jv_tcp = (
        Jv_task

        -

        skew(
            r_world
        )

        @

        Jw_task
    )


    J_tcp = np.vstack([
        Jv_tcp,
        Jw_task,
    ])


    # ========================================================
    # F. TCP Jacobian time derivative
    # ========================================================

    task_twist = (
        J_task

        @

        v_pin
    )


    omega_world = (
        task_twist[
            3:
        ]
    )


    r_dot_world = np.cross(
        omega_world,
        r_world,
    )


    dJv_tcp = (
        dJ_task[
            :3,
            :
        ]

        -

        skew(
            r_dot_world
        )

        @

        Jw_task

        -

        skew(
            r_world
        )

        @

        dJ_task[
            3:,
            :
        ]
    )


    dJ_tcp = np.vstack([
        dJv_tcp,
        dJ_task[
            3:,
            :
        ],
    ])


    # ========================================================
    # G. 当前 TCP twist
    # ========================================================

    tcp_twist = (
        J_tcp

        @

        v_pin
    )


    current_linear_velocity = (
        tcp_twist[
            :3
        ]
    )


    current_angular_velocity = (
        tcp_twist[
            3:
        ]
    )


    # ========================================================
    # H. TCP position task
    #
    # TCP 必须始终保持固定。
    # ========================================================

    position_error = (
        tcp_position_initial

        -

        current_tcp_position
    )


    desired_linear_velocity = (
        np.zeros(
            3
        )
    )


    desired_linear_acceleration = (
        np.zeros(
            3
        )
    )


    # ========================================================
    # I. Orientation error
    # ========================================================

    R_error = (
        desired_rotation

        @

        R_task.T
    )


    orientation_error = (
        pin.log3(
            R_error
        )
    )


    # ========================================================
    # J. Raw linear acceleration command
    # ========================================================

    linear_acceleration_raw = (
        desired_linear_acceleration

        +

        Kp_position
        *
        position_error

        +

        Kd_position
        *
        (
            desired_linear_velocity

            -

            current_linear_velocity
        )
    )


    # --------------------------------------------------------
    # 记录限幅前的最大需求
    # --------------------------------------------------------

    linear_acceleration_demand = float(
        np.max(
            np.abs(
                linear_acceleration_raw
            )
        )
    )


    linear_clip_active = bool(
        linear_acceleration_demand

        >

        linear_acceleration_limit
    )


    # ========================================================
    # K. Raw angular acceleration command
    # ========================================================

    angular_acceleration_raw = (
        desired_angular_acceleration

        +

        Kp_orientation
        *
        orientation_error

        +

        Kd_orientation
        *
        (
            desired_angular_velocity

            -

            current_angular_velocity
        )
    )


    angular_acceleration_demand = float(
        np.max(
            np.abs(
                angular_acceleration_raw
            )
        )
    )


    angular_clip_active = bool(
        angular_acceleration_demand

        >

        angular_acceleration_limit
    )


    # ========================================================
    # L. Safety clipping
    # ========================================================

    linear_acceleration_command = np.clip(
        linear_acceleration_raw,
        -linear_acceleration_limit,
        linear_acceleration_limit,
    )


    angular_acceleration_command = np.clip(
        angular_acceleration_raw,
        -angular_acceleration_limit,
        angular_acceleration_limit,
    )


    task_acceleration_command = (
        np.concatenate([
            linear_acceleration_command,
            angular_acceleration_command,
        ])
    )


    # ========================================================
    # M. Extract 6 UR3 columns
    # ========================================================

    J_tcp_arm = (
        J_tcp[
            :,
            arm_pin_v_indices
        ]
    )


    # ========================================================
    # N. Acceleration kinematics
    #
    # xdd
    #
    # =
    #
    # J qdd
    #
    # +
    #
    # Jdot qdot
    #
    #
    # =>
    #
    # J qdd
    #
    # =
    #
    # xdd_cmd
    #
    # -
    #
    # Jdot qdot
    # ========================================================

    rhs = (
        task_acceleration_command

        -

        dJ_tcp

        @

        v_pin
    )


    # ========================================================
    # O. Jacobian diagnostics
    # ========================================================

    singular_values = np.linalg.svd(
        J_tcp_arm,
        compute_uv=False,
    )


    sigma_max = float(
        singular_values[
            0
        ]
    )


    sigma_min = float(
        singular_values[
            -1
        ]
    )


    if sigma_min > 1e-12:

        jacobian_condition = (
            sigma_max
            /
            sigma_min
        )


    else:

        jacobian_condition = (
            np.inf
        )


    # ========================================================
    # P. Exact 6 x 6 acceleration solve
    # ========================================================

    qdd_arm = np.linalg.solve(
        J_tcp_arm,
        rhs,
    )


    # ========================================================
    # Q. Put arm acceleration into full Pinocchio vector
    # ========================================================

    qdd_pin = np.zeros(
        pin_model.nv
    )


    for i in range(
        6
    ):

        qdd_pin[
            arm_pin_v_indices[
                i
            ]
        ] = (
            qdd_arm[
                i
            ]
        )


    # ========================================================
    # R. Pinocchio inverse dynamics
    # ========================================================
    #
    # tau_rnea
    #
    # =
    #
    # M*qdd_des
    #
    # +
    #
    # h(q, qdot)
    # ========================================================

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )


    # ========================================================
    # S. Pinocchio generalized torque -> MuJoCo ordering
    # ========================================================

    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # ========================================================
    # T. Motor command
    #
    # !!! 本版本最关键的修改 !!!
    #
    #
    # MuJoCo:
    #
    # total generalized force
    #
    # =
    #
    # actuator
    #
    # +
    #
    # passive
    #
    # + ...
    #
    #
    # 我们希望：
    #
    # actuator + passive
    #
    # =
    #
    # tau_rnea
    #
    #
    # 所以：
    #
    # actuator
    #
    # =
    #
    # tau_rnea
    #
    # -
    #
    # passive
    # ========================================================

    saturation_count = (
        0
    )


    max_motor_usage = (
        0.0
    )


    max_passive_torque = (
        0.0
    )


    for i in range(
        6
    ):

        actuator_id = (
            arm_actuator_ids[
                i
            ]
        )


        dof_id = (
            arm_dof_indices[
                i
            ]
        )


        # ----------------------------------------------------
        # Pinocchio RNEA desired generalized torque
        # ----------------------------------------------------

        rnea_torque = float(
            tau_mj[
                dof_id
            ]
        )


        # ----------------------------------------------------
        # MuJoCo current passive generalized force
        #
        # 通常关节 damping 会体现在这里。
        # ----------------------------------------------------

        passive_torque = float(
            mj_data.qfrc_passive[
                dof_id
            ]
        )


        max_passive_torque = max(
            max_passive_torque,
            abs(
                passive_torque
            ),
        )


        # ----------------------------------------------------
        # Passive compensation
        #
        # tau_motor
        #
        # =
        #
        # tau_rnea
        #
        # -
        #
        # qfrc_passive
        # ----------------------------------------------------

        requested_torque = (
            rnea_torque

            -

            passive_torque
        )


        # ----------------------------------------------------
        # MuJoCo actuator limits
        # ----------------------------------------------------

        lower = float(
            mj_model.actuator_ctrlrange[
                actuator_id,
                0
            ]
        )


        upper = float(
            mj_model.actuator_ctrlrange[
                actuator_id,
                1
            ]
        )


        # ----------------------------------------------------
        # Saturation diagnostic
        # ----------------------------------------------------

        if (
            requested_torque
            <
            lower

            or

            requested_torque
            >
            upper
        ):

            saturation_count += (
                1
            )


        # ----------------------------------------------------
        # Motor usage diagnostic
        # ----------------------------------------------------

        torque_limit = max(
            abs(
                lower
            ),
            abs(
                upper
            ),
        )


        if torque_limit > 1e-12:

            motor_usage = (
                abs(
                    requested_torque
                )

                /

                torque_limit
            )


            max_motor_usage = max(
                max_motor_usage,
                motor_usage,
            )


        # ----------------------------------------------------
        # Apply actuator command
        # ----------------------------------------------------

        mj_data.ctrl[
            actuator_id
        ] = np.clip(
            requested_torque,
            lower,
            upper,
        )


    # ========================================================
    # U. Keep gripper actuator fixed
    # ========================================================

    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = (
            0.0
        )


    # ========================================================
    # V. Diagnostic errors
    # ========================================================

    tcp_error = (
        np.linalg.norm(
            position_error
        )
    )


    orientation_error_deg = (
        np.rad2deg(
            np.linalg.norm(
                orientation_error
            )
        )
    )


    # --------------------------------------------------------
    # Actual tool axis
    # --------------------------------------------------------

    actual_tool_axis = (
        R_task

        @

        tool_axis_task_local
    )


    actual_tool_axis /= np.linalg.norm(
        actual_tool_axis
    )


    # --------------------------------------------------------
    # Desired tool axis
    # --------------------------------------------------------

    desired_tool_axis = (
        desired_rotation

        @

        tool_axis_task_local
    )


    desired_tool_axis /= np.linalg.norm(
        desired_tool_axis
    )


    # --------------------------------------------------------
    # Tool-axis angular error
    # --------------------------------------------------------

    alignment = float(
        np.clip(
            np.dot(
                actual_tool_axis,
                desired_tool_axis,
            ),
            -1.0,
            1.0,
        )
    )


    tool_error_deg = (
        np.rad2deg(
            np.arccos(
                alignment
            )
        )
    )


    desired_angular_speed = float(
        np.linalg.norm(
            desired_angular_velocity
        )
    )


    # ========================================================
    # W. Return compact diagnostics
    # ========================================================

    return (
        tcp_error,
        orientation_error_deg,
        tool_error_deg,
        jacobian_condition,
        sigma_min,
        saturation_count,
        100.0
        *
        max_motor_usage,
        angular_clip_active,
        desired_angular_speed,
        linear_clip_active,
        linear_acceleration_demand,
        angular_acceleration_demand,
        max_passive_torque,
    )


# ============================================================
# 19. Main simulation
# ============================================================

step = (
    0
)


print(
    "\n"
    "========== Free-space TCP Pivot "
    "自由空间TCP旋转诊断 =========="
)


print(
    f"Virtual tool length "
    f"虚拟工具长度: "
    f"{tool_length * 1000.0:.0f} mm"
)


print(
    "Contact 接触: none 无"
)


print(
    "TCP position TCP位置: fixed 固定"
)


print(
    "Passive compensation "
    "被动力补偿: enabled 已启用"
)


# ============================================================
# 20. Viewer
# ============================================================

with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        current_time = (
            mj_data.time
        )


        # ====================================================
        # A. Generate same trajectory as curved-surface test
        # ====================================================

        (
            curve_s,
            curve_s_dot,
            curve_s_ddot,
        ) = desired_curve_parameter(
            current_time
        )


        # ====================================================
        # B. Desired orientation trajectory
        # ====================================================

        (
            theta,
            theta_s,
            theta_ss,
        ) = normal_angle_derivatives(
            curve_s
        )


        desired_rotation = (
            rotation_z(
                theta
            )

            @

            R_initial
        )


        # ====================================================
        # C. Desired angular velocity
        #
        # theta_dot
        #
        # =
        #
        # dtheta/ds * ds/dt
        # ====================================================

        theta_dot = (
            theta_s

            *

            curve_s_dot
        )


        desired_angular_velocity = np.array([
            0.0,
            0.0,
            theta_dot,
        ])


        # ====================================================
        # D. Desired angular acceleration
        #
        # theta_ddot
        #
        # =
        #
        # theta_ss * s_dot²
        #
        # +
        #
        # theta_s * s_ddot
        # ====================================================

        theta_ddot = (
            theta_ss

            *
            curve_s_dot
            *
            curve_s_dot

            +

            theta_s
            *
            curve_s_ddot
        )


        desired_angular_acceleration = np.array([
            0.0,
            0.0,
            theta_ddot,
        ])


        # ====================================================
        # E. Controller
        # ====================================================

        (
            tcp_error,
            orientation_error,
            tool_error,
            jacobian_condition,
            sigma_min,
            saturation_count,
            motor_usage,
            angular_clip_active,
            desired_angular_speed,
            linear_clip_active,
            linear_acceleration_demand,
            angular_acceleration_demand,
            max_passive_torque,
        ) = tcp_controller(
            desired_rotation,
            desired_angular_velocity,
            desired_angular_acceleration,
        )


        # ====================================================
        # F. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # G. 每秒只打印关键诊断
        # ====================================================

        if (
            step >= 1000

            and

            step % 1000 == 0
        ):

            print(
                "\n"
                "========== Free-space Pivot "
                "自由空间旋转 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Desired tool angle "
                f"目标工具偏转: "
                f"{np.rad2deg(theta):+.2f} deg"
            )


            print(
                f"Tool error "
                f"工具姿态误差: "
                f"{tool_error:.3f} deg"
            )


            print(
                f"TCP error "
                f"固定TCP误差: "
                f"{tcp_error * 1000.0:.3f} mm"
            )


            print(
                f"Desired angular speed "
                f"目标角速度: "
                f"{np.rad2deg(desired_angular_speed):.2f} deg/s"
            )


            print(
                f"Angular accel demand "
                f"姿态加速度需求: "
                f"{angular_acceleration_demand:.2f} rad/s²"
            )


            print(
                f"Angular accel clipping "
                f"姿态加速度限幅: "
                f"{'YES 是' if angular_clip_active else 'NO 否'}"
            )


            print(
                f"Linear accel demand "
                f"TCP线加速度需求: "
                f"{linear_acceleration_demand:.2f} m/s²"
            )


            print(
                f"Linear accel clipping "
                f"TCP线加速度限幅: "
                f"{'YES 是' if linear_clip_active else 'NO 否'}"
            )


            print(
                f"Jacobian condition "
                f"TCP雅可比条件数: "
                f"{jacobian_condition:.2f}"
            )


            print(
                f"Passive torque "
                f"最大被动力矩: "
                f"{max_passive_torque:.3f} N·m"
            )


            print(
                f"Motor saturation "
                f"电机饱和数量: "
                f"{saturation_count}"
            )


            print(
                f"Max motor usage "
                f"最大电机使用率: "
                f"{motor_usage:.1f}%"
            )


        step += (
            1
        )


        # ====================================================
        # H. 仅控制 Viewer 播放速度
        #
        # 不参与数值积分稳定性。
        # ====================================================

        time.sleep(
            dt
        )