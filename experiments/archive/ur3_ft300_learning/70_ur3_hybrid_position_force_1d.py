#!/usr/bin/env python3

"""
70_ur3_hybrid_position_force_1d.py


实验目标
============================================================

实现最基础的 Hybrid Position / Force Control：

WORLD X:
    力控制

    Desired contact force:
        5 N


WORLD Y:
    位置控制

    接触稳定以后：
        沿墙面移动 +20 mm


WORLD Z:
    位置保持


Orientation:
    姿态保持


控制结构
============================================================

真实 MuJoCo 接触
        ↓
FT300
        ↓
Bias compensation
        ↓
Payload gravity compensation
        ↓
6D low-pass filter
        ↓
External wrench
        ↓

X:
Force error
        ↓
Force servo
        ↓
Desired X position

Y/Z/Orientation:
Desired pose
        ↓

Cartesian inner controller
        ↓
Pinocchio RNEA
        ↓
-J^T W_external
        ↓
UR3 torque


实验阶段
============================================================

0 ~ 1 s
    FT300 bias calibration

1 ~ 3 s
    接近墙面
    但还不发生接触

3 s以后
    X方向进入 5 N 力控制

5 ~ 9 s
    一边保持 5 N
    一边沿 WORLD Y 方向移动 +20 mm

9 s以后
    保持：
        X force = 5 N
        Y position = +20 mm
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 已经验证过的模型工具
# ============================================================

from ur3_ft300_robotiq_pinocchio import (
    load_models,
    mujoco_to_pinocchio_q,
    mujoco_to_pinocchio_v,
    pinocchio_to_mujoco_tau,
)


# ============================================================
# 2. 文件路径
# ============================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)


SOURCE_XML = (
    SCRIPT_DIR
    /
    "ur3_ft300_robotiq_force_control.xml"
)


CONTACT_XML = (
    SCRIPT_DIR
    /
    "ur3_ft300_robotiq_hybrid_contact.xml"
)


# ============================================================
# 3. 先加载原始模型
# ============================================================
#
# 这里主要有两个目的：
#
# 1.
# 得到已经同步好的 Pinocchio model
#
# 2.
# 得到 home 状态下：
#
# FT300 和 Robotiq 的真实世界位姿
#
# 然后我们才能把接触 probe / wall
# 放到一个合适的位置。
# ============================================================

base_mj_model, pin_model = (
    load_models()
)


base_data = mujoco.MjData(
    base_mj_model
)


pin_data = (
    pin_model.createData()
)


# ============================================================
# 4. 原模型 home
# ============================================================

base_home_key_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)


if base_home_key_id < 0:

    raise RuntimeError(
        "原始模型找不到 home keyframe"
    )


mujoco.mj_resetDataKeyframe(
    base_mj_model,
    base_data,
    base_home_key_id,
)


mujoco.mj_forward(
    base_mj_model,
    base_data,
)


# ============================================================
# 5. 找 FT300 site
# ============================================================

base_ft300_site_id = (
    mujoco.mj_name2id(
        base_mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ft300_site",
    )
)


if base_ft300_site_id < 0:

    raise RuntimeError(
        "找不到 ft300_site"
    )


sensor_position_world_initial = (
    base_data.site_xpos[
        base_ft300_site_id
    ]
    .copy()
)


# ============================================================
# 6. 找 Robotiq base body
# ============================================================

base_robotiq_body_id = (
    mujoco.mj_name2id(
        base_mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "robotiq_85_base_link",
    )
)


if base_robotiq_body_id < 0:

    raise RuntimeError(
        "找不到 robotiq_85_base_link"
    )


body_position_world_initial = (
    base_data.xpos[
        base_robotiq_body_id
    ]
    .copy()
)


R_body_world_initial = (
    base_data.xmat[
        base_robotiq_body_id
    ]
    .reshape(
        3,
        3
    )
    .copy()
)


# ============================================================
# 7. 定义实验 probe
# ============================================================
#
# 我们在 FT300 前方 WORLD +X：
#
#     120 mm
#
# 位置添加一个球形 probe。
#
#
# 为什么不是直接放在 FT300 原点？
#
# 因为：
#
# - 避免和机器人已有几何碰撞
# - 更容易看清楚接触
#
#
# Probe 和 FT300 在 X 方向共线。
#
# 因此墙面法向力也是 X：
#
#     r // F
#
# 所以：
#
#     r × F = 0
#
# 不会额外产生接触力矩。
# ============================================================

probe_forward_distance = (
    0.120
)


probe_radius = (
    0.015
)


probe_position_world_initial = (

    sensor_position_world_initial

    +

    np.array([
        probe_forward_distance,
        0.0,
        0.0,
    ])

)


# ============================================================
# 8. WORLD probe position -> Robotiq local position
# ============================================================

probe_position_local = (

    R_body_world_initial.T

    @

    (
        probe_position_world_initial

        -

        body_position_world_initial
    )

)


# ============================================================
# 9. 定义墙面
# ============================================================
#
# Wall center：
#
# probe 初始中心
#
#     +
#
# 60 mm X
#
#
# Wall X half-size：
#
#     10 mm
#
#
# 因此墙面最近的一侧：
#
#     +50 mm
#
# Probe radius：
#
#     15 mm
#
#
# 所以最初 Probe 与墙面的间隙：
#
#     50 - 15
#
#   = 35 mm
# ============================================================

wall_center_world = (

    probe_position_world_initial

    +

    np.array([
        0.060,
        0.0,
        0.0,
    ])

)


# ============================================================
# 10. 自动生成带接触环境的 MJCF
# ============================================================

tree = ET.parse(
    SOURCE_XML
)


root = tree.getroot()


worldbody_element = (
    root.find(
        "worldbody"
    )
)


if worldbody_element is None:

    raise RuntimeError(
        "MJCF 找不到 worldbody"
    )


robotiq_body_element = (
    root.find(
        ".//body[@name='robotiq_85_base_link']"
    )
)


if robotiq_body_element is None:

    raise RuntimeError(
        "MJCF 找不到 robotiq_85_base_link"
    )


# ============================================================
# 11. 添加 contact probe
# ============================================================
#
# contype = 2
# conaffinity = 2
#
# 我们给 probe 和 wall 单独使用 collision bit 2。
#
# 原机器人通常使用 bit 1。
#
# 因此：
#
# probe 只和 wall 接触
#
# wall 只和 probe 接触
#
# 避免墙面误撞机器人其它 link。
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "hybrid_contact_probe",

        "type":
            "sphere",

        "size":
            f"{probe_radius:.9f}",

        "pos":
            (
                f"{probe_position_local[0]:.9f} "
                f"{probe_position_local[1]:.9f} "
                f"{probe_position_local[2]:.9f}"
            ),

        "rgba":
            "0.2 0.8 0.2 1",

        "contype":
            "2",

        "conaffinity":
            "2",

        "friction":
            "0.8 0.01 0.001",
    },
)


# ============================================================
# 12. 添加墙面
# ============================================================

ET.SubElement(
    worldbody_element,
    "geom",
    {
        "name":
            "hybrid_contact_wall",

        "type":
            "box",

        "size":
            "0.01 0.08 0.08",

        "pos":
            (
                f"{wall_center_world[0]:.9f} "
                f"{wall_center_world[1]:.9f} "
                f"{wall_center_world[2]:.9f}"
            ),

        "rgba":
            "0.7 0.7 0.7 1",

        "contype":
            "2",

        "conaffinity":
            "2",

        "friction":
            "0.8 0.01 0.001",
    },
)


tree.write(
    CONTACT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


# ============================================================
# 13. 加载真正用于实验的 MuJoCo model
# ============================================================

mj_model = (
    mujoco.MjModel.from_xml_path(
        str(
            CONTACT_XML
        )
    )
)


mj_data = mujoco.MjData(
    mj_model
)


dt = float(
    mj_model.opt.timestep
)


# ============================================================
# 14. UR3 joint / actuator
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


for name in ARM_JOINT_NAMES:

    # MuJoCo

    mj_joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )


    if mj_joint_id < 0:

        raise RuntimeError(
            f"MuJoCo 找不到 joint: {name}"
        )


    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        )
    )


    # Pinocchio

    pin_joint_id = (
        pin_model.getJointId(
            name
        )
    )


    if pin_joint_id == 0:

        raise RuntimeError(
            f"Pinocchio 找不到 joint: {name}"
        )


    arm_pin_v_indices.append(
        int(
            pin_model.idx_vs[
                pin_joint_id
            ]
        )
    )


for name in ARM_ACTUATOR_NAMES:

    actuator_id = (
        mujoco.mj_name2id(
            mj_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
    )


    if actuator_id < 0:

        raise RuntimeError(
            f"找不到 actuator: {name}"
        )


    arm_actuator_ids.append(
        actuator_id
    )


gripper_actuator_id = (
    mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "gripper_position",
    )
)


# ============================================================
# 15. Task frame
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
        f"找不到 task frame: {TASK_FRAME_NAME}"
    )


# ============================================================
# 16. FT300
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
    or
    torque_sensor_id < 0
    or
    ft300_site_id < 0
):

    raise RuntimeError(
        "找不到 FT300"
    )


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


sensor_body_id = int(
    mj_model.site_bodyid[
        ft300_site_id
    ]
)


# ============================================================
# 17. Payload bodies
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
# 18. Reset contact model
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
# 19. Initial task pose
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


T_initial = (
    pin_data.oMf[
        task_frame_id
    ]
)


p_initial = (
    T_initial.translation.copy()
)


R_initial = (
    T_initial.rotation.copy()
)


# ============================================================
# 20. FT300 raw
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
# 21. Payload gravity
# ============================================================

def payload_gravity_wrench():

    weighted_com_world = (
        np.zeros(3)
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


    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    gravity_world = np.array(
        mj_model.opt.gravity
    )


    force_world = (

        -payload_mass

        *

        gravity_world

    )


    r_world = (

        payload_com_world

        -

        sensor_position_world

    )


    torque_world = np.cross(
        r_world,
        force_world,
    )


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
# 22. Sensor -> WORLD external wrench
# ============================================================

def sensor_wrench_to_external_world(
    wrench_sensor,
):

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


    force_world = (

        -R_sensor_world

        @

        wrench_sensor[:3]

    )


    torque_world = (

        -R_sensor_world

        @

        wrench_sensor[3:]

    )


    return np.concatenate([
        force_world,
        torque_world,
    ])


# ============================================================
# 23. Bias
# ============================================================

bias_start_time = 0.2

bias_end_time = 1.0


bias_sum = np.zeros(
    6
)


bias_count = 0

bias_wrench = None


# ============================================================
# 24. Measurement noise
# ============================================================

rng = np.random.default_rng(
    42
)


force_noise_std = (
    0.03
)


torque_noise_std = (
    0.002
)


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
# 25. 6D low-pass filter
# ============================================================

cutoff_frequency = (
    10.0
)


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


filter_alpha = (

    filter_time_constant

    /

    (
        filter_time_constant
        +
        dt
    )

)


filtered_wrench = (
    np.zeros(6)
)


filter_initialized = False


def low_pass_filter(
    wrench,
):

    global filtered_wrench
    global filter_initialized


    if not filter_initialized:

        filtered_wrench = (
            wrench.copy()
        )

        filter_initialized = True


    else:

        filtered_wrench = (

            filter_alpha
            *
            filtered_wrench

            +

            (
                1.0
                -
                filter_alpha
            )
            *
            wrench

        )


    return (
        filtered_wrench.copy()
    )


# ============================================================
# 26. Smoothstep
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


    r = (

        (t - start_time)

        /

        duration

    )


    s = (
        3.0 * r**2
        -
        2.0 * r**3
    )


    s_dot = (

        (
            6.0 * r
            -
            6.0 * r**2
        )

        /

        duration

    )


    s_ddot = (

        (
            6.0
            -
            12.0 * r
        )

        /

        (
            duration
            **
            2
        )

    )


    return (
        s,
        s_dot,
        s_ddot,
    )


# ============================================================
# 27. Approach trajectory
# ============================================================
#
# Initial gap:
#
# 35 mm
#
#
# Approach:
#
# 32 mm
#
#
# 所以最终距离墙面：
#
# 约 3 mm
#
# 然后交给 force servo。
# ============================================================

approach_start_time = (
    1.0
)


approach_duration = (
    2.0
)


approach_offset = (
    0.032
)


# ============================================================
# 28. Force controller
# ============================================================
#
# 注意：
#
# 这不是：
#
#     F -> x
#
# 的普通导纳。
#
#
# 这里存在一个明确目标：
#
#     F_desired = 5 N
#
#
# 我们用：
#
#     e_f = F_des - F_measured
#
#
# 然后：
#
#     xdot_ref = Kf * e_f
#
#
# 再积分：
#
#     x_ref += xdot_ref * dt
#
#
# 如果力太小：
#
#     向墙面继续压
#
#
# 如果力太大：
#
#     往后退
# ============================================================

desired_contact_force = (
    5.0
)


force_velocity_gain = (
    0.001
)


force_velocity_limit = (
    0.010
)


force_deadband = (
    0.03
)


normal_offset_min = (
    0.025
)


normal_offset_max = (
    0.050
)


normal_offset = (
    0.0
)


normal_velocity = (
    0.0
)


normal_acceleration = (
    0.0
)


def update_force_controller(
    measured_contact_force,
):

    global normal_offset
    global normal_velocity
    global normal_acceleration


    force_error = (

        desired_contact_force

        -

        measured_contact_force

    )


    if abs(
        force_error
    ) < force_deadband:

        force_error = (
            0.0
        )


    velocity_command = (

        force_velocity_gain

        *

        force_error

    )


    velocity_command = np.clip(

        velocity_command,

        -force_velocity_limit,

        force_velocity_limit,

    )


    normal_velocity = (
        velocity_command
    )


    # 当前这个简单力环不使用加速度前馈

    normal_acceleration = (
        0.0
    )


    normal_offset += (

        normal_velocity

        *

        dt

    )


    normal_offset = float(
        np.clip(
            normal_offset,
            normal_offset_min,
            normal_offset_max,
        )
    )


# ============================================================
# 29. Tangential Y trajectory
# ============================================================
#
# 5 ~ 9 s
#
# 沿墙面移动：
#
# +20 mm WORLD Y
#
#
# 与此同时：
#
# X方向仍保持5 N力控制。
# ============================================================

slide_start_time = (
    5.0
)


slide_duration = (
    4.0
)


slide_distance_y = (
    0.020
)


def desired_tangential_motion(
    t,
):

    (
        s,
        s_dot,
        s_ddot,
    ) = smoothstep(
        t,
        slide_start_time,
        slide_duration,
    )


    y_offset = (
        slide_distance_y
        *
        s
    )


    y_velocity = (
        slide_distance_y
        *
        s_dot
    )


    y_acceleration = (
        slide_distance_y
        *
        s_ddot
    )


    return (
        y_offset,
        y_velocity,
        y_acceleration,
    )


# ============================================================
# 30. Cartesian inner-loop gains
# ============================================================

Kp_position = np.array([
    100.0,
    100.0,
    100.0,
])


Kd_position = np.array([
    20.0,
    20.0,
    20.0,
])


Kp_orientation = np.array([
    100.0,
    100.0,
    100.0,
])


Kd_orientation = np.array([
    20.0,
    20.0,
    20.0,
])


dls_lambda = (
    0.03
)


# ============================================================
# 31. Cartesian controller
# ============================================================

def cartesian_controller(
    external_world_wrench,
    desired_position,
    desired_linear_velocity,
    desired_linear_acceleration,
):

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Kinematics
    # --------------------------------------------------------

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


    current_pose = (
        pin_data.oMf[
            task_frame_id
        ]
    )


    current_position = (
        current_pose.translation.copy()
    )


    current_rotation = (
        current_pose.rotation.copy()
    )


    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    position_error = (

        desired_position

        -

        current_position

    )


    # --------------------------------------------------------
    # Orientation stays at initial pose
    # --------------------------------------------------------

    rotation_error = (

        R_initial

        @

        current_rotation.T

    )


    orientation_error = (
        pin.log3(
            rotation_error
        )
    )


    # --------------------------------------------------------
    # Jacobian
    # --------------------------------------------------------

    J_full = (
        pin.getFrameJacobian(
            pin_model,
            pin_data,
            task_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
    )


    dJ_full = (
        pin.getFrameJacobianTimeVariation(
            pin_model,
            pin_data,
            task_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
    )


    J_arm = (
        J_full[
            :,
            arm_pin_v_indices
        ]
    )


    # --------------------------------------------------------
    # Current Cartesian velocity
    # --------------------------------------------------------

    twist = (

        J_full

        @

        v_pin

    )


    current_linear_velocity = (
        twist[:3]
    )


    current_angular_velocity = (
        twist[3:]
    )


    # --------------------------------------------------------
    # Linear acceleration command
    # --------------------------------------------------------

    linear_acceleration_command = (

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
    # Orientation acceleration command
    # --------------------------------------------------------

    angular_acceleration_command = (

        Kp_orientation
        *
        orientation_error

        -

        Kd_orientation
        *
        current_angular_velocity

    )


    task_acceleration_command = (
        np.concatenate([
            linear_acceleration_command,
            angular_acceleration_command,
        ])
    )


    # --------------------------------------------------------
    # Jdot qdot
    # --------------------------------------------------------

    jdot_v = (

        dJ_full

        @

        v_pin

    )


    rhs = (

        task_acceleration_command

        -

        jdot_v

    )


    # --------------------------------------------------------
    # Damped Least Squares
    # --------------------------------------------------------

    system_matrix = (

        J_arm

        @

        J_arm.T

        +

        (
            dls_lambda
            **
            2
        )

        *

        np.eye(6)

    )


    qdd_arm = (

        J_arm.T

        @

        np.linalg.solve(
            system_matrix,
            rhs,
        )

    )


    # --------------------------------------------------------
    # Full qdd
    # --------------------------------------------------------

    qdd_pin = (
        np.zeros(
            pin_model.nv
        )
    )


    for i in range(6):

        qdd_pin[
            arm_pin_v_indices[i]
        ] = (
            qdd_arm[i]
        )


    # --------------------------------------------------------
    # Inverse dynamics
    # --------------------------------------------------------

    tau_pin = (
        pin.rnea(
            pin_model,
            pin_data,
            q_pin,
            v_pin,
            qdd_pin,
        )
    )


    # --------------------------------------------------------
    # External wrench disturbance compensation
    #
    # 外部接触力仍然真实作用在机器人上。
    #
    # 内环补偿掉这个扰动，
    # 让机器人能够准确跟踪：
    #
    # force loop 生成的 x_ref
    #
    # 和
    #
    # position loop 生成的 y/z/orientation
    # --------------------------------------------------------

    tau_external_arm = (

        J_arm.T

        @

        external_world_wrench

    )


    for i in range(6):

        tau_pin[
            arm_pin_v_indices[i]
        ] -= (
            tau_external_arm[i]
        )


    # --------------------------------------------------------
    # Pin -> MuJoCo
    # --------------------------------------------------------

    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # Motors
    # --------------------------------------------------------

    for i in range(6):

        actuator_id = (
            arm_actuator_ids[i]
        )


        dof_id = (
            arm_dof_indices[i]
        )


        torque = (
            tau_mj[
                dof_id
            ]
        )


        lower = (
            mj_model.actuator_ctrlrange[
                actuator_id,
                0
            ]
        )


        upper = (
            mj_model.actuator_ctrlrange[
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


    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = (
            0.0
        )


    position_error_norm = (
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


    return (
        current_position,
        position_error_norm,
        orientation_error_deg,
    )


# ============================================================
# 32. Phase name
# ============================================================

def phase_name(
    t,
):

    if t < 1.0:

        return (
            "Bias calibration 零偏标定"
        )


    if t < 3.0:

        return (
            "Approach 接近墙面"
        )


    if t < 5.0:

        return (
            "Force regulation 恒力建立"
        )


    if t < 9.0:

        return (
            "Force + sliding 恒力滑动"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 33. Simulation
# ============================================================

step = (
    0
)


print(
    "\n"
    "========== Hybrid Position / Force "
    "位置力混合控制 =========="
)


print(
    f"Desired contact force "
    f"目标接触力: "
    f"{desired_contact_force:.1f} N"
)


print(
    "Tangential motion "
    "切向运动: "
    "+20.0 mm WORLD Y"
)


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:


    while viewer.is_running():

        current_time = (
            mj_data.time
        )


        # ====================================================
        # A. FT300 raw
        # ====================================================

        raw_wrench = (
            read_ft300_raw()
        )


        gravity_wrench = (
            payload_gravity_wrench()
        )


        # ====================================================
        # B. Bias calibration
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


            bias_count += (
                1
            )


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
        # C. Signal processing
        # ====================================================

        if bias_wrench is None:

            external_world_wrench = (
                np.zeros(6)
            )


        else:

            compensated_sensor_wrench = (

                raw_wrench

                -

                gravity_wrench

                -

                bias_wrench

            )


            noisy_sensor_wrench = (
                add_measurement_noise(
                    compensated_sensor_wrench
                )
            )


            filtered_sensor_wrench = (
                low_pass_filter(
                    noisy_sensor_wrench
                )
            )


            external_world_wrench = (
                sensor_wrench_to_external_world(
                    filtered_sensor_wrench
                )
            )


        # ====================================================
        # D. Normal contact force
        # ====================================================
        #
        # 墙在 +X。
        #
        # 机器人向 +X 压墙。
        #
        # 墙对机器人的反作用力：
        #
        #     -X
        #
        # 所以：
        #
        # contact_force =
        #
        #     -F_external_x
        # ====================================================

        measured_contact_force = max(
            0.0,
            -external_world_wrench[0],
        )


        # ====================================================
        # E. X direction
        # ====================================================

        if current_time < (
            approach_start_time
        ):

            normal_offset = (
                0.0
            )

            normal_velocity = (
                0.0
            )

            normal_acceleration = (
                0.0
            )


        elif current_time < (
            approach_start_time
            +
            approach_duration
        ):

            (
                s,
                s_dot,
                s_ddot,
            ) = smoothstep(
                current_time,
                approach_start_time,
                approach_duration,
            )


            normal_offset = (

                approach_offset

                *

                s

            )


            normal_velocity = (

                approach_offset

                *

                s_dot

            )


            normal_acceleration = (

                approach_offset

                *

                s_ddot

            )


        else:

            # -----------------------------------------------
            # Force-controlled direction
            # -----------------------------------------------

            update_force_controller(
                measured_contact_force
            )


        # ====================================================
        # F. Y direction position trajectory
        # ====================================================

        (
            y_offset,
            y_velocity,
            y_acceleration,
        ) = desired_tangential_motion(
            current_time
        )


        # ====================================================
        # G. Desired Cartesian position
        # ====================================================

        desired_position = (
            p_initial.copy()
        )


        desired_position[0] += (
            normal_offset
        )


        desired_position[1] += (
            y_offset
        )


        # Z remains p_initial[2]


        # ====================================================
        # H. Desired velocity
        # ====================================================

        desired_linear_velocity = np.array([
            normal_velocity,
            y_velocity,
            0.0,
        ])


        # ====================================================
        # I. Desired acceleration
        # ====================================================

        desired_linear_acceleration = np.array([
            normal_acceleration,
            y_acceleration,
            0.0,
        ])


        # ====================================================
        # J. Inner Cartesian controller
        # ====================================================

        (
            current_position,
            position_tracking_error,
            orientation_tracking_error,
        ) = cartesian_controller(
            external_world_wrench,
            desired_position,
            desired_linear_velocity,
            desired_linear_acceleration,
        )


        # ====================================================
        # K. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # L. Concise output
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            if current_time >= 3.0:

                force_target_print = (
                    desired_contact_force
                )

            else:

                force_target_print = (
                    0.0
                )


            actual_y_displacement = (

                current_position[1]

                -

                p_initial[1]

            )


            print(
                "\n"
                "========== Hybrid Control "
                "混合控制实验 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Phase 阶段: "
                f"{phase_name(current_time)}"
            )


            print(
                f"Desired contact force "
                f"目标接触力: "
                f"{force_target_print:.2f} N"
            )


            print(
                f"Measured contact force "
                f"实际接触力: "
                f"{measured_contact_force:.2f} N"
            )


            print(
                f"Normal X offset "
                f"法向X位移: "
                f"{normal_offset * 1000.0:+.2f} mm"
            )


            print(
                f"Tangential Y displacement "
                f"切向Y位移: "
                f"{actual_y_displacement * 1000.0:+.2f} mm"
            )


            print(
                f"Position tracking error "
                f"位置跟踪误差: "
                f"{position_tracking_error * 1000.0:.3f} mm"
            )


            print(
                f"Orientation tracking error "
                f"姿态跟踪误差: "
                f"{orientation_tracking_error:.3f} deg"
            )


        step += (
            1
        )


        time.sleep(
            dt
        )