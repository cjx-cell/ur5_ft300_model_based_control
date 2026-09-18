#!/usr/bin/env python3

"""
71_ur3_surface_force_tracking.py


目标
============================================================

在倾斜表面上实现：

    Surface normal 法向:
        恒力控制 5 N

    Surface tangent 切向:
        位置轨迹 20 mm

    Z:
        位置保持

    Orientation:
        姿态保持


相比 70：

70:
    WORLD X = Force control
    WORLD Y = Position control

71:
    Surface normal n = Force control
    Surface tangent t = Position control


这意味着：

    Force / Position 控制方向

不再绑定于：

    WORLD XYZ

而是绑定于：

    Surface frame 表面坐标系


实验流程
============================================================

0 ~ 1 s
    FT300 Bias calibration

1 ~ 3 s
    沿表面法向接近墙面
    仍保留约 3 mm 间隙

3 ~ 5 s
    建立 5 N 法向接触力

5 ~ 9 s
    保持 5 N
    同时沿表面切向滑动 20 mm

9 s以后
    保持最终位置和 5 N 接触力
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 已验证的模型与映射工具
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
    "ur3_ft300_robotiq_surface_contact.xml"
)


# ============================================================
# 3. 表面几何参数
# ============================================================
#
# 墙面绕 WORLD Z 旋转 20°
#
#
# 原来的 70：
#
# wall normal = WORLD +X
#
#
# 现在：
#
# wall normal =
#
# [cos(theta),
#  sin(theta),
#  0]
#
#
# wall tangent =
#
# [-sin(theta),
#   cos(theta),
#   0]
# ============================================================

surface_angle_deg = 20.0


surface_angle = np.deg2rad(
    surface_angle_deg
)


surface_normal_world = np.array([
    np.cos(surface_angle),
    np.sin(surface_angle),
    0.0,
])


surface_tangent_world = np.array([
    -np.sin(surface_angle),
    np.cos(surface_angle),
    0.0,
])


# ============================================================
# 4. 先加载正式原始模型
# ============================================================
#
# Pinocchio model：
#
# 仍然使用已经验证一致性的正式模型。
#
#
# base_mj_model：
#
# 只用来获取 home 状态下各个 frame/body 的位置。
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
# 5. Reset 原模型到 home
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
# 6. FT300 site
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
# 7. Robotiq base body
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
# 8. Probe 参数
# ============================================================
#
# Probe center：
#
# FT300
#
#   +
#
# 120 mm * surface_normal
#
#
# 也就是说：
#
# probe 与 FT300 原点沿墙面法向共线。
#
#
# 当墙面法向力作用于 probe 时：
#
# r // F
#
# 所以：
#
# r × F ≈ 0
#
# 法向力不会产生大的额外力矩。
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

    probe_forward_distance
    *
    surface_normal_world

)


# ============================================================
# 9. Probe WORLD -> Robotiq local
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
# 10. 墙面位置
# ============================================================
#
# Wall center 距离 probe center：
#
# 60 mm
#
#
# wall half thickness：
#
# 10 mm
#
#
# probe radius：
#
# 15 mm
#
#
# Initial gap：
#
# 60 - 10 - 15
#
# = 35 mm
# ============================================================

wall_center_distance = (
    0.060
)


wall_half_thickness = (
    0.010
)


initial_gap = (

    wall_center_distance

    -

    wall_half_thickness

    -

    probe_radius

)


wall_center_world = (

    probe_position_world_initial

    +

    wall_center_distance
    *
    surface_normal_world

)


# ============================================================
# 11. 墙面旋转 quaternion
# ============================================================
#
# MuJoCo quaternion：
#
# [w x y z]
#
#
# 绕 WORLD Z：
#
# q =
#
# [
#   cos(theta/2),
#   0,
#   0,
#   sin(theta/2)
# ]
#
#
# Box local X axis
#
# 经过这个旋转以后
#
# 就是：
#
# surface_normal_world
# ============================================================

wall_quaternion = np.array([
    np.cos(
        surface_angle / 2.0
    ),
    0.0,
    0.0,
    np.sin(
        surface_angle / 2.0
    ),
])


# ============================================================
# 12. 自动生成 contact MJCF
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
# 13. Probe
# ============================================================
#
# 非常重要：
#
# mass = 0
#
#
# 这个 probe 只是：
#
#     collision geometry
#
# 不应该改变：
#
#     Robotiq mass
#     COM
#     inertia
#
#
# 否则：
#
# MuJoCo dynamics
#
# 和
#
# Pinocchio dynamics
#
# 又会不一致。
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "surface_contact_probe",

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

        "mass":
            "0",

        "rgba":
            "0.2 0.8 0.2 1",

        "contype":
            "2",

        "conaffinity":
            "2",

        "friction":
            "0.4 0.01 0.001",
    },
)


# ============================================================
# 14. 倾斜墙面
# ============================================================
#
# Box local X：
#
# surface normal
#
#
# Box local Y：
#
# surface tangent
# ============================================================

ET.SubElement(
    worldbody_element,
    "geom",
    {
        "name":
            "inclined_contact_wall",

        "type":
            "box",

        "size":
            "0.01 0.10 0.08",

        "pos":
            (
                f"{wall_center_world[0]:.9f} "
                f"{wall_center_world[1]:.9f} "
                f"{wall_center_world[2]:.9f}"
            ),

        "quat":
            (
                f"{wall_quaternion[0]:.9f} "
                f"{wall_quaternion[1]:.9f} "
                f"{wall_quaternion[2]:.9f} "
                f"{wall_quaternion[3]:.9f}"
            ),

        "rgba":
            "0.7 0.7 0.7 1",

        "contype":
            "2",

        "conaffinity":
            "2",

        "friction":
            "0.4 0.01 0.001",
    },
)


tree.write(
    CONTACT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


# ============================================================
# 15. 加载 contact model
# ============================================================

mj_model = mujoco.MjModel.from_xml_path(
    str(
        CONTACT_XML
    )
)


mj_data = mujoco.MjData(
    mj_model
)


dt = float(
    mj_model.opt.timestep
)


# ============================================================
# 16. UR3 joints / actuators
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


gripper_actuator_id = (
    mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "gripper_position",
    )
)


# ============================================================
# 17. Task frame
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
# 18. FT300
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
# 19. FT300 downstream payload
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
# 20. Reset
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
# 21. Initial task pose
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
# 22. FT300 raw
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
# 23. Payload gravity compensation
# ============================================================

def payload_gravity_wrench():

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
# 24. Sensor wrench -> WORLD external wrench
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
# 25. Bias calibration
# ============================================================

bias_start_time = (
    0.2
)


bias_end_time = (
    1.0
)


bias_sum = np.zeros(
    6
)


bias_count = (
    0
)


bias_wrench = (
    None
)


# ============================================================
# 26. Measurement noise
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
# 27. 6D Low-pass filter
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


filtered_wrench = np.zeros(
    6
)


filter_initialized = (
    False
)


def low_pass_filter(
    wrench,
):

    global filtered_wrench
    global filter_initialized


    if not filter_initialized:

        filtered_wrench = (
            wrench.copy()
        )

        filter_initialized = (
            True
        )


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
# 28. Smoothstep
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
# 29. Approach
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
# 剩余：
#
# 3 mm
#
#
# 注意：
#
# 运动方向不是 WORLD X。
#
# 而是：
#
# surface_normal_world
# ============================================================

approach_start_time = (
    1.0
)


approach_duration = (
    2.0
)


approach_distance = (
    0.032
)


# ============================================================
# 30. Surface-normal force controller
# ============================================================
#
# 法向方向：
#
# n
#
#
# Environment force：
#
# 墙面对 robot 的作用方向为：
#
# -n
#
#
# 所以接触压力大小：
#
# F_normal =
#
# -F_external dot n
#
#
# Force error：
#
# eF =
#
# F_desired
#
# -
#
# F_normal
#
#
# 然后：
#
# v_normal =
#
# Kf * eF
# ============================================================

desired_normal_force = (
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
    0.055
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


def update_normal_force_controller(
    measured_normal_force,
):

    global normal_offset
    global normal_velocity
    global normal_acceleration


    force_error = (
        desired_normal_force
        -
        measured_normal_force
    )


    if abs(
        force_error
    ) < force_deadband:

        force_error = (
            0.0
        )


    normal_velocity = (
        force_velocity_gain
        *
        force_error
    )


    normal_velocity = float(
        np.clip(
            normal_velocity,
            -force_velocity_limit,
            force_velocity_limit,
        )
    )


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
# 31. Surface tangent trajectory
# ============================================================
#
# 5 ~ 9 s
#
# 沿墙面自身：
#
# tangent direction
#
# 滑动 20 mm。
#
#
# 因为墙倾斜20°：
#
# tangent =
#
# [-sin20,
#   cos20,
#   0]
#
#
# 所以这个运动同时包含：
#
# WORLD X
# WORLD Y
#
# 两个分量。
# ============================================================

slide_start_time = (
    5.0
)


slide_duration = (
    4.0
)


slide_distance = (
    0.020
)


def desired_surface_slide(
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


    tangent_offset = (
        slide_distance
        *
        s
    )


    tangent_velocity = (
        slide_distance
        *
        s_dot
    )


    tangent_acceleration = (
        slide_distance
        *
        s_ddot
    )


    return (
        tangent_offset,
        tangent_velocity,
        tangent_acceleration,
    )


# ============================================================
# 32. Cartesian inner-loop gains
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
# 33. Cartesian inner controller
# ============================================================

def cartesian_controller(
    external_world_wrench,
    desired_position,
    desired_linear_velocity,
    desired_linear_acceleration,
):

    # --------------------------------------------------------
    # Pinocchio state
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
    # Errors
    # --------------------------------------------------------

    position_error = (
        desired_position
        -
        current_position
    )


    R_error = (
        R_initial
        @
        current_rotation.T
    )


    orientation_error = (
        pin.log3(
            R_error
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
    # Current task velocity
    # --------------------------------------------------------

    twist = (
        J_full
        @
        v_pin
    )


    linear_velocity = (
        twist[:3]
    )


    angular_velocity = (
        twist[3:]
    )


    # --------------------------------------------------------
    # Desired task acceleration
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
            linear_velocity
        )
    )


    angular_acceleration_command = (
        Kp_orientation
        *
        orientation_error

        -

        Kd_orientation
        *
        angular_velocity
    )


    task_acceleration_command = (
        np.concatenate([
            linear_acceleration_command,
            angular_acceleration_command,
        ])
    )


    # --------------------------------------------------------
    # Jdot * qdot
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
    # DLS
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

    qdd_pin = np.zeros(
        pin_model.nv
    )


    for i in range(6):

        qdd_pin[
            arm_pin_v_indices[i]
        ] = (
            qdd_arm[i]
        )


    # --------------------------------------------------------
    # RNEA
    # --------------------------------------------------------

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )


    # --------------------------------------------------------
    # External wrench disturbance compensation
    #
    # tau_ext =
    #
    # J^T W_ext
    #
    #
    # Motor:
    #
    # tau =
    #
    # tau_ID
    #
    # -
    #
    # J^T W_ext
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
# 34. Experiment phase
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
            "Surface approach 表面接近"
        )


    if t < 5.0:

        return (
            "Normal force regulation 法向恒力建立"
        )


    if t < 9.0:

        return (
            "Surface tracking 表面恒力跟踪"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 35. Main simulation
# ============================================================

step = (
    0
)


print(
    "\n"
    "========== Surface Force Tracking "
    "表面恒力跟踪 =========="
)


print(
    f"Surface angle 表面倾角: "
    f"{surface_angle_deg:.1f} deg"
)


print(
    f"Desired normal force "
    f"目标法向力: "
    f"{desired_normal_force:.1f} N"
)


print(
    f"Surface slide "
    f"沿表面滑动: "
    f"{slide_distance * 1000.0:.1f} mm"
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
        # A. FT300 Raw
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
            current_time >= bias_end_time
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
        # D. Surface-normal contact force
        # ====================================================
        #
        # external force：
        #
        # Environment -> Robot
        #
        #
        # Robot向 +n 压墙。
        #
        # 墙反作用：
        #
        # -n
        #
        #
        # 所以接触压力：
        #
        # F_normal =
        #
        # -F_ext dot n
        # ====================================================

        measured_normal_force = max(
            0.0,
            -float(
                np.dot(
                    external_world_wrench[:3],
                    surface_normal_world,
                )
            ),
        )


        # ====================================================
        # E. Normal direction
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
                approach_distance
                *
                s
            )


            normal_velocity = (
                approach_distance
                *
                s_dot
            )


            normal_acceleration = (
                approach_distance
                *
                s_ddot
            )


        else:

            update_normal_force_controller(
                measured_normal_force
            )


        # ====================================================
        # F. Surface tangent trajectory
        # ====================================================

        (
            tangent_offset,
            tangent_velocity,
            tangent_acceleration,
        ) = desired_surface_slide(
            current_time
        )


        # ====================================================
        # G. Desired Cartesian position
        # ====================================================
        #
        # p_des =
        #
        # p0
        #
        # +
        #
        # n * normal_offset
        #
        # +
        #
        # t * tangent_offset
        # ====================================================

        desired_position = (
            p_initial

            +

            surface_normal_world
            *
            normal_offset

            +

            surface_tangent_world
            *
            tangent_offset
        )


        # ====================================================
        # H. Desired Cartesian velocity
        # ====================================================

        desired_linear_velocity = (
            surface_normal_world
            *
            normal_velocity

            +

            surface_tangent_world
            *
            tangent_velocity
        )


        # ====================================================
        # I. Desired Cartesian acceleration
        # ====================================================

        desired_linear_acceleration = (
            surface_normal_world
            *
            normal_acceleration

            +

            surface_tangent_world
            *
            tangent_acceleration
        )


        # ====================================================
        # J. Inner controller
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
        # L. 精简输出
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            # ------------------------------------------------
            # 实际沿墙切向移动了多少
            # ------------------------------------------------

            displacement_world = (
                current_position
                -
                p_initial
            )


            actual_tangent_displacement = float(
                np.dot(
                    displacement_world,
                    surface_tangent_world,
                )
            )


            actual_world_x_displacement = (
                displacement_world[0]
            )


            if current_time >= 3.0:

                force_target_print = (
                    desired_normal_force
                )

            else:

                force_target_print = (
                    0.0
                )


            print(
                "\n"
                "========== Surface Tracking "
                "表面跟踪实验 =========="
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
                f"Desired normal force "
                f"目标法向力: "
                f"{force_target_print:.2f} N"
            )


            print(
                f"Measured normal force "
                f"实际法向力: "
                f"{measured_normal_force:.2f} N"
            )


            print(
                f"Normal offset "
                f"法向位移: "
                f"{normal_offset * 1000.0:+.2f} mm"
            )


            print(
                f"Surface tangent displacement "
                f"沿表面位移: "
                f"{actual_tangent_displacement * 1000.0:+.2f} mm"
            )


            print(
                f"World X displacement "
                f"世界X位移: "
                f"{actual_world_x_displacement * 1000.0:+.2f} mm"
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
