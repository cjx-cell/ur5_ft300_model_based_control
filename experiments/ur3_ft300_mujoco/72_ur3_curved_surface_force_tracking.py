#!/usr/bin/env python3

"""
72_ur3_curved_surface_force_tracking.py


目标
============================================================

已知曲面上的动态恒力跟踪：

局部法向 n(s):
    5 N force control

局部切向 t(s):
    沿已知曲面运动

Orientation:
    保持初始姿态


相比上一版本
============================================================

上一版外力环：

    d_dot = Kf * (Fd - F)

现在：

    Mf * d_ddot
    +
    Df * d_dot
    =
    Fd - F


也就是：

    Force error
        ↓
    Virtual mass + damping
        ↓
    Normal acceleration
        ↓
    Normal velocity
        ↓
    Normal correction


同时：

3.0 ~ 4.5 s
    Desired force:
        0 -> 5 N

平滑建立接触力，减少刚接触时的冲击。


实验流程
============================================================

0 ~ 1 s
    FT300 Bias calibration

1 ~ 3 s
    接近曲面
    最后保留约 3 mm 间隙

3 ~ 4.5 s
    目标力 0 -> 5 N 平滑建立

4.5 ~ 5 s
    5 N 稳定

5 ~ 13 s
    沿曲面运动 30 mm
    同时保持 5 N

13 s以后
    曲面终点保持 5 N
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

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
# 1. 文件路径
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
    "ur3_ft300_robotiq_curved_surface.xml"
)


# ============================================================
# 2. 基础表面坐标系
# ============================================================

base_surface_angle_deg = 20.0


base_surface_angle = np.deg2rad(
    base_surface_angle_deg
)


base_normal_world = np.array([
    np.cos(base_surface_angle),
    np.sin(base_surface_angle),
    0.0,
])


base_tangent_world = np.array([
    -np.sin(base_surface_angle),
    np.cos(base_surface_angle),
    0.0,
])


# ============================================================
# 3. 曲面参数
# ============================================================
#
# h(s) =
#
# A/2 * (1 - cos(2*pi*s/L))
#
#
# A = 4 mm
#
# L = 30 mm
# ============================================================

curve_amplitude = 0.004

curve_length = 0.030


# ============================================================
# 4. 曲面高度 + 斜率
# ============================================================

def surface_height(s):

    # 曲面外使用平面延伸

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
            np.cos(phase)
        )
    )


    slope = (
        curve_amplitude
        *
        np.pi
        /
        curve_length
        *
        np.sin(phase)
    )


    return (
        height,
        slope,
    )


# ============================================================
# 5. 局部表面坐标系
# ============================================================
#
# p(s)
#
# =
#
# p0
# +
# s*t0
# +
# h(s)*n0
#
#
# tangent:
#
# t(s)
# ∝
# t0 + h'(s)n0
#
#
# normal:
#
# n(s)
# ∝
# n0 - h'(s)t0
# ============================================================

def surface_frame(s):

    (
        height,
        slope,
    ) = surface_height(s)


    tangent = (
        base_tangent_world
        +
        slope
        *
        base_normal_world
    )


    tangent /= np.linalg.norm(
        tangent
    )


    normal = (
        base_normal_world
        -
        slope
        *
        base_tangent_world
    )


    normal /= np.linalg.norm(
        normal
    )


    return (
        height,
        normal,
        tangent,
    )


# ============================================================
# 6. 加载正式原始模型
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
# 7. Reset 原模型
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
# 8. FT300 初始位置
# ============================================================

base_ft300_site_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ft300_site",
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
# 9. Robotiq base
# ============================================================

base_robotiq_body_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robotiq_85_base_link",
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
# 10. 虚拟工具
# ============================================================

tool_length = 0.120

tool_rod_radius = 0.005

probe_radius = 0.005


probe_position_world_initial = (
    sensor_position_world_initial

    +

    tool_length
    *
    base_normal_world
)


sensor_position_local = (
    R_body_world_initial.T

    @

    (
        sensor_position_world_initial

        -

        body_position_world_initial
    )
)


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
# 11. 初始间隙
# ============================================================

initial_gap = 0.035


contact_surface_distance = (
    initial_gap
    +
    probe_radius
)


# ============================================================
# 12. 曲面世界位置
# ============================================================

def surface_point_world(s):

    (
        height,
        _,
    ) = surface_height(s)


    return (
        probe_position_world_initial

        +

        contact_surface_distance
        *
        base_normal_world

        +

        s
        *
        base_tangent_world

        +

        height
        *
        base_normal_world
    )


# ============================================================
# 13. 理论接触时球心
# ============================================================

def contact_probe_center_world(s):

    (
        _,
        normal,
        _,
    ) = surface_frame(s)


    return (
        surface_point_world(s)

        -

        probe_radius
        *
        normal
    )


# ============================================================
# 14. 理论接触轨迹相对初始位移
# ============================================================

def contact_displacement(s):

    return (
        contact_probe_center_world(s)

        -

        probe_position_world_initial
    )


# ============================================================
# 15. 生成实验 MJCF
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
# 16. 灰色工具杆
# ============================================================
#
# 仅用于显示：
#
# mass = 0
#
# 不参与碰撞。
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "virtual_tool_rod",

        "type":
            "cylinder",

        "size":
            f"{tool_rod_radius:.9f}",

        "fromto":
            (
                f"{sensor_position_local[0]:.9f} "
                f"{sensor_position_local[1]:.9f} "
                f"{sensor_position_local[2]:.9f} "
                f"{probe_position_local[0]:.9f} "
                f"{probe_position_local[1]:.9f} "
                f"{probe_position_local[2]:.9f}"
            ),

        "mass":
            "0",

        "contype":
            "0",

        "conaffinity":
            "0",

        "rgba":
            "0.45 0.45 0.45 1",
    },
)


# ============================================================
# 17. 绿色球形接触头
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "curved_surface_probe",

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
            "0.30 0.01 0.001",
    },
)


# ============================================================
# 18. 曲面离散
# ============================================================

surface_start_s = (
    -0.005
)


surface_end_s = (
    curve_length
    +
    0.005
)


surface_segment_count = (
    161
)


surface_samples = np.linspace(
    surface_start_s,
    surface_end_s,
    surface_segment_count,
)


segment_spacing = (
    surface_samples[1]

    -

    surface_samples[0]
)


wall_half_thickness = (
    0.010
)


# 只保留很小重叠
#
# 总长度：
#
# 2 * 0.52 * spacing
#
# =
#
# 1.04 * spacing

wall_half_segment_length = (
    0.52
    *
    segment_spacing
)


wall_half_height = (
    0.080
)


# ============================================================
# 19. 创建曲面小 box
# ============================================================

for index, s_value in enumerate(
    surface_samples
):

    (
        _,
        normal,
        _,
    ) = surface_frame(
        s_value
    )


    surface_point = (
        surface_point_world(
            s_value
        )
    )


    box_center = (
        surface_point

        +

        wall_half_thickness
        *
        normal
    )


    yaw = np.arctan2(
        normal[1],
        normal[0],
    )


    quaternion = np.array([
        np.cos(
            yaw / 2.0
        ),
        0.0,
        0.0,
        np.sin(
            yaw / 2.0
        ),
    ])


    ET.SubElement(
        worldbody_element,
        "geom",
        {
            "name":
                f"curved_wall_{index}",

            "type":
                "box",

            "size":
                (
                    f"{wall_half_thickness:.9f} "
                    f"{wall_half_segment_length:.9f} "
                    f"{wall_half_height:.9f}"
                ),

            "pos":
                (
                    f"{box_center[0]:.9f} "
                    f"{box_center[1]:.9f} "
                    f"{box_center[2]:.9f}"
                ),

            "quat":
                (
                    f"{quaternion[0]:.9f} "
                    f"{quaternion[1]:.9f} "
                    f"{quaternion[2]:.9f} "
                    f"{quaternion[3]:.9f}"
                ),

            "rgba":
                "0.70 0.70 0.70 1",

            "contype":
                "2",

            "conaffinity":
                "2",

            "friction":
                "0.30 0.01 0.001",
        },
    )


tree.write(
    CONTACT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


# ============================================================
# 20. 加载实验模型
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
# 21. UR3 joints / motors
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


gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)


# ============================================================
# 22. Task frame
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
# 23. FT300
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
# 24. Payload bodies
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
# 25. Reset 实验模型
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
# 26. Initial Cartesian pose
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
# 27. Read FT300
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
# 28. Payload gravity wrench
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
# 29. Sensor -> WORLD wrench
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
# 30. Bias calibration
# ============================================================

bias_start_time = 0.2

bias_end_time = 1.0


bias_sum = np.zeros(
    6
)


bias_count = 0

bias_wrench = None


# ============================================================
# 31. Measurement noise
# ============================================================

rng = np.random.default_rng(
    42
)


force_noise_std = 0.03

torque_noise_std = 0.002


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
# 32. Six-dimensional LPF
# ============================================================

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
# 33. Smoothstep
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
# 34. Approach
# ============================================================

approach_start_time = 1.0

approach_duration = 2.0

approach_distance = 0.032


# ============================================================
# 35. Desired force smooth ramp
# ============================================================
#
# 3.0 ~ 4.5 s:
#
# 0 -> 5 N
#
#
# 避免：
#
# 0 N
#
# 突然跳变为
#
# 5 N
# ============================================================

force_ramp_start_time = 3.0

force_ramp_duration = 1.5

maximum_normal_force = 5.0


def desired_normal_force_at_time(
    t,
):

    (
        s,
        _,
        _,
    ) = smoothstep(
        t,
        force_ramp_start_time,
        force_ramp_duration,
    )


    return (
        maximum_normal_force
        *
        s
    )


# ============================================================
# 36. Curve trajectory
# ============================================================

slide_start_time = 5.0

slide_duration = 8.0


def desired_curve_parameter(
    t,
):

    (
        progress,
        progress_dot,
        progress_ddot,
    ) = smoothstep(
        t,
        slide_start_time,
        slide_duration,
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
# 37. 二阶动态力环
# ============================================================
#
# 关键公式：
#
# Mf * d_ddot
#
# +
#
# Df * d_dot
#
# =
#
# F_des - F_measured
#
#
# 其中：
#
# d
#
# =
#
# 相对于理论接触轨迹的
# 法向位置修正
#
#
# 参数：
#
# Mf = 2 kg
#
# Df = 120 N*s/m
#
#
# 对1 N力误差：
#
# 稳态法向修正速度约：
#
# 1 / 120
#
# =
#
# 0.0083 m/s
#
# =
#
# 8.3 mm/s
#
#
# 实际再受到速度限制。
# ============================================================

force_virtual_mass = (
    2.0
)


force_virtual_damping = (
    120.0
)


force_velocity_limit = (
    0.006
)


force_acceleration_limit = (
    0.20
)


force_deadband = (
    0.03
)


normal_correction_min = (
    -0.006
)


normal_correction_max = (
    0.006
)


# 初始时理论接触轨迹
# 在机器人前方35 mm。
#
# 所以 -35 mm 刚好回到 home。

normal_correction = (
    -initial_gap
)


normal_velocity = (
    0.0
)


normal_acceleration = (
    0.0
)


def update_dynamic_force_controller(
    desired_force,
    measured_force,
):

    global normal_correction
    global normal_velocity
    global normal_acceleration


    # --------------------------------------------------------
    # Force error
    # --------------------------------------------------------

    force_error = (
        desired_force
        -
        measured_force
    )


    if (
        desired_force
        >= 0.5
        and
        abs(force_error)
        <
        force_deadband
    ):

        force_error = 0.0


    # --------------------------------------------------------
    # Virtual mass-damping dynamics
    #
    # Mf*a + Df*v = eF
    #
    # =>
    #
    # a = (eF - Df*v) / Mf
    # --------------------------------------------------------

    normal_acceleration = (
        (
            force_error

            -

            force_virtual_damping
            *
            normal_velocity
        )

        /

        force_virtual_mass
    )


    normal_acceleration = float(
        np.clip(
            normal_acceleration,
            -force_acceleration_limit,
            force_acceleration_limit,
        )
    )


    # --------------------------------------------------------
    # Semi-implicit Euler
    # --------------------------------------------------------

    normal_velocity += (
        normal_acceleration
        *
        dt
    )


    normal_velocity = float(
        np.clip(
            normal_velocity,
            -force_velocity_limit,
            force_velocity_limit,
        )
    )


    new_correction = (
        normal_correction

        +

        normal_velocity
        *
        dt
    )


    # --------------------------------------------------------
    # Position safety + anti-windup
    # --------------------------------------------------------

    if (
        new_correction
        >
        normal_correction_max
    ):

        normal_correction = (
            normal_correction_max
        )


        if normal_velocity > 0.0:

            normal_velocity = 0.0


    elif (
        new_correction
        <
        normal_correction_min
    ):

        normal_correction = (
            normal_correction_min
        )


        if normal_velocity < 0.0:

            normal_velocity = 0.0


    else:

        normal_correction = (
            new_correction
        )


# ============================================================
# 38. Desired displacement
# ============================================================

def desired_displacement(
    s,
    correction,
):

    (
        _,
        normal,
        _,
    ) = surface_frame(
        s
    )


    return (
        contact_displacement(
            s
        )

        +

        correction
        *
        normal
    )


# ============================================================
# 39. Curve geometry derivatives
# ============================================================

geometry_epsilon = (
    1e-4
)


def geometry_derivatives(
    s,
    correction,
):

    eps = (
        geometry_epsilon
    )


    p_minus = desired_displacement(
        s - eps,
        correction,
    )


    p_center = desired_displacement(
        s,
        correction,
    )


    p_plus = desired_displacement(
        s + eps,
        correction,
    )


    dp_ds = (
        p_plus
        -
        p_minus
    ) / (
        2.0
        *
        eps
    )


    d2p_ds2 = (
        p_plus
        -
        2.0
        *
        p_center
        +
        p_minus
    ) / (
        eps
        **
        2
    )


    (
        _,
        normal_minus,
        _,
    ) = surface_frame(
        s - eps
    )


    (
        _,
        normal_center,
        _,
    ) = surface_frame(
        s
    )


    (
        _,
        normal_plus,
        _,
    ) = surface_frame(
        s + eps
    )


    dn_ds = (
        normal_plus
        -
        normal_minus
    ) / (
        2.0
        *
        eps
    )


    return (
        p_center,
        dp_ds,
        d2p_ds2,
        normal_center,
        dn_ds,
    )


# ============================================================
# 40. Cartesian inner loop
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
# 41. Cartesian controller
# ============================================================

def cartesian_controller(
    external_world_wrench,
    desired_position,
    desired_linear_velocity,
    desired_linear_acceleration,
):

    q_pin = mujoco_to_pinocchio_q(
        mj_data.qpos,
        mj_model,
        pin_model,
    )


    v_pin = mujoco_to_pinocchio_v(
        mj_data.qvel,
        mj_model,
        pin_model,
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
    # Cartesian errors
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

    J_full = pin.getFrameJacobian(
        pin_model,
        pin_data,
        task_frame_id,
        pin.LOCAL_WORLD_ALIGNED,
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


    current_linear_velocity = (
        twist[:3]
    )


    current_angular_velocity = (
        twist[3:]
    )


    # --------------------------------------------------------
    # Cartesian acceleration command
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
    # Full Pinocchio acceleration
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
    # RNEA inverse dynamics
    # --------------------------------------------------------

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )


    # --------------------------------------------------------
    # External wrench compensation
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
    # Pinocchio -> MuJoCo
    # --------------------------------------------------------

    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # UR3 motors
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
# 42. Phase
# ============================================================

def phase_name(t):

    if t < 1.0:

        return (
            "Bias calibration 零偏标定"
        )


    if t < 3.0:

        return (
            "Surface approach 曲面接近"
        )


    if t < 4.5:

        return (
            "Force ramp 力平滑建立"
        )


    if t < 5.0:

        return (
            "Force regulation 恒力稳定"
        )


    if t < 13.0:

        return (
            "Curved tracking 曲面恒力跟踪"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 43. Main simulation
# ============================================================

step = 0


print(
    "\n"
    "========== Dynamic Curved Surface Tracking "
    "动态曲面恒力跟踪 =========="
)


print(
    f"Curve amplitude 曲面高度: "
    f"{curve_amplitude * 1000.0:.1f} mm"
)


print(
    f"Curve length 曲面长度: "
    f"{curve_length * 1000.0:.1f} mm"
)


print(
    f"Maximum force 最大目标力: "
    f"{maximum_normal_force:.1f} N"
)


print(
    f"Force virtual mass "
    f"力环虚拟质量: "
    f"{force_virtual_mass:.1f} kg"
)


print(
    f"Force virtual damping "
    f"力环虚拟阻尼: "
    f"{force_virtual_damping:.1f} N·s/m"
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
        # A. Curve parameter
        # ====================================================

        (
            curve_s,
            curve_s_dot,
            curve_s_ddot,
        ) = desired_curve_parameter(
            current_time
        )


        (
            _,
            local_normal,
            _,
        ) = surface_frame(
            curve_s
        )


        # ====================================================
        # B. FT300
        # ====================================================

        raw_wrench = (
            read_ft300_raw()
        )


        gravity_wrench = (
            payload_gravity_wrench()
        )


        # ====================================================
        # C. Bias calibration
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
        # D. Signal processing
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
        # E. Local normal force
        # ====================================================

        measured_normal_force = max(
            0.0,

            -float(
                np.dot(
                    external_world_wrench[:3],
                    local_normal,
                )
            ),
        )


        desired_force = (
            desired_normal_force_at_time(
                current_time
            )
        )


        # ====================================================
        # F. Approach / force control
        # ====================================================

        if current_time < (
            approach_start_time
        ):

            normal_correction = (
                -initial_gap
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
                approach_progress,
                approach_progress_dot,
                approach_progress_ddot,
            ) = smoothstep(
                current_time,
                approach_start_time,
                approach_duration,
            )


            current_approach = (
                approach_distance
                *
                approach_progress
            )


            normal_correction = (
                current_approach
                -
                initial_gap
            )


            normal_velocity = (
                approach_distance
                *
                approach_progress_dot
            )


            normal_acceleration = (
                approach_distance
                *
                approach_progress_ddot
            )


        else:

            update_dynamic_force_controller(
                desired_force,
                measured_normal_force,
            )


        # ====================================================
        # G. Geometry
        # ====================================================

        (
            displacement,
            displacement_s,
            displacement_ss,
            local_normal,
            local_normal_s,
        ) = geometry_derivatives(
            curve_s,
            normal_correction,
        )


        desired_position = (
            p_initial
            +
            displacement
        )


        # ====================================================
        # H. Desired velocity
        # ====================================================

        desired_linear_velocity = (
            displacement_s
            *
            curve_s_dot

            +

            local_normal
            *
            normal_velocity
        )


        # ====================================================
        # I. Desired acceleration
        # ====================================================

        desired_linear_acceleration = (
            displacement_ss
            *
            curve_s_dot
            *
            curve_s_dot

            +

            displacement_s
            *
            curve_s_ddot

            +

            2.0
            *
            local_normal_s
            *
            curve_s_dot
            *
            normal_velocity

            +

            local_normal
            *
            normal_acceleration
        )


        desired_linear_acceleration = (
            np.clip(
                desired_linear_acceleration,
                -2.0,
                2.0,
            )
        )


        # ====================================================
        # J. Cartesian controller
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
        # L. 每秒精简打印
        # ====================================================

        if (
            step >= 1000

            and

            step % 1000 == 0
        ):

            local_normal_angle = (
                np.arctan2(
                    local_normal[1],
                    local_normal[0],
                )
            )


            normal_deflection = (
                local_normal_angle
                -
                base_surface_angle
            )


            normal_deflection = (
                np.arctan2(
                    np.sin(
                        normal_deflection
                    ),
                    np.cos(
                        normal_deflection
                    ),
                )
            )


            normal_deflection_deg = (
                np.rad2deg(
                    normal_deflection
                )
            )


            force_error_print = (
                desired_force

                -

                measured_normal_force
            )


            print(
                "\n"
                "========== Dynamic Surface "
                "动态曲面实验 =========="
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
                f"Desired force "
                f"目标法向力: "
                f"{desired_force:.2f} N"
            )


            print(
                f"Measured force "
                f"实际法向力: "
                f"{measured_normal_force:.2f} N"
            )


            print(
                f"Force error "
                f"法向力误差: "
                f"{force_error_print:+.2f} N"
            )


            print(
                f"Curve progress "
                f"曲面参数位置: "
                f"{curve_s * 1000.0:.2f} mm"
            )


            print(
                f"Local normal angle "
                f"局部法向偏转: "
                f"{normal_deflection_deg:+.2f} deg"
            )


            print(
                f"Force correction "
                f"力控法向修正: "
                f"{normal_correction * 1000.0:+.2f} mm"
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


        step += 1


        time.sleep(
            dt
        )