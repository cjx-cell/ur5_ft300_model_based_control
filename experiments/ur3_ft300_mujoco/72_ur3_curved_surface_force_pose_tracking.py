#!/usr/bin/env python3

"""
72_ur3_curved_surface_force_pose_tracking.py


目标
============================================================

已知曲面上的：

1. 局部法向恒力控制
2. 曲面切向轨迹跟踪
3. 工具姿态跟随局部法向


相比前一个72：

前一个72：

    球头跟踪曲面
    法向力约5 N
    但工具姿态保持 R_initial

因此灰色工具杆不会始终垂直曲面。


这个版本：

    球头跟踪曲面
    法向力约5 N
    工具轴始终跟随 n(s)


最终希望实现：

                n(s)
                 ↑
                 │
                 │ 灰色工具杆
                 │
                 ● 绿色球头
                /
        _______/_______
             曲面


控制器知道曲面几何。

所以这仍然属于：

    Known surface
    已知曲面

下一步73才会研究：

    Unknown surface normal estimation
    未知表面法向估计
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
    "ur3_ft300_robotiq_curved_surface_pose.xml"
)


# ============================================================
# 2. 基础表面坐标系
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


base_tangent_world = np.array([
    -np.sin(base_surface_angle),
    np.cos(base_surface_angle),
    0.0,
])


# ============================================================
# 3. 曲面参数
# ============================================================
#
# h(s)
#
# =
#
# A/2 * (1 - cos(2*pi*s/L))
#
#
# A = 4 mm
#
# L = 30 mm
# ============================================================

curve_amplitude = (
    0.004
)


curve_length = (
    0.030
)


# ============================================================
# 4. 曲面高度、斜率
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
# 5. 局部表面坐标系
# ============================================================
#
# 曲面：
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
#
# ∝
#
# t0 + h'(s)n0
#
#
# normal:
#
# n(s)
#
# ∝
#
# n0 - h'(s)t0
# ============================================================

def surface_frame(
    s,
):

    (
        height,
        slope,
    ) = surface_height(
        s
    )


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
# 6. 法向相对于基础法向的旋转角
# ============================================================
#
# 因为：
#
# n(s)
#
# =
#
# normalize(
#     n0 - h'(s)t0
# )
#
#
# 所以在XY平面内：
#
# delta_theta
#
# =
#
# -atan(h'(s))
#
#
# 例如：
#
# slope > 0
#
# ->
#
# normal向负方向偏转
# ============================================================

def normal_deflection_angle(
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
# 7. 曲面姿态角的一阶、二阶导数
# ============================================================
#
# theta(s)
#
# =
#
# -atan(m)
#
# m = h'(s)
#
#
# theta_s
#
# =
#
# -m_s / (1 + m^2)
#
#
# theta_ss
#
# =
#
# -[
#   m_ss(1+m^2)
#   -
#   2m(m_s)^2
# ] / (1+m^2)^2
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


    k = (
        2.0
        *
        np.pi
        /
        curve_length
    )


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


    theta = (
        -np.arctan(
            slope
        )
    )


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


    denominator = (
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
        denominator
    )


    return (
        theta,
        theta_s,
        theta_ss,
    )


# ============================================================
# 8. World Z rotation
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


# ============================================================
# 9. 加载正式模型
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
# 10. Reset 原模型
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
# 11. FT300 initial position
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
# 12. Robotiq body
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
# 13. 虚拟工具
# ============================================================

tool_length = (
    0.120
)


tool_rod_radius = (
    0.005
)


probe_radius = (
    0.0005
)


probe_position_world_initial = (
    sensor_position_world_initial

    +

    tool_length
    *
    base_normal_world
)


sensor_position_local_body = (
    R_body_world_initial.T

    @

    (
        sensor_position_world_initial

        -

        body_position_world_initial
    )
)


probe_position_local_body = (
    R_body_world_initial.T

    @

    (
        probe_position_world_initial

        -

        body_position_world_initial
    )
)


# ============================================================
# 14. 初始间隙
# ============================================================

initial_gap = (
    0.035
)


contact_surface_distance = (
    initial_gap
    +
    probe_radius
)


# ============================================================
# 15. 曲面位置
# ============================================================

def surface_point_world(
    s,
):

    (
        height,
        _,
    ) = surface_height(
        s
    )


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
# 16. 球头刚好接触曲面时的球心
# ============================================================

def contact_probe_center_world(
    s,
):

    (
        _,
        normal,
        _,
    ) = surface_frame(
        s
    )


    return (
        surface_point_world(
            s
        )

        -

        probe_radius
        *
        normal
    )


# ============================================================
# 17. 自动生成 MuJoCo 接触环境
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
# 18. 灰色工具杆
# ============================================================
#
# 只有视觉效果：
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
            "pose_tracking_tool_rod",

        "type":
            "cylinder",

        "size":
            f"{tool_rod_radius:.9f}",

        "fromto":
            (
                f"{sensor_position_local_body[0]:.9f} "
                f"{sensor_position_local_body[1]:.9f} "
                f"{sensor_position_local_body[2]:.9f} "
                f"{probe_position_local_body[0]:.9f} "
                f"{probe_position_local_body[1]:.9f} "
                f"{probe_position_local_body[2]:.9f}"
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
# 19. 绿色球头
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "pose_tracking_probe",

        "type":
            "sphere",

        "size":
            f"{probe_radius:.9f}",

        "pos":
            (
                f"{probe_position_local_body[0]:.9f} "
                f"{probe_position_local_body[1]:.9f} "
                f"{probe_position_local_body[2]:.9f}"
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
# 20. 曲面碰撞几何
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


wall_half_segment_length = (
    0.52
    *
    segment_spacing
)


wall_half_height = (
    0.080
)


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
                f"pose_curve_{index}",

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
# 21. 加载实验模型
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
# 22. UR3 joints / actuators
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
# 23. Task frame
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
# 24. FT300 sensor
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
# 25. Payload bodies
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
# 26. Reset
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
# 27. Initial task pose
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
# 28. 非常重要：
#
# Task frame -> 球心
#
# 的固定局部坐标
# ============================================================
#
# 初始时：
#
# p_probe
#
# =
#
# p_task
#
# +
#
# R_initial * r_local
#
#
# 所以：
#
# r_local
#
# =
#
# R_initial^T
#
# *
#
# (p_probe - p_task)
#
#
# 以后姿态改变：
#
# p_probe
#
# =
#
# p_task
#
# +
#
# R_des * r_local
# ============================================================

probe_offset_task_local = (
    R_initial.T

    @

    (
        probe_position_world_initial

        -

        p_initial
    )
)


# ============================================================
# 29. 工具轴在 Task frame 中的方向
# ============================================================

tool_axis_task_local = (
    R_initial.T
    @
    base_normal_world
)


tool_axis_task_local /= np.linalg.norm(
    tool_axis_task_local
)


# ============================================================
# 30. Desired orientation
# ============================================================

def desired_orientation(
    s,
):

    delta_theta = (
        normal_deflection_angle(
            s
        )
    )


    return (
        rotation_z(
            delta_theta
        )
        @
        R_initial
    )


# ============================================================
# 31. Desired probe center
# ============================================================

def desired_probe_center(
    s,
    normal_correction,
):

    (
        _,
        normal,
        _,
    ) = surface_frame(
        s
    )


    return (
        contact_probe_center_world(
            s
        )

        +

        normal_correction
        *
        normal
    )


# ============================================================
# 32. Desired task position
# ============================================================
#
# 最关键的 Tool Offset 公式：
#
# p_task
#
# =
#
# p_probe
#
# -
#
# R_des * r_local
#
#
# 如果没有这一项，
#
# 工具一旋转，
#
# 球头位置就会跑掉。
# ============================================================

def desired_task_position(
    s,
    normal_correction,
):

    R_des = (
        desired_orientation(
            s
        )
    )


    p_probe_des = (
        desired_probe_center(
            s,
            normal_correction,
        )
    )


    return (
        p_probe_des

        -

        R_des
        @
        probe_offset_task_local
    )


# ============================================================
# 33. FT300 raw
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
# 34. Payload gravity compensation
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
# 35. Sensor -> WORLD external wrench
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
# 36. Bias
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
# 37. Noise
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
# 38. Low-pass filter
# ============================================================

cutoff_frequency = (
    10.0
)


filter_tau = (
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
    filter_tau

    /

    (
        filter_tau
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
# 39. Smoothstep
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
# 40. Approach
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
# 41. Force ramp
# ============================================================

force_ramp_start_time = (
    3.0
)


force_ramp_duration = (
    1.5
)


maximum_normal_force = (
    5.0
)


def desired_force_at_time(
    t,
):

    (
        progress,
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
        progress
    )


# ============================================================
# 42. Curve trajectory
# ============================================================

slide_start_time = (
    5.0
)


slide_duration = (
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
        slide_start_time,
        slide_duration,
    )


    s = (
        curve_length
        *
        progress
    )


    s_dot = (
        curve_length
        *
        progress_dot
    )


    s_ddot = (
        curve_length
        *
        progress_ddot
    )


    return (
        s,
        s_dot,
        s_ddot,
    )


# ============================================================
# 43. 二阶动态力环
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


    force_error = (
        desired_force
        -
        measured_force
    )


    if (
        desired_force >= 0.5
        and
        abs(
            force_error
        )
        <
        force_deadband
    ):

        force_error = (
            0.0
        )


    # --------------------------------------------------------
    # Mf*a + Df*v = eF
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

            normal_velocity = (
                0.0
            )


    elif (
        new_correction
        <
        normal_correction_min
    ):

        normal_correction = (
            normal_correction_min
        )


        if normal_velocity < 0.0:

            normal_velocity = (
                0.0
            )


    else:

        normal_correction = (
            new_correction
        )


# ============================================================
# 44. Task position geometry derivatives
# ============================================================
#
# 注意：
#
# 现在求导的是：
#
# task frame 的目标位置
#
# 而不是球头位置。
#
#
# 因为：
#
# 工具姿态改变以后，
#
# FT300 / task origin
#
# 必须为了保持球头位置而自动补偿。
# ============================================================

geometry_epsilon = (
    1e-4
)


def task_geometry_derivatives(
    s,
    correction,
):

    eps = (
        geometry_epsilon
    )


    p_minus = desired_task_position(
        s - eps,
        correction,
    )


    p_center = desired_task_position(
        s,
        correction,
    )


    p_plus = desired_task_position(
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
        normal_plus,
        _,
    ) = surface_frame(
        s + eps
    )


    normal_s = (
        normal_plus
        -
        normal_minus
    ) / (
        2.0
        *
        eps
    )


    (
        _,
        normal_center,
        _,
    ) = surface_frame(
        s
    )


    return (
        p_center,
        dp_ds,
        d2p_ds2,
        normal_center,
        normal_s,
    )


# ============================================================
# 45. Cartesian gains
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
# 46. Cartesian pose controller
# ============================================================

def cartesian_controller(
    external_world_wrench,
    desired_position,
    desired_rotation,
    desired_linear_velocity,
    desired_linear_acceleration,
    desired_angular_velocity,
    desired_angular_acceleration,
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
    # Position error
    # --------------------------------------------------------

    position_error = (
        desired_position
        -
        current_position
    )


    # --------------------------------------------------------
    # Orientation error
    #
    # LOCAL_WORLD_ALIGNED：
    #
    # orientation error也用World表示。
    # --------------------------------------------------------

    R_error = (
        desired_rotation
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
    # Translational acceleration command
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
    # Rotational acceleration command
    #
    # 以前：
    #
    # desired omega = 0
    #
    #
    # 现在：
    #
    # 曲面变化时：
    #
    # desired omega != 0
    # --------------------------------------------------------

    angular_acceleration_command = (
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


    task_acceleration_command = (
        np.concatenate([
            linear_acceleration_command,
            angular_acceleration_command,
        ])
    )


    # --------------------------------------------------------
    # Jdot*qdot
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
    # Damped least squares
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
    # Full acceleration
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
    # Inverse dynamics
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
        current_rotation,
        position_error_norm,
        orientation_error_deg,
    )


# ============================================================
# 47. Phase name
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
            "Force + pose tracking 恒力姿态跟踪"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 48. Main
# ============================================================

step = (
    0
)


print(
    "\n"
    "========== Curved Surface Force + Pose "
    "曲面恒力姿态跟踪 =========="
)


print(
    f"Desired force "
    f"目标法向力: "
    f"{maximum_normal_force:.1f} N"
)


print(
    f"Tool length "
    f"工具长度: "
    f"{tool_length * 1000.0:.0f} mm"
)


print(
    "Tool orientation "
    "工具姿态: follow surface normal 跟随曲面法向"
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
        # A. 曲面轨迹
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
        # C. Bias
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
        # D. FT signal processing
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
        # E. Local normal contact force
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
            desired_force_at_time(
                current_time
            )
        )


        # ====================================================
        # F. Approach / force controller
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
        # G. Desired task position geometry
        # ====================================================

        (
            desired_position,
            position_s,
            position_ss,
            local_normal,
            local_normal_s,
        ) = task_geometry_derivatives(
            curve_s,
            normal_correction,
        )


        # ====================================================
        # H. Desired orientation
        # ====================================================

        desired_rotation = (
            desired_orientation(
                curve_s
            )
        )


        (
            theta,
            theta_s,
            theta_ss,
        ) = normal_angle_derivatives(
            curve_s
        )


        # ====================================================
        # I. Angular velocity
        #
        # theta_dot
        #
        # =
        #
        # theta_s * s_dot
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
        # J. Angular acceleration
        #
        # theta_ddot
        #
        # =
        #
        # theta_ss*s_dot^2
        #
        # +
        #
        # theta_s*s_ddot
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
        # K. Desired linear velocity
        # ====================================================

        desired_linear_velocity = (
            position_s
            *
            curve_s_dot

            +

            local_normal
            *
            normal_velocity
        )


        # ====================================================
        # L. Desired linear acceleration
        # ====================================================

        desired_linear_acceleration = (
            position_ss
            *
            curve_s_dot
            *
            curve_s_dot

            +

            position_s
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


        desired_linear_acceleration = np.clip(
            desired_linear_acceleration,
            -2.0,
            2.0,
        )


        desired_angular_acceleration = np.clip(
            desired_angular_acceleration,
            -4.0,
            4.0,
        )


        # ====================================================
        # M. Cartesian pose controller
        # ====================================================

        (
            current_position,
            current_rotation,
            position_tracking_error,
            orientation_tracking_error,
        ) = cartesian_controller(
            external_world_wrench,
            desired_position,
            desired_rotation,
            desired_linear_velocity,
            desired_linear_acceleration,
            desired_angular_velocity,
            desired_angular_acceleration,
        )


        # ====================================================
        # N. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # O. 每秒精简输出
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            # ------------------------------------------------
            # 曲面法向偏转
            # ------------------------------------------------

            normal_angle_deg = (
                np.rad2deg(
                    normal_deflection_angle(
                        curve_s
                    )
                )
            )


            # ------------------------------------------------
            # 当前实际工具杆方向
            # ------------------------------------------------

            actual_tool_axis = (
                current_rotation
                @
                tool_axis_task_local
            )


            actual_tool_axis /= (
                np.linalg.norm(
                    actual_tool_axis
                )
            )


            alignment_cos = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        local_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_normal_error_deg = (
                np.rad2deg(
                    np.arccos(
                        alignment_cos
                    )
                )
            )


            print(
                "\n"
                "========== Force + Pose "
                "恒力姿态实验 =========="
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
                f"Surface normal angle "
                f"曲面法向偏转: "
                f"{normal_angle_deg:+.2f} deg"
            )


            print(
                f"Tool-normal error "
                f"工具法向夹角误差: "
                f"{tool_normal_error_deg:.3f} deg"
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


        step += (
            1
        )


        time.sleep(
            dt
        )
