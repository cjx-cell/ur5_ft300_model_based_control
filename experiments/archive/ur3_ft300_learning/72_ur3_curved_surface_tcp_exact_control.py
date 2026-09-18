#!/usr/bin/env python3

"""
72_ur3_curved_surface_tcp_exact_control.py


目标
============================================================

已知曲面上的：

1. 绿色球心 TCP 曲面跟踪
2. 局部法向 5 N 恒力控制
3. 工具轴跟随局部曲面法向
4. 完整 6D TCP acceleration control
5. Jacobian 条件良好时使用精确 6x6 求解


相比上一版
============================================================

上一版：

    qdd =
        J^T (J J^T + lambda^2 I)^-1 b

无论 Jacobian 是否健康，
始终使用 DLS。


这一版：

如果：

    condition number < threshold

并且：

    sigma_min > threshold

则直接：

    J qdd = b

    qdd = solve(J, b)


只有 Jacobian 接近奇异时才退回：

    DLS


目的
============================================================

验证上一版 9~10 s 的：

    姿态误差
        ->
    长工具放大
        ->
    TCP误差
        ->
    接触力波动

是否主要来自 DLS 对完整6D任务的近似。


控制结构
============================================================

FT300
   ↓
法向力反馈
   ↓
二阶力环
   ↓
TCP reference

曲面几何
   ↓
TCP position + tool orientation
   ↓
6D TCP acceleration command
   ↓
Exact Jacobian solve / DLS fallback
   ↓
Pinocchio RNEA
   ↓
Joint torque
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

SCRIPT_DIR = Path(__file__).resolve().parent


SOURCE_XML = (
    SCRIPT_DIR
    /
    "ur3_ft300_robotiq_force_control.xml"
)


CONTACT_XML = (
    SCRIPT_DIR
    /
    "ur3_ft300_robotiq_curved_surface_tcp_exact.xml"
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
# 3. 曲面
# ============================================================

curve_amplitude = 0.004

curve_length = 0.030


def surface_height(s):

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
# 4. 曲面局部坐标系
# ============================================================

def surface_frame(s):

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
# 5. 法向偏转角
# ============================================================

def normal_deflection_angle(s):

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
# 6. 法向角导数
# ============================================================

def normal_angle_derivatives(s):

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
# 7. 基本数学工具
# ============================================================

def rotation_z(angle):

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


def skew(v):

    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


# ============================================================
# 8. 加载正式模型
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
# 9. Reset 原模型
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
# 10. 初始 FT300
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
    ].copy()
)


# ============================================================
# 11. Robotiq base
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
    ].copy()
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
# 12. 虚拟工具
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
# 13. 初始间隙
# ============================================================

initial_gap = 0.035


contact_surface_distance = (
    initial_gap
    +
    probe_radius
)


# ============================================================
# 14. 曲面世界坐标
# ============================================================

def surface_point_world(s):

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
# 15. 球头理论接触中心
# ============================================================

def contact_probe_center_world(s):

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
# 16. 创建实验 MJCF
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


robotiq_body_element = (
    root.find(
        ".//body[@name='robotiq_85_base_link']"
    )
)


if worldbody_element is None:

    raise RuntimeError(
        "MJCF 找不到 worldbody"
    )


if robotiq_body_element is None:

    raise RuntimeError(
        "MJCF 找不到 robotiq_85_base_link"
    )


# ============================================================
# 17. 灰色工具杆
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "exact_tcp_tool_rod",

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
# 18. 绿色球形 TCP
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "exact_tcp_probe",

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
# 19. 曲面碰撞几何
# ============================================================

surface_samples = np.linspace(
    -0.005,
    curve_length + 0.005,
    161,
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
                f"exact_tcp_curve_{index}",

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
# 21. Arm mapping
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

    joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )


    if joint_id < 0:

        raise RuntimeError(
            f"MuJoCo 找不到 joint: {name}"
        )


    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                joint_id
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
# 22. Pinocchio task frame
# ============================================================

task_frame_id = pin_model.getFrameId(
    "robotiq_ft_frame_id"
)


if task_frame_id >= pin_model.nframes:

    raise RuntimeError(
        "Pinocchio 找不到 robotiq_ft_frame_id"
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
    for body_id in payload_body_ids
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
# 26. 初始 task pose
# ============================================================

q_pin_initial = mujoco_to_pinocchio_q(
    mj_data.qpos,
    mj_model,
    pin_model,
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
# 27. TCP offset
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


tool_axis_task_local = (
    R_initial.T
    @
    base_normal_world
)


tool_axis_task_local /= np.linalg.norm(
    tool_axis_task_local
)


# ============================================================
# 28. FT300 functions
# ============================================================

def read_ft300_raw():

    force = (
        mj_data.sensordata[
            force_sensor_adr
            :
            force_sensor_adr + 3
        ].copy()
    )


    torque = (
        mj_data.sensordata[
            torque_sensor_adr
            :
            torque_sensor_adr + 3
        ].copy()
    )


    return np.concatenate([
        force,
        torque,
    ])


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
        ].copy()
    )


    gravity_world = np.array(
        mj_model.opt.gravity
    )


    force_world = (
        -payload_mass
        *
        gravity_world
    )


    torque_world = np.cross(
        payload_com_world
        -
        sensor_position_world,
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


    return np.concatenate([

        R_sensor_world.T
        @
        force_world,

        R_sensor_world.T
        @
        torque_world,

    ])


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


    return np.concatenate([

        -R_sensor_world
        @
        wrench_sensor[:3],

        -R_sensor_world
        @
        wrench_sensor[3:],

    ])


# ============================================================
# 29. Bias / noise / LPF
# ============================================================

bias_sum = np.zeros(
    6
)


bias_count = 0

bias_wrench = None


rng = np.random.default_rng(
    42
)


force_noise_std = 0.03

torque_noise_std = 0.002


cutoff_frequency = 10.0


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
# 30. Smoothstep
# ============================================================

def smoothstep(
    t,
    start,
    duration,
):

    if t <= start:

        return (
            0.0,
            0.0,
            0.0,
        )


    if t >= (
        start
        +
        duration
    ):

        return (
            1.0,
            0.0,
            0.0,
        )


    r = (
        (t - start)
        /
        duration
    )


    s = (
        3.0
        *
        r**2

        -

        2.0
        *
        r**3
    )


    sd = (
        (
            6.0
            *
            r

            -

            6.0
            *
            r**2
        )

        /

        duration
    )


    sdd = (
        (
            6.0
            -
            12.0
            *
            r
        )

        /

        duration**2
    )


    return (
        s,
        sd,
        sdd,
    )


# ============================================================
# 31. 时间参数
# ============================================================

approach_start = 1.0

approach_duration = 2.0

approach_distance = 0.032


force_ramp_start = 3.0

force_ramp_duration = 1.5

maximum_normal_force = 5.0


slide_start = 5.0

slide_duration = 8.0


def desired_force_at_time(t):

    (
        progress,
        _,
        _,
    ) = smoothstep(
        t,
        force_ramp_start,
        force_ramp_duration,
    )


    return (
        maximum_normal_force
        *
        progress
    )


def desired_curve_parameter(t):

    (
        progress,
        progress_dot,
        progress_ddot,
    ) = smoothstep(
        t,
        slide_start,
        slide_duration,
    )


    return (
        curve_length
        *
        progress,

        curve_length
        *
        progress_dot,

        curve_length
        *
        progress_ddot,
    )


# ============================================================
# 32. 二阶法向力环
#
# 完全保持上一版参数。
# ============================================================

force_virtual_mass = 2.0

force_virtual_damping = 120.0


force_velocity_limit = 0.006

force_acceleration_limit = 0.20


normal_correction_min = -0.006

normal_correction_max = 0.006


normal_correction = (
    -initial_gap
)


normal_velocity = 0.0

normal_acceleration = 0.0


def update_force_controller(
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
        0.03
    ):

        force_error = 0.0


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
# 33. TCP目标位置
# ============================================================

def desired_probe_position(
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
        contact_probe_center_world(
            s
        )

        +

        correction
        *
        normal
    )


# ============================================================
# 34. TCP轨迹几何导数
# ============================================================

geometry_epsilon = (
    1e-4
)


def probe_geometry_derivatives(
    s,
    correction,
):

    eps = (
        geometry_epsilon
    )


    p_minus = desired_probe_position(
        s - eps,
        correction,
    )


    p_center = desired_probe_position(
        s,
        correction,
    )


    p_plus = desired_probe_position(
        s + eps,
        correction,
    )


    p_s = (
        p_plus
        -
        p_minus
    ) / (
        2.0
        *
        eps
    )


    p_ss = (
        p_plus
        -
        2.0
        *
        p_center
        +
        p_minus
    ) / (
        eps**2
    )


    (
        _,
        n_minus,
        _,
    ) = surface_frame(
        s - eps
    )


    (
        _,
        n_center,
        _,
    ) = surface_frame(
        s
    )


    (
        _,
        n_plus,
        _,
    ) = surface_frame(
        s + eps
    )


    n_s = (
        n_plus
        -
        n_minus
    ) / (
        2.0
        *
        eps
    )


    return (
        p_center,
        p_s,
        p_ss,
        n_center,
        n_s,
    )


# ============================================================
# 35. Desired orientation
# ============================================================

def desired_orientation(s):

    theta = (
        normal_deflection_angle(
            s
        )
    )


    return (
        rotation_z(
            theta
        )
        @
        R_initial
    )


# ============================================================
# 36. 6D TCP gains
#
# 这一版先保持与上一版相同。
#
# 这样实验只改变：
#
#     Jacobian solver
#
# 不同时乱改多个参数。
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
# 37. Exact / DLS 切换条件
# ============================================================
#
# 如果：
#
# sigma_min > 0.05
#
# 且：
#
# condition < 40
#
# 使用 exact solve。
#
#
# 否则：
#
# 自动退回DLS。
# ============================================================

exact_sigma_min_limit = (
    0.05
)


exact_condition_limit = (
    40.0
)


dls_lambda = (
    0.03
)


# ============================================================
# 38. TCP Cartesian controller
# ============================================================

def tcp_cartesian_controller(
    external_world_wrench,
    desired_probe_position_world,
    desired_rotation,
    desired_linear_velocity,
    desired_linear_acceleration,
    desired_angular_velocity,
    desired_angular_acceleration,
):

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

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
    # Pinocchio kinematics
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


    # --------------------------------------------------------
    # 当前 TCP offset in WORLD
    # --------------------------------------------------------

    r_world = (
        R_task
        @
        probe_offset_task_local
    )


    current_probe_position = (
        p_task
        +
        r_world
    )


    # --------------------------------------------------------
    # Task-frame Jacobian
    # --------------------------------------------------------

    J_task = pin.getFrameJacobian(
        pin_model,
        pin_data,
        task_frame_id,
        pin.LOCAL_WORLD_ALIGNED,
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
        J_task[:3, :]
    )


    Jw_task = (
        J_task[3:, :]
    )


    # --------------------------------------------------------
    # TCP Jacobian
    #
    # v_tcp =
    #
    # v_task
    #
    # +
    #
    # omega x r
    #
    #
    # =
    #
    # v_task
    #
    # -
    #
    # [r]x omega
    # --------------------------------------------------------

    Jv_probe = (
        Jv_task

        -

        skew(
            r_world
        )
        @
        Jw_task
    )


    J_probe = np.vstack([
        Jv_probe,
        Jw_task,
    ])


    # --------------------------------------------------------
    # TCP Jdot
    # --------------------------------------------------------

    task_twist = (
        J_task
        @
        v_pin
    )


    omega_world = (
        task_twist[3:]
    )


    r_dot_world = np.cross(
        omega_world,
        r_world,
    )


    dJv_probe = (
        dJ_task[:3, :]

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
        dJ_task[3:, :]
    )


    dJ_probe = np.vstack([
        dJv_probe,
        dJ_task[3:, :],
    ])


    # --------------------------------------------------------
    # TCP velocity
    # --------------------------------------------------------

    probe_twist = (
        J_probe
        @
        v_pin
    )


    current_linear_velocity = (
        probe_twist[:3]
    )


    current_angular_velocity = (
        probe_twist[3:]
    )


    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    position_error = (
        desired_probe_position_world

        -

        current_probe_position
    )


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


    # --------------------------------------------------------
    # Safety acceleration limit
    #
    # 限制的是反馈后的最终任务加速度，
    # 防止意外误差导致瞬间巨大命令。
    # --------------------------------------------------------

    linear_acceleration_command = np.clip(
        linear_acceleration_command,
        -3.0,
        3.0,
    )


    angular_acceleration_command = np.clip(
        angular_acceleration_command,
        -8.0,
        8.0,
    )


    task_acceleration_command = (
        np.concatenate([
            linear_acceleration_command,
            angular_acceleration_command,
        ])
    )


    # --------------------------------------------------------
    # 只取 UR3 6个DoF
    # --------------------------------------------------------

    J_probe_arm = (
        J_probe[
            :,
            arm_pin_v_indices
        ]
    )


    # --------------------------------------------------------
    # Acceleration kinematics
    #
    # xdd =
    #
    # J qdd
    #
    # +
    #
    # Jdot qdot
    #
    #
    # 所以：
    #
    # J qdd =
    #
    # xdd_cmd - Jdot*qdot
    # --------------------------------------------------------

    rhs = (
        task_acceleration_command

        -

        dJ_probe
        @
        v_pin
    )


    # ========================================================
    # Jacobian diagnostics
    # ========================================================

    singular_values = np.linalg.svd(
        J_probe_arm,
        compute_uv=False,
    )


    sigma_max = float(
        singular_values[0]
    )


    sigma_min = float(
        singular_values[-1]
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
    # Exact 6x6 solve / DLS fallback
    # ========================================================

    use_exact_solver = (
        sigma_min
        >
        exact_sigma_min_limit

        and

        jacobian_condition
        <
        exact_condition_limit
    )


    if use_exact_solver:

        try:

            qdd_arm = np.linalg.solve(
                J_probe_arm,
                rhs,
            )


            solver_name = (
                "Exact 精确求解"
            )


        except np.linalg.LinAlgError:

            use_exact_solver = (
                False
            )


    if not use_exact_solver:

        system_matrix = (
            J_probe_arm
            @
            J_probe_arm.T

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
            J_probe_arm.T

            @

            np.linalg.solve(
                system_matrix,
                rhs,
            )
        )


        solver_name = (
            "DLS 阻尼最小二乘"
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
    # FT300 external wrench compensation
    #
    # Wrench定义在传感器/task origin，
    # 所以这里仍使用 J_task。
    # --------------------------------------------------------

    J_sensor_arm = (
        J_task[
            :,
            arm_pin_v_indices
        ]
    )


    tau_external_arm = (
        J_sensor_arm.T
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


    # ========================================================
    # Motor diagnostics
    # ========================================================

    saturation_count = (
        0
    )


    max_motor_usage = (
        0.0
    )


    for i in range(6):

        actuator_id = (
            arm_actuator_ids[i]
        )


        dof_id = (
            arm_dof_indices[i]
        )


        requested_torque = float(
            tau_mj[
                dof_id
            ]
        )


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


        if (
            requested_torque < lower
            or
            requested_torque > upper
        ):

            saturation_count += (
                1
            )


        torque_limit = max(
            abs(lower),
            abs(upper),
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


        mj_data.ctrl[
            actuator_id
        ] = np.clip(
            requested_torque,
            lower,
            upper,
        )


    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = (
            0.0
        )


    probe_error = np.linalg.norm(
        position_error
    )


    orientation_error_deg = np.rad2deg(
        np.linalg.norm(
            orientation_error
        )
    )


    return (
        current_probe_position,
        R_task,
        probe_error,
        orientation_error_deg,
        jacobian_condition,
        sigma_min,
        solver_name,
        saturation_count,
        100.0
        *
        max_motor_usage,
    )


# ============================================================
# 39. Phase
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
            "Exact TCP tracking 精确TCP跟踪"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 40. Main
# ============================================================

step = 0


print(
    "\n"
    "========== Exact TCP Curved Surface "
    "精确TCP曲面控制 =========="
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
    "Jacobian solver "
    "雅可比求解: exact when healthy "
    "条件良好时精确求解"
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
        # A. Curve trajectory
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


        # ----------------------------------------------------
        # Bias calibration
        # ----------------------------------------------------

        if (
            0.2
            <= current_time
            <
            1.0
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
            current_time >= 1.0
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
        # C. Wrench filtering
        # ====================================================

        if bias_wrench is None:

            external_world_wrench = (
                np.zeros(6)
            )


        else:

            compensated_wrench = (
                raw_wrench

                -

                gravity_wrench

                -

                bias_wrench
            )


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


            filtered_sensor_wrench = (
                low_pass_filter(
                    compensated_wrench
                    +
                    noise
                )
            )


            external_world_wrench = (
                sensor_wrench_to_external_world(
                    filtered_sensor_wrench
                )
            )


        # ====================================================
        # D. Normal force
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
        # E. Approach / force control
        # ====================================================

        if current_time < 1.0:

            normal_correction = (
                -initial_gap
            )


            normal_velocity = (
                0.0
            )


            normal_acceleration = (
                0.0
            )


        elif current_time < 3.0:

            (
                progress,
                progress_dot,
                progress_ddot,
            ) = smoothstep(
                current_time,
                approach_start,
                approach_duration,
            )


            normal_correction = (
                approach_distance
                *
                progress

                -

                initial_gap
            )


            normal_velocity = (
                approach_distance
                *
                progress_dot
            )


            normal_acceleration = (
                approach_distance
                *
                progress_ddot
            )


        else:

            update_force_controller(
                desired_force,
                measured_normal_force,
            )


        # ====================================================
        # F. TCP geometry
        # ====================================================

        (
            probe_position_des,
            probe_position_s,
            probe_position_ss,
            local_normal,
            local_normal_s,
        ) = probe_geometry_derivatives(
            curve_s,
            normal_correction,
        )


        # ====================================================
        # G. TCP linear trajectory
        # ====================================================

        desired_linear_velocity = (
            probe_position_s
            *
            curve_s_dot

            +

            local_normal
            *
            normal_velocity
        )


        desired_linear_acceleration = (
            probe_position_ss
            *
            curve_s_dot
            *
            curve_s_dot

            +

            probe_position_s
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
        # H. Tool orientation
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


        theta_dot = (
            theta_s
            *
            curve_s_dot
        )


        theta_ddot = (
            theta_ss
            *
            curve_s_dot**2

            +

            theta_s
            *
            curve_s_ddot
        )


        desired_angular_velocity = np.array([
            0.0,
            0.0,
            theta_dot,
        ])


        desired_angular_acceleration = np.array([
            0.0,
            0.0,
            theta_ddot,
        ])


        desired_angular_acceleration = (
            np.clip(
                desired_angular_acceleration,
                -4.0,
                4.0,
            )
        )


        # ====================================================
        # I. Exact TCP controller
        # ====================================================

        (
            current_probe_position,
            current_rotation,
            probe_tracking_error,
            orientation_tracking_error,
            jacobian_condition,
            sigma_min,
            solver_name,
            saturation_count,
            max_motor_usage_percent,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            probe_position_des,
            desired_rotation,
            desired_linear_velocity,
            desired_linear_acceleration,
            desired_angular_velocity,
            desired_angular_acceleration,
        )


        # ====================================================
        # J. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # K. 精简诊断输出
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            actual_tool_axis = (
                current_rotation

                @

                tool_axis_task_local
            )


            actual_tool_axis /= np.linalg.norm(
                actual_tool_axis
            )


            alignment = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        local_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_normal_error = np.rad2deg(
                np.arccos(
                    alignment
                )
            )


            normal_angle = np.rad2deg(
                normal_deflection_angle(
                    curve_s
                )
            )


            print(
                "\n"
                "========== Exact TCP "
                "精确TCP实验 =========="
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
                f"Force 目标/实际法向力: "
                f"{desired_force:.2f} / "
                f"{measured_normal_force:.2f} N"
            )


            print(
                f"Surface normal "
                f"曲面法向偏转: "
                f"{normal_angle:+.2f} deg"
            )


            print(
                f"Tool-normal error "
                f"工具法向夹角误差: "
                f"{tool_normal_error:.3f} deg"
            )


            print(
                f"Probe TCP error "
                f"球头TCP跟踪误差: "
                f"{probe_tracking_error * 1000.0:.3f} mm"
            )


            print(
                f"Jacobian condition "
                f"TCP雅可比条件数: "
                f"{jacobian_condition:.2f}"
            )


            print(
                f"Solver 求解器: "
                f"{solver_name}"
            )


            print(
                f"Motor saturation "
                f"电机饱和数量: "
                f"{saturation_count}"
            )


            print(
                f"Max motor usage "
                f"最大电机使用率: "
                f"{max_motor_usage_percent:.1f}%"
            )


        step += (
            1
        )


        time.sleep(
            dt
        )
