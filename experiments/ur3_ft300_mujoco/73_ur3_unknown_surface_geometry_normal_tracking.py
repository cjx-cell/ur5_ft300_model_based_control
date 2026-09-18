#!/usr/bin/env python3

"""
73_ur3_unknown_surface_geometry_normal_tracking.py


73B
Geometry-assisted Unknown Surface Following

轨迹几何辅助未知曲面跟踪


============================================================
核心区别
============================================================

73A:
    FT300力方向 -> 法向估计
    工具姿态固定
    稳定

73A2:
    FT300力方向 -> 法向估计 -> 工具姿态
    出现 estimator-controller coupling
    失败

73B:
    实际TCP轨迹 -> 几何切向 -> 几何法向
                              ↓
                         工具姿态

    FT300 -> 法向力大小控制

FT300力方向法向仍然计算，
但只用于诊断，不参与控制。


============================================================
控制器不知道真实曲面
============================================================

hidden_surface_xxx()

只能用于：

    1. 构建MuJoCo环境
    2. Ground-truth evaluation

不能进入控制器。


============================================================
当前假设
============================================================

1. 曲面变化位于WORLD XY平面
2. 已知一个粗略初始接近方向
3. 单点球形接触
4. 低摩擦
5. 接触后沿表面连续运动
"""

from pathlib import Path
from collections import deque
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
# 1. Paths
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
    "ur3_ft300_unknown_surface_geometry_normal.xml"
)


# ============================================================
# 2. Initial coarse surface direction
# ============================================================

initial_surface_angle_deg = 20.0


initial_surface_angle = np.deg2rad(
    initial_surface_angle_deg
)


initial_normal_world = np.array([
    np.cos(initial_surface_angle),
    np.sin(initial_surface_angle),
    0.0,
])


initial_tangent_world = np.array([
    -np.sin(initial_surface_angle),
    np.cos(initial_surface_angle),
    0.0,
])


# ============================================================
# 3. Math helpers
# ============================================================

def rotation_z(angle):

    c = np.cos(angle)

    s = np.sin(angle)


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


def normalize(v):

    norm = np.linalg.norm(
        v
    )


    if norm < 1e-12:

        return v.copy()


    return (
        v
        /
        norm
    )


def wrap_to_pi(angle):

    return (
        angle
        +
        np.pi
    ) % (
        2.0
        *
        np.pi
    ) - np.pi


# ============================================================
# 4. ========================================================
# HIDDEN ENVIRONMENT ONLY
# 隐藏真实曲面
#
# 控制器禁止调用下面的真实曲面函数。
# ============================================================

hidden_curve_amplitude = 0.004

hidden_curve_length = 0.030


def hidden_surface_height(s):

    if s <= 0.0:

        return (
            0.0,
            0.0,
        )


    if s >= hidden_curve_length:

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
        hidden_curve_length
    )


    height = (
        0.5
        *
        hidden_curve_amplitude
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
        hidden_curve_amplitude
        *
        np.pi
        /
        hidden_curve_length
        *
        np.sin(
            phase
        )
    )


    return (
        height,
        slope,
    )


def hidden_surface_frame(s):

    (
        height,
        slope,
    ) = hidden_surface_height(
        s
    )


    tangent = (
        initial_tangent_world

        +

        slope
        *
        initial_normal_world
    )


    tangent = normalize(
        tangent
    )


    normal = (
        initial_normal_world

        -

        slope
        *
        initial_tangent_world
    )


    normal = normalize(
        normal
    )


    return (
        height,
        normal,
        tangent,
    )


# ============================================================
# 5. Load formal robot model
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
# 6. Reset original model
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
# 7. Initial geometry
# ============================================================

base_ft300_site_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_SITE,
    "ft300_site",
)


base_robotiq_body_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robotiq_85_base_link",
)


if (
    base_ft300_site_id < 0
    or
    base_robotiq_body_id < 0
):

    raise RuntimeError(
        "找不到FT300或Robotiq body"
    )


sensor_position_world_initial = (
    base_data.site_xpos[
        base_ft300_site_id
    ].copy()
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
# 8. Tool geometry
# ============================================================

tool_length = 0.120

tool_rod_radius = 0.003

probe_radius = 0.005


probe_position_world_initial = (
    sensor_position_world_initial

    +

    tool_length
    *
    initial_normal_world
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
# 9. Hidden surface placement
# ============================================================

initial_gap = 0.035


hidden_contact_surface_distance = (
    initial_gap
    +
    probe_radius
)


def hidden_surface_point_world(s):

    (
        height,
        _,
    ) = hidden_surface_height(
        s
    )


    return (
        probe_position_world_initial

        +

        hidden_contact_surface_distance
        *
        initial_normal_world

        +

        s
        *
        initial_tangent_world

        +

        height
        *
        initial_normal_world
    )


def hidden_contact_center_world(s):

    (
        _,
        normal,
        _,
    ) = hidden_surface_frame(
        s
    )


    return (
        hidden_surface_point_world(
            s
        )

        -

        probe_radius
        *
        normal
    )


# ============================================================
# 10. Generate experiment MJCF
# ============================================================

tree = ET.parse(
    SOURCE_XML
)


root = tree.getroot()


worldbody_element = root.find(
    "worldbody"
)


robotiq_body_element = root.find(
    ".//body[@name='robotiq_85_base_link']"
)


if worldbody_element is None:

    raise RuntimeError(
        "MJCF找不到worldbody"
    )


if robotiq_body_element is None:

    raise RuntimeError(
        "MJCF找不到robotiq_85_base_link"
    )


# ============================================================
# 11. Gray tool rod
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "geometry_normal_tool_rod",

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
# 12. Green TCP sphere
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "geometry_normal_probe",

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
            "0.05 0.005 0.0005",
    },
)


# ============================================================
# 13. Hidden physical curved surface
# ============================================================

surface_samples = np.linspace(
    -0.005,
    hidden_curve_length + 0.005,
    161,
)


segment_spacing = (
    surface_samples[1]
    -
    surface_samples[0]
)


wall_half_thickness = 0.010


wall_half_segment_length = (
    0.52
    *
    segment_spacing
)


wall_half_height = 0.080


for index, s_value in enumerate(
    surface_samples
):

    (
        _,
        hidden_normal,
        _,
    ) = hidden_surface_frame(
        s_value
    )


    surface_point = (
        hidden_surface_point_world(
            s_value
        )
    )


    box_center = (
        surface_point

        +

        wall_half_thickness
        *
        hidden_normal
    )


    yaw = np.arctan2(
        hidden_normal[1],
        hidden_normal[0],
    )


    quat = np.array([
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
                f"geometry_hidden_curve_{index}",

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
                    f"{quat[0]:.9f} "
                    f"{quat[1]:.9f} "
                    f"{quat[2]:.9f} "
                    f"{quat[3]:.9f}"
                ),

            "rgba":
                "0.70 0.70 0.70 1",

            "contype":
                "2",

            "conaffinity":
                "2",

            "friction":
                "0.05 0.005 0.0005",
        },
    )


tree.write(
    CONTACT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


# ============================================================
# 14. Load experiment model
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


print_every_steps = max(
    1,
    int(
        round(
            1.0
            /
            dt
        )
    ),
)


# ============================================================
# 15. Arm mapping
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
            f"MuJoCo找不到关节: {name}"
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
            f"Pinocchio找不到关节: {name}"
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
            f"找不到电机: {name}"
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
# 16. Task frame
# ============================================================

task_frame_id = (
    pin_model.getFrameId(
        "robotiq_ft_frame_id"
    )
)


if task_frame_id >= pin_model.nframes:

    raise RuntimeError(
        "Pinocchio找不到robotiq_ft_frame_id"
    )


# ============================================================
# 17. FT300
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
        "找不到FT300"
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
# 18. Payload
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
# 19. Reset
# ============================================================

home_key_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)


if home_key_id < 0:

    raise RuntimeError(
        "找不到home keyframe"
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
# 20. Initial Pinocchio pose
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


probe_offset_task_local = (
    R_initial.T

    @

    (
        probe_position_world_initial

        -

        p_task_initial
    )
)


tool_axis_task_local = (
    R_initial.T

    @

    initial_normal_world
)


tool_axis_task_local = normalize(
    tool_axis_task_local
)


# ============================================================
# 21. Current TCP helper
# ============================================================

def get_current_probe_position():

    q_pin = (
        mujoco_to_pinocchio_q(
            mj_data.qpos,
            mj_model,
            pin_model,
        )
    )


    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin,
    )


    pin.updateFramePlacements(
        pin_model,
        pin_data,
    )


    pose = (
        pin_data.oMf[
            task_frame_id
        ]
    )


    p_task = (
        pose.translation.copy()
    )


    R_task = (
        pose.rotation.copy()
    )


    r_world = (
        R_task

        @

        probe_offset_task_local
    )


    return (
        p_task

        +

        r_world
    )


# ============================================================
# 22. FT300 helpers
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

    weighted_com = np.zeros(
        3
    )


    for body_id in payload_body_ids:

        weighted_com += (
            mj_model.body_mass[
                body_id
            ]

            *

            mj_data.xipos[
                body_id
            ]
        )


    payload_com = (
        weighted_com

        /

        payload_mass
    )


    sensor_position = (
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
        payload_com
        -
        sensor_position,

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
        wrench_sensor[
            :3
        ],

        -R_sensor_world
        @
        wrench_sensor[
            3:
        ],

    ])


# ============================================================
# 23. Bias + wrench filter
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


wrench_cutoff = 10.0


wrench_tau = (
    1.0

    /

    (
        2.0
        *
        np.pi
        *
        wrench_cutoff
    )
)


wrench_alpha = (
    wrench_tau

    /

    (
        wrench_tau
        +
        dt
    )
)


filtered_wrench = np.zeros(
    6
)


wrench_filter_initialized = False


def filter_wrench(
    wrench,
):

    global filtered_wrench
    global wrench_filter_initialized


    if not wrench_filter_initialized:

        filtered_wrench = (
            wrench.copy()
        )


        wrench_filter_initialized = True


    else:

        filtered_wrench = (
            wrench_alpha
            *
            filtered_wrench

            +

            (
                1.0
                -
                wrench_alpha
            )
            *
            wrench
        )


    return (
        filtered_wrench.copy()
    )


# ============================================================
# 24. ========================================================
# Geometry normal estimator
# 几何法向估计器
#
# 控制器核心。
#
# 完全不使用FT方向。
# ============================================================

geometry_window_seconds = 0.25


geometry_window_samples = max(
    20,
    int(
        round(
            geometry_window_seconds

            /

            dt
        )
    ),
)


geometry_points = deque(
    maxlen=geometry_window_samples
)


geometry_normal = (
    initial_normal_world.copy()
)


geometry_tangent = (
    initial_tangent_world.copy()
)


geometry_normal_angle = (
    initial_surface_angle
)


# ------------------------------------------------------------
# 轨迹跨度太短时不更新。
# ------------------------------------------------------------

geometry_minimum_span = (
    0.0004
)


# ------------------------------------------------------------
# 切向速度太小时冻结估计。
# ------------------------------------------------------------

geometry_minimum_tangent_speed = (
    0.00020
)


# ------------------------------------------------------------
# 几何法向本身再做轻微滤波。
# ------------------------------------------------------------

geometry_normal_cutoff = (
    4.0
)


geometry_normal_tau = (
    1.0

    /

    (
        2.0
        *
        np.pi
        *
        geometry_normal_cutoff
    )
)


geometry_normal_alpha = (
    geometry_normal_tau

    /

    (
        geometry_normal_tau
        +
        dt
    )
)


geometry_max_normal_rate = np.deg2rad(
    30.0
)


def update_geometry_normal(
    current_probe_position,
    tangential_speed_command,
    contact_force,
):

    global geometry_normal
    global geometry_tangent
    global geometry_normal_angle


    # --------------------------------------------------------
    # 只有：
    #
    # 1. 接触稳定
    # 2. 确实正在沿曲面运动
    #
    # 才允许更新。
    # --------------------------------------------------------

    allow_update = (
        contact_force
        >
        3.0

        and

        abs(
            tangential_speed_command
        )
        >
        geometry_minimum_tangent_speed
    )


    if not allow_update:

        return (
            geometry_normal.copy(),
            geometry_tangent.copy(),
            False,
        )


    # --------------------------------------------------------
    # 保存实际TCP的XY位置
    # --------------------------------------------------------

    geometry_points.append(
        current_probe_position[
            :2
        ].copy()
    )


    if len(
        geometry_points
    ) < 20:

        return (
            geometry_normal.copy(),
            geometry_tangent.copy(),
            False,
        )


    points = np.array(
        geometry_points
    )


    # --------------------------------------------------------
    # 整段轨迹空间跨度
    # --------------------------------------------------------

    span = np.linalg.norm(
        points[
            -1
        ]

        -

        points[
            0
        ]
    )


    if span < geometry_minimum_span:

        return (
            geometry_normal.copy(),
            geometry_tangent.copy(),
            False,
        )


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    center = np.mean(
        points,
        axis=0,
    )


    centered_points = (
        points

        -

        center
    )


    covariance = (
        centered_points.T

        @

        centered_points
    ) / float(
        len(
            centered_points
        )
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )


    principal_index = int(
        np.argmax(
            eigenvalues
        )
    )


    tangent_xy = (
        eigenvectors[
            :,
            principal_index
        ]
    )


    tangent_candidate = np.array([
        tangent_xy[0],
        tangent_xy[1],
        0.0,
    ])


    tangent_candidate = normalize(
        tangent_candidate
    )


    # --------------------------------------------------------
    # PCA特征向量存在正负号二义性。
    #
    # 保证和上一次切向同向。
    # --------------------------------------------------------

    if (
        np.dot(
            tangent_candidate,
            geometry_tangent,
        )
        <
        0.0
    ):

        tangent_candidate = (
            -tangent_candidate
        )


    # --------------------------------------------------------
    # XY平面：
    #
    # t = [tx, ty]
    #
    # n = [ty, -tx]
    # --------------------------------------------------------

    normal_candidate = np.array([
        tangent_candidate[1],
        -tangent_candidate[0],
        0.0,
    ])


    normal_candidate = normalize(
        normal_candidate
    )


    # --------------------------------------------------------
    # 保证法向不突然翻转180度
    # --------------------------------------------------------

    if (
        np.dot(
            normal_candidate,
            geometry_normal,
        )
        <
        0.0
    ):

        normal_candidate = (
            -normal_candidate
        )


    candidate_angle = np.arctan2(
        normal_candidate[1],
        normal_candidate[0],
    )


    angle_error = wrap_to_pi(
        candidate_angle

        -

        geometry_normal_angle
    )


    # --------------------------------------------------------
    # 轻微低通
    # --------------------------------------------------------

    filtered_step = (
        (
            1.0
            -
            geometry_normal_alpha
        )

        *

        angle_error
    )


    # --------------------------------------------------------
    # 变化率保护
    # --------------------------------------------------------

    maximum_step = (
        geometry_max_normal_rate

        *

        dt
    )


    filtered_step = float(
        np.clip(
            filtered_step,
            -maximum_step,
            maximum_step,
        )
    )


    geometry_normal_angle = wrap_to_pi(
        geometry_normal_angle

        +

        filtered_step
    )


    geometry_normal = np.array([
        np.cos(
            geometry_normal_angle
        ),

        np.sin(
            geometry_normal_angle
        ),

        0.0,
    ])


    geometry_tangent = np.array([
        -geometry_normal[1],
        geometry_normal[0],
        0.0,
    ])


    geometry_tangent = normalize(
        geometry_tangent
    )


    if (
        np.dot(
            geometry_tangent,
            initial_tangent_world,
        )
        <
        0.0
    ):

        geometry_tangent = (
            -geometry_tangent
        )


    return (
        geometry_normal.copy(),
        geometry_tangent.copy(),
        True,
    )


# ============================================================
# 25. FT force-direction normal
#
# !!! DIAGNOSTIC ONLY !!!
#
# 不进入控制器。
# ============================================================

def force_direction_normal(
    external_force_world,
    fallback_normal,
):

    planar_force = np.array([
        external_force_world[0],
        external_force_world[1],
        0.0,
    ])


    magnitude = np.linalg.norm(
        planar_force
    )


    if magnitude < 0.5:

        return (
            fallback_normal.copy()
        )


    normal_force_direction = (
        -planar_force

        /

        magnitude
    )


    if (
        np.dot(
            normal_force_direction,
            fallback_normal,
        )
        <
        0.0
    ):

        normal_force_direction = (
            -normal_force_direction
        )


    return (
        normal_force_direction
    )


# ============================================================
# 26. Smoothstep
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


    ratio = (
        (t - start)

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
# 27. Timeline
# ============================================================

approach_start = 1.0

approach_duration = 2.0

approach_distance = 0.032


force_ramp_start = 3.0

force_ramp_duration = 2.0


maximum_normal_force = 5.0


slide_start = 6.0

slide_duration = 12.0

slide_distance = 0.030


def desired_force_at_time(
    t,
):

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


def tangential_motion(
    t,
):

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
        slide_distance
        *
        progress,

        slide_distance
        *
        progress_dot,

        slide_distance
        *
        progress_ddot,
    )


# ============================================================
# 28. Dynamic normal force controller
# ============================================================

force_virtual_mass = 2.0

force_virtual_damping = 120.0


normal_velocity_limit = 0.003

normal_acceleration_limit = 0.15


normal_velocity = 0.0

normal_acceleration = 0.0


normal_search_displacement = 0.0


normal_search_min = -0.006

normal_search_max = +0.012


def update_force_controller(
    desired_force,
    measured_force,
):

    global normal_velocity
    global normal_acceleration
    global normal_search_displacement


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
        ) < 0.03
    ):

        force_error = (
            0.0
        )


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
            -normal_acceleration_limit,
            normal_acceleration_limit,
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
            -normal_velocity_limit,
            normal_velocity_limit,
        )
    )


    proposed_displacement = (
        normal_search_displacement

        +

        normal_velocity

        *

        dt
    )


    if proposed_displacement > normal_search_max:

        normal_search_displacement = (
            normal_search_max
        )


        if normal_velocity > 0.0:

            normal_velocity = (
                0.0
            )


    elif proposed_displacement < normal_search_min:

        normal_search_displacement = (
            normal_search_min
        )


        if normal_velocity < 0.0:

            normal_velocity = (
                0.0
            )


    else:

        normal_search_displacement = (
            proposed_displacement
        )


# ============================================================
# 29. Tool orientation reference
#
# 这次输入是geometry normal，
# 不再是FT force direction。
# ============================================================

tool_command_relative_angle = 0.0

tool_command_angular_velocity = 0.0

tool_command_angular_acceleration = 0.0


tool_orientation_gain = (
    4.0
)


tool_command_max_rate = np.deg2rad(
    25.0
)


tool_rate_gain = (
    15.0
)


tool_command_max_acceleration = np.deg2rad(
    120.0
)


tool_angle_deadband = np.deg2rad(
    0.10
)


orientation_follow_start = (
    6.0
)


def update_tool_orientation_reference(
    geometry_relative_angle,
    current_time,
):

    global tool_command_relative_angle
    global tool_command_angular_velocity
    global tool_command_angular_acceleration


    if current_time < orientation_follow_start:

        tool_command_relative_angle = (
            0.0
        )


        tool_command_angular_velocity = (
            0.0
        )


        tool_command_angular_acceleration = (
            0.0
        )


        return (
            tool_command_relative_angle,
            tool_command_angular_velocity,
            tool_command_angular_acceleration,
        )


    angle_error = wrap_to_pi(
        geometry_relative_angle

        -

        tool_command_relative_angle
    )


    if abs(
        angle_error
    ) < tool_angle_deadband:

        target_angular_velocity = (
            0.0
        )


    else:

        target_angular_velocity = (
            tool_orientation_gain

            *

            angle_error
        )


        target_angular_velocity = float(
            np.clip(
                target_angular_velocity,
                -tool_command_max_rate,
                tool_command_max_rate,
            )
        )


    tool_command_angular_acceleration = (
        tool_rate_gain

        *

        (
            target_angular_velocity

            -

            tool_command_angular_velocity
        )
    )


    tool_command_angular_acceleration = float(
        np.clip(
            tool_command_angular_acceleration,
            -tool_command_max_acceleration,
            tool_command_max_acceleration,
        )
    )


    tool_command_angular_velocity += (
        tool_command_angular_acceleration

        *

        dt
    )


    tool_command_angular_velocity = float(
        np.clip(
            tool_command_angular_velocity,
            -tool_command_max_rate,
            tool_command_max_rate,
        )
    )


    tool_command_relative_angle = wrap_to_pi(
        tool_command_relative_angle

        +

        tool_command_angular_velocity

        *

        dt
    )


    return (
        tool_command_relative_angle,
        tool_command_angular_velocity,
        tool_command_angular_acceleration,
    )


# ============================================================
# 30. TCP reference
# ============================================================

probe_reference_position = (
    probe_position_world_initial.copy()
)


previous_reference_velocity = np.zeros(
    3
)


reference_acceleration = np.zeros(
    3
)


reference_accel_cutoff = 4.0


reference_accel_tau = (
    1.0

    /

    (
        2.0
        *
        np.pi
        *
        reference_accel_cutoff
    )
)


reference_accel_alpha = (
    reference_accel_tau

    /

    (
        reference_accel_tau
        +
        dt
    )
)


# ============================================================
# 31. Cartesian control gains
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


linear_acceleration_limit = 30.0

angular_acceleration_limit = 80.0


exact_sigma_min_limit = 0.05

exact_condition_limit = 40.0

dls_lambda = 0.03


# ============================================================
# 32. 6D TCP controller
# ============================================================

def tcp_cartesian_controller(
    external_world_wrench,
    desired_probe_position,
    desired_rotation,
    desired_linear_velocity,
    desired_linear_acceleration,
    desired_angular_velocity,
    desired_angular_acceleration,
):

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
    # Current TCP
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
    # Frame Jacobian
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Shift Jacobian to TCP
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TCP Jdot
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Current TCP twist
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Position / orientation error
    # --------------------------------------------------------

    position_error = (
        desired_probe_position

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


    linear_acceleration_command = np.clip(
        linear_acceleration_command,
        -linear_acceleration_limit,
        linear_acceleration_limit,
    )


    angular_acceleration_command = np.clip(
        angular_acceleration_command,
        -angular_acceleration_limit,
        angular_acceleration_limit,
    )


    xdd_command = np.concatenate([
        linear_acceleration_command,
        angular_acceleration_command,
    ])


    # --------------------------------------------------------
    # Arm Jacobian
    # --------------------------------------------------------

    J_arm = (
        J_tcp[
            :,
            arm_pin_v_indices
        ]
    )


    rhs = (
        xdd_command

        -

        dJ_tcp

        @

        v_pin
    )


    singular_values = np.linalg.svd(
        J_arm,
        compute_uv=False,
    )


    sigma_min = float(
        singular_values[
            -1
        ]
    )


    condition = float(
        singular_values[
            0
        ]

        /

        singular_values[
            -1
        ]
    )


    use_exact = (
        sigma_min
        >
        exact_sigma_min_limit

        and

        condition
        <
        exact_condition_limit
    )


    if use_exact:

        try:

            qdd_arm = np.linalg.solve(
                J_arm,
                rhs,
            )


        except np.linalg.LinAlgError:

            use_exact = False


    if not use_exact:

        matrix = (
            J_arm

            @

            J_arm.T

            +

            dls_lambda**2

            *

            np.eye(
                6
            )
        )


        qdd_arm = (
            J_arm.T

            @

            np.linalg.solve(
                matrix,
                rhs,
            )
        )


    # --------------------------------------------------------
    # Full Pinocchio acceleration
    # --------------------------------------------------------

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
    # FT external wrench compensation
    #
    # Wrench is referenced at sensor/task origin.
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


    for i in range(
        6
    ):

        tau_pin[
            arm_pin_v_indices[
                i
            ]
        ] -= (
            tau_external_arm[
                i
            ]
        )


    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # MuJoCo passive compensation
    # --------------------------------------------------------

    saturation_count = (
        0
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


        desired_torque = float(
            tau_mj[
                dof_id
            ]
        )


        passive_torque = float(
            mj_data.qfrc_passive[
                dof_id
            ]
        )


        requested_torque = (
            desired_torque

            -

            passive_torque
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


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    actual_tool_axis = (
        R_task

        @

        tool_axis_task_local
    )


    actual_tool_axis = normalize(
        actual_tool_axis
    )


    tcp_error = float(
        np.linalg.norm(
            position_error
        )
    )


    return (
        current_probe_position,
        actual_tool_axis,
        tcp_error,
        condition,
        saturation_count,
    )


# ============================================================
# 33. Ground truth evaluation only
# ============================================================

evaluation_s = np.linspace(
    0.0,
    hidden_curve_length,
    1001,
)


evaluation_centers = []

evaluation_normals = []


for s_value in evaluation_s:

    evaluation_centers.append(
        hidden_contact_center_world(
            s_value
        )
    )


    (
        _,
        true_normal,
        _,
    ) = hidden_surface_frame(
        s_value
    )


    evaluation_normals.append(
        true_normal
    )


evaluation_centers = np.array(
    evaluation_centers
)


evaluation_normals = np.array(
    evaluation_normals
)


def evaluate_true_normal(
    current_probe_position,
):

    distances = np.linalg.norm(
        evaluation_centers

        -

        current_probe_position[
            None,
            :
        ],

        axis=1,
    )


    index = int(
        np.argmin(
            distances
        )
    )


    return (
        evaluation_s[
            index
        ],

        evaluation_normals[
            index
        ].copy(),
    )


# ============================================================
# 34. Main
# ============================================================

step = 0


print(
    "\n"
    "========== 73B Geometry Normal "
    "73B轨迹几何法向 =========="
)


print(
    f"Desired force "
    f"目标法向力: "
    f"{maximum_normal_force:.1f} N"
)


print(
    "Surface model "
    "控制器曲面模型: NONE 无"
)


print(
    "Control normal source "
    "控制法向来源: TCP trajectory geometry "
    "TCP轨迹几何"
)


print(
    "FT force direction "
    "FT力方向: diagnostic only 仅诊断"
)


print(
    f"Geometry window "
    f"几何估计窗口: "
    f"{geometry_window_seconds:.2f} s"
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
        # A. Current actual TCP
        # ====================================================

        current_probe_for_estimator = (
            get_current_probe_position()
        )


        # ====================================================
        # B. FT300 measurement
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
                    "没有采集到Bias数据"
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


        # ----------------------------------------------------
        # Compensated wrench
        # ----------------------------------------------------

        if bias_wrench is None:

            external_world_wrench = (
                np.zeros(
                    6
                )
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
                filter_wrench(
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
        # C. Tangential command
        #
        # 只得到速度大小。
        #
        # 方向来自geometry_tangent。
        # ====================================================

        (
            tangential_distance_command,
            tangential_speed,
            _,
        ) = tangential_motion(
            current_time
        )


        # ====================================================
        # D. Current force projection
        #
        # 先使用上一周期geometry normal。
        # ====================================================

        measured_normal_force = max(
            0.0,

            -float(
                np.dot(
                    external_world_wrench[
                        :3
                    ],
                    geometry_normal,
                )
            ),
        )


        # ====================================================
        # E. Geometry normal estimator
        # ====================================================

        (
            normal_geom,
            tangent_geom,
            geometry_updated,
        ) = update_geometry_normal(
            current_probe_for_estimator,
            tangential_speed,
            measured_normal_force,
        )


        # ====================================================
        # F. Recompute force using latest geometry normal
        # ====================================================

        measured_normal_force = max(
            0.0,

            -float(
                np.dot(
                    external_world_wrench[
                        :3
                    ],
                    normal_geom,
                )
            ),
        )


        desired_force = (
            desired_force_at_time(
                current_time
            )
        )


        # ====================================================
        # G. FT force direction
        #
        # Diagnostic only.
        # ====================================================

        normal_force_diag = (
            force_direction_normal(
                external_world_wrench[
                    :3
                ],
                normal_geom,
            )
        )


        # ====================================================
        # H. TCP position reference
        # ====================================================

        if current_time < 1.0:

            probe_reference_position = (
                probe_position_world_initial.copy()
            )


            desired_reference_velocity = (
                np.zeros(
                    3
                )
            )


            normal_velocity = (
                0.0
            )


            normal_acceleration = (
                0.0
            )


        elif current_time < 3.0:

            (
                approach_progress,
                approach_progress_dot,
                _,
            ) = smoothstep(
                current_time,
                approach_start,
                approach_duration,
            )


            probe_reference_position = (
                probe_position_world_initial

                +

                initial_normal_world

                *

                approach_distance

                *

                approach_progress
            )


            desired_reference_velocity = (
                initial_normal_world

                *

                approach_distance

                *

                approach_progress_dot
            )


            normal_velocity = (
                0.0
            )


            normal_acceleration = (
                0.0
            )


        else:

            update_force_controller(
                desired_force,
                measured_normal_force,
            )


            desired_reference_velocity = (
                tangent_geom

                *

                tangential_speed

                +

                normal_geom

                *

                normal_velocity
            )


            probe_reference_position += (
                desired_reference_velocity

                *

                dt
            )


        # ====================================================
        # I. Reference acceleration
        # ====================================================

        raw_reference_acceleration = (
            desired_reference_velocity

            -

            previous_reference_velocity

        ) / dt


        reference_acceleration = (
            reference_accel_alpha

            *

            reference_acceleration

            +

            (
                1.0
                -
                reference_accel_alpha
            )

            *

            raw_reference_acceleration
        )


        reference_acceleration = np.clip(
            reference_acceleration,
            -2.0,
            2.0,
        )


        previous_reference_velocity = (
            desired_reference_velocity.copy()
        )


        # ====================================================
        # J. Tool orientation from GEOMETRY normal
        # ====================================================

        geometry_relative_angle = wrap_to_pi(
            geometry_normal_angle

            -

            initial_surface_angle
        )


        (
            command_relative_angle,
            command_angular_velocity,
            command_angular_acceleration,
        ) = update_tool_orientation_reference(
            geometry_relative_angle,
            current_time,
        )


        desired_rotation = (
            rotation_z(
                command_relative_angle
            )

            @

            R_initial
        )


        desired_angular_velocity = np.array([
            0.0,
            0.0,
            command_angular_velocity,
        ])


        desired_angular_acceleration = np.array([
            0.0,
            0.0,
            command_angular_acceleration,
        ])


        # ====================================================
        # K. 6D TCP controller
        # ====================================================

        (
            current_probe_position,
            actual_tool_axis,
            tcp_tracking_error,
            jacobian_condition,
            saturation_count,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            probe_reference_position,
            desired_rotation,
            desired_reference_velocity,
            reference_acceleration,
            desired_angular_velocity,
            desired_angular_acceleration,
        )


        # ====================================================
        # L. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # M. Evaluation
        # ====================================================

        if (
            step >= print_every_steps

            and

            step % print_every_steps == 0
        ):

            (
                true_s_eval,
                true_normal,
            ) = evaluate_true_normal(
                current_probe_position
            )


            # ------------------------------------------------
            # Angles
            # ------------------------------------------------

            geometry_relative_deg = np.rad2deg(
                geometry_relative_angle
            )


            force_world_angle = np.arctan2(
                normal_force_diag[
                    1
                ],
                normal_force_diag[
                    0
                ],
            )


            force_relative_deg = np.rad2deg(
                wrap_to_pi(
                    force_world_angle

                    -

                    initial_surface_angle
                )
            )


            true_world_angle = np.arctan2(
                true_normal[
                    1
                ],
                true_normal[
                    0
                ],
            )


            true_relative_angle = wrap_to_pi(
                true_world_angle

                -

                initial_surface_angle
            )


            true_relative_deg = np.rad2deg(
                true_relative_angle
            )


            command_relative_deg = np.rad2deg(
                command_relative_angle
            )


            # ------------------------------------------------
            # Geometry normal error
            # ------------------------------------------------

            geometry_alignment = float(
                np.clip(
                    np.dot(
                        normal_geom,
                        true_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            geometry_error_deg = np.rad2deg(
                np.arccos(
                    geometry_alignment
                )
            )


            # ------------------------------------------------
            # FT force direction error
            # ------------------------------------------------

            force_alignment = float(
                np.clip(
                    np.dot(
                        normal_force_diag,
                        true_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            force_direction_error_deg = np.rad2deg(
                np.arccos(
                    force_alignment
                )
            )


            # ------------------------------------------------
            # Tool vs geometry normal
            # ------------------------------------------------

            tool_geometry_alignment = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        normal_geom,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_geometry_error_deg = np.rad2deg(
                np.arccos(
                    tool_geometry_alignment
                )
            )


            # ------------------------------------------------
            # Tool vs true normal
            # ------------------------------------------------

            tool_true_alignment = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        true_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_true_error_deg = np.rad2deg(
                np.arccos(
                    tool_true_alignment
                )
            )


            estimator_state = (
                "UPDATE 更新"

                if geometry_updated

                else

                "HOLD 保持"
            )


            print(
                "\n"
                "========== 73B Unknown Surface "
                "73B未知曲面实验 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Force 目标/实际法向力: "
                f"{desired_force:.2f} / "
                f"{measured_normal_force:.2f} N"
            )


            print(
                f"Normal geom/force/true "
                f"几何/力方向/真实法向: "
                f"{geometry_relative_deg:+.2f} / "
                f"{force_relative_deg:+.2f} / "
                f"{true_relative_deg:+.2f} deg"
            )


            print(
                f"Normal error geom/force "
                f"几何/力方向法向误差: "
                f"{geometry_error_deg:.2f} / "
                f"{force_direction_error_deg:.2f} deg"
            )


            print(
                f"Tool cmd "
                f"工具姿态指令: "
                f"{command_relative_deg:+.2f} deg"
            )


            print(
                f"Tool error geom/true "
                f"工具对几何/真实法向误差: "
                f"{tool_geometry_error_deg:.2f} / "
                f"{tool_true_error_deg:.2f} deg"
            )


            print(
                f"TCP tracking error "
                f"TCP跟踪误差: "
                f"{tcp_tracking_error * 1000.0:.3f} mm"
            )


            print(
                f"Tangential travel "
                f"切向指令距离: "
                f"{tangential_distance_command * 1000.0:.1f} mm"
            )


            print(
                f"Geometry estimator "
                f"几何估计器: "
                f"{estimator_state}"
            )


            print(
                f"Jacobian / saturation "
                f"雅可比条件数/电机饱和: "
                f"{jacobian_condition:.2f} / "
                f"{saturation_count}"
            )


        step += (
            1
        )


        time.sleep(
            dt
        )

