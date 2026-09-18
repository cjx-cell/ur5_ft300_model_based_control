#!/usr/bin/env python3

"""
73_ur3_unknown_surface_force_pose_tracking.py


Unknown Surface Normal Estimation
+ Force Tracking
+ Tool Pose Tracking

未知曲面：
在线法向估计 + 恒力控制 + TCP姿态跟踪


============================================================
与72最大的区别
============================================================

72：

    controller knows:

        h(s)
        h'(s)
        n(s)
        t(s)

73：

    controller DOES NOT know:

        h(s)
        h'(s)
        true n(s)
        true t(s)


控制器只知道：

1. 初始粗略接近方向 n0
2. FT300测量的接触力
3. 自己希望保持5 N
4. 自己希望沿表面向前运动


在线估计：

    n_est
        =
    -F_xy / |F_xy|

并做：

    direction low-pass filtering


然后：

    t_est
        =
    [-n_y, n_x, 0]


控制：

    probe velocity reference

        =
    tangent_speed * t_est
        +
    normal_velocity * n_est


工具姿态：

    tool axis
        ->
    n_est


============================================================
重要
============================================================

真实曲面函数仍然存在。

但是只能用于：

    1. MuJoCo环境生成
    2. Ground-truth evaluation

不能进入控制器。


代码里会明确标记：

    HIDDEN ENVIRONMENT ONLY
    仅环境真值

和：

    CONTROLLER
    控制器


============================================================
当前版本限制
============================================================

这是73A。

暂时假设：

1. 曲面变化只发生在WORLD XY平面
2. 已知初始粗略接近方向
3. 低摩擦
4. 单主要接触
5. 低速运动

后续73B再加入：

    high friction
    friction bias
    improved normal estimation
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
    "ur3_ft300_unknown_surface.xml"
)


# ============================================================
# 2. 初始粗略接近方向
#
# 注意：
#
# 控制器允许知道初始大致朝向。
#
# 但不知道之后曲面如何变化。
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
# 3. 数学工具
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

    norm = np.linalg.norm(v)


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
# !!! 以下信息控制器禁止使用 !!!
# ============================================================
#
# 机器人不知道：
#
#     curve amplitude
#     curve length
#     true slope
#     true normal
#
# 这些只用于：
#
#     MuJoCo创建真实环境
#     最后评价估计误差
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
            np.cos(phase)
        )
    )


    slope = (
        hidden_curve_amplitude
        *
        np.pi
        /
        hidden_curve_length
        *
        np.sin(phase)
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
# 5. Load base model
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


base_home_key_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)


if base_home_key_id < 0:

    raise RuntimeError(
        "原始模型找不到home keyframe"
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
# 6. Initial sensor / robot geometry
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
# 7. Tool geometry
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
# 8. Hidden environment geometry
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
# 9. Generate MuJoCo environment
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
# 10. Visual tool rod
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "unknown_surface_tool_rod",

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
# 11. Green probe
#
# 低摩擦：
#
# 这是73A的重要假设。
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "unknown_surface_probe",

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
# 12. Hidden curved surface
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
                f"hidden_curve_{index}",

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
# 13. Load experiment model
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
# 14. Joint / actuator mapping
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
# 15. Task frame
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
    for body_id in payload_body_ids
)


# ============================================================
# 18. Reset
# ============================================================

home_key_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
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
# 19. Initial Pinocchio pose
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
# 20. FT300 helpers
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
# 21. Bias + measurement filter
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


wrench_cutoff_frequency = 10.0


wrench_tau = (
    1.0
    /
    (
        2.0
        *
        np.pi
        *
        wrench_cutoff_frequency
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


filter_initialized = False


def low_pass_wrench(
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
# 22. ========================================================
# CONTROLLER
# 在线法向估计器
#
# 从这里开始：
#
# 禁止使用任何 hidden_surface_xxx()
# ============================================================

estimated_normal = (
    initial_normal_world.copy()
)


previous_estimated_tangent = (
    initial_tangent_world.copy()
)


normal_force_threshold = 0.8


normal_estimator_cutoff = 5.0


normal_estimator_tau = (
    1.0
    /
    (
        2.0
        *
        np.pi
        *
        normal_estimator_cutoff
    )
)


normal_estimator_alpha = (
    normal_estimator_tau
    /
    (
        normal_estimator_tau
        +
        dt
    )
)


def update_normal_estimate(
    external_force_world,
):

    global estimated_normal
    global previous_estimated_tangent


    # --------------------------------------------------------
    # 73A只估计XY平面的法向
    # --------------------------------------------------------

    planar_force = np.array([
        external_force_world[0],
        external_force_world[1],
        0.0,
    ])


    planar_force_norm = (
        np.linalg.norm(
            planar_force
        )
    )


    # --------------------------------------------------------
    # 力足够大才能更新法向
    # --------------------------------------------------------

    if (
        planar_force_norm
        >
        normal_force_threshold
    ):

        raw_normal = (
            -planar_force
            /
            planar_force_norm
        )


        # ----------------------------------------------------
        # 防止法向突然翻转180°
        # ----------------------------------------------------

        if (
            np.dot(
                raw_normal,
                estimated_normal,
            )
            <
            0.0
        ):

            raw_normal = (
                -raw_normal
            )


        filtered_direction = (
            normal_estimator_alpha
            *
            estimated_normal

            +

            (
                1.0
                -
                normal_estimator_alpha
            )
            *
            raw_normal
        )


        estimated_normal = normalize(
            filtered_direction
        )


    # --------------------------------------------------------
    # 根据估计法向构造二维切向
    # --------------------------------------------------------

    estimated_tangent = np.array([
        -estimated_normal[1],
        estimated_normal[0],
        0.0,
    ])


    estimated_tangent = normalize(
        estimated_tangent
    )


    # --------------------------------------------------------
    # 保持切向符号连续
    # --------------------------------------------------------

    if (
        np.dot(
            estimated_tangent,
            previous_estimated_tangent,
        )
        <
        0.0
    ):

        estimated_tangent = (
            -estimated_tangent
        )


    previous_estimated_tangent = (
        estimated_tangent.copy()
    )


    return (
        estimated_normal.copy(),
        estimated_tangent.copy(),
    )


# ============================================================
# 23. Smoothstep
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
# 24. Timeline
# ============================================================

approach_start = 1.0

approach_duration = 2.0

approach_distance = 0.032


force_ramp_start = 3.0

force_ramp_duration = 1.5


maximum_normal_force = 5.0


slide_start = 5.0

slide_duration = 10.0

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
# 25. Dynamic force controller
# ============================================================

force_virtual_mass = 2.0

force_virtual_damping = 120.0


normal_velocity_limit = 0.004

normal_acceleration_limit = 0.20


normal_velocity = 0.0

normal_acceleration = 0.0


# 从3秒以后累计多少法向搜索位移
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


    if (
        proposed_displacement
        >
        normal_search_max
    ):

        normal_search_displacement = (
            normal_search_max
        )


        if normal_velocity > 0.0:

            normal_velocity = 0.0


    elif (
        proposed_displacement
        <
        normal_search_min
    ):

        normal_search_displacement = (
            normal_search_min
        )


        if normal_velocity < 0.0:

            normal_velocity = 0.0


    else:

        normal_search_displacement = (
            proposed_displacement
        )


# ============================================================
# 26. TCP reference state
#
# 与72完全不同：
#
# 不知道曲面几何，所以不能：
#
#     p_des = known_curve(s)
#
# 而必须在线积分。
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


reference_accel_cutoff = 5.0


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
# 27. Estimated orientation state
# ============================================================

previous_theta_estimated = 0.0


estimated_angular_velocity = 0.0


angular_rate_cutoff = 5.0


angular_rate_tau = (
    1.0
    /
    (
        2.0
        *
        np.pi
        *
        angular_rate_cutoff
    )
)


angular_rate_alpha = (
    angular_rate_tau
    /
    (
        angular_rate_tau
        +
        dt
    )
)


# ============================================================
# 28. Cartesian gains
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
# 29. TCP Cartesian controller
# ============================================================

def tcp_cartesian_controller(
    external_world_wrench,
    desired_probe_position,
    desired_rotation,
    desired_linear_velocity,
    desired_linear_acceleration,
    desired_angular_velocity,
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
    # Shift to TCP
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
    # Errors
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
    # Cartesian acceleration
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


    task_acceleration_command = np.concatenate([
        linear_acceleration_command,
        angular_acceleration_command,
    ])


    # --------------------------------------------------------
    # 6D solve
    # --------------------------------------------------------

    J_tcp_arm = (
        J_tcp[
            :,
            arm_pin_v_indices
        ]
    )


    rhs = (
        task_acceleration_command

        -

        dJ_tcp
        @
        v_pin
    )


    singular_values = np.linalg.svd(
        J_tcp_arm,
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
                J_tcp_arm,
                rhs,
            )


            solver_name = (
                "Exact 精确求解"
            )


        except np.linalg.LinAlgError:

            use_exact = False


    if not use_exact:

        matrix = (
            J_tcp_arm
            @
            J_tcp_arm.T

            +

            (
                dls_lambda**2
            )
            *
            np.eye(
                6
            )
        )


        qdd_arm = (
            J_tcp_arm.T
            @
            np.linalg.solve(
                matrix,
                rhs,
            )
        )


        solver_name = (
            "DLS 阻尼最小二乘"
        )


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
    # External wrench compensation
    #
    # wrench still referenced at FT/task origin
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
    # Passive compensation
    # --------------------------------------------------------

    saturation_count = 0

    max_motor_usage = 0.0


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

            saturation_count += 1


        torque_limit = max(
            abs(
                lower
            ),
            abs(
                upper
            ),
        )


        if torque_limit > 1e-12:

            usage = (
                abs(
                    requested_torque
                )
                /
                torque_limit
            )


            max_motor_usage = max(
                max_motor_usage,
                usage,
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
        ] = 0.0


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    tcp_error = float(
        np.linalg.norm(
            position_error
        )
    )


    actual_tool_axis = (
        R_task
        @
        tool_axis_task_local
    )


    actual_tool_axis = normalize(
        actual_tool_axis
    )


    return (
        current_probe_position,
        R_task,
        actual_tool_axis,
        tcp_error,
        condition,
        solver_name,
        saturation_count,
        100.0
        *
        max_motor_usage,
    )


# ============================================================
# 30. ========================================================
# GROUND-TRUTH EVALUATION ONLY
# 真值只用于评价
#
# 绝对不能反馈给控制器。
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
# 31. Main
# ============================================================

step = 0


print(
    "\n"
    "========== Unknown Surface Tracking "
    "未知曲面跟踪 =========="
)


print(
    f"Desired force "
    f"目标法向力: "
    f"{maximum_normal_force:.1f} N"
)


print(
    "Surface model in controller "
    "控制器曲面模型: NONE 无"
)


print(
    "Normal source "
    "法向来源: FT300 force direction "
    "FT300力方向"
)


print(
    "Surface friction "
    "曲面摩擦系数: 0.05"
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
        # A. FT300
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


            bias_count += 1


        if (
            bias_wrench is None
            and
            current_time >= 1.0
        ):

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

            compensated = (
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
                low_pass_wrench(
                    compensated
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
        # B. CONTROLLER:
        #    estimate surface normal from force
        # ====================================================

        (
            normal_est,
            tangent_est,
        ) = update_normal_estimate(
            external_world_wrench[
                :3
            ]
        )


        # ====================================================
        # C. Normal force
        #
        # 只使用估计法向。
        # ====================================================

        measured_normal_force = max(
            0.0,

            -float(
                np.dot(
                    external_world_wrench[
                        :3
                    ],
                    normal_est,
                )
            ),
        )


        desired_force = (
            desired_force_at_time(
                current_time
            )
        )


        # ====================================================
        # D. Initial approach
        # ====================================================

        if current_time < 1.0:

            probe_reference_position = (
                probe_position_world_initial.copy()
            )


            normal_velocity = 0.0

            normal_acceleration = 0.0


            desired_reference_velocity = np.zeros(
                3
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


            normal_velocity = 0.0

            normal_acceleration = 0.0


        # ====================================================
        # E. Unknown-surface following
        #
        # 完全不使用真实曲面。
        # ====================================================

        else:

            update_force_controller(
                desired_force,
                measured_normal_force,
            )


            (
                tangential_distance_command,
                tangential_speed,
                _,
            ) = tangential_motion(
                current_time
            )


            desired_reference_velocity = (
                tangent_est
                *
                tangential_speed

                +

                normal_est
                *
                normal_velocity
            )


            probe_reference_position += (
                desired_reference_velocity
                *
                dt
            )


        # ====================================================
        # F. Reference acceleration
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
        # G. Tool orientation from ESTIMATED normal
        # ====================================================

        normal_world_angle = np.arctan2(
            normal_est[
                1
            ],
            normal_est[
                0
            ],
        )


        theta_estimated = wrap_to_pi(
            normal_world_angle

            -

            initial_surface_angle
        )


        desired_rotation = (
            rotation_z(
                theta_estimated
            )

            @

            R_initial
        )


        # ----------------------------------------------------
        # Estimate angular velocity from filtered normal angle
        # ----------------------------------------------------

        theta_delta = wrap_to_pi(
            theta_estimated

            -

            previous_theta_estimated
        )


        raw_theta_dot = (
            theta_delta
            /
            dt
        )


        estimated_angular_velocity = (
            angular_rate_alpha
            *
            estimated_angular_velocity

            +

            (
                1.0
                -
                angular_rate_alpha
            )
            *
            raw_theta_dot
        )


        estimated_angular_velocity = float(
            np.clip(
                estimated_angular_velocity,
                -1.0,
                1.0,
            )
        )


        previous_theta_estimated = (
            theta_estimated
        )


        desired_angular_velocity = np.array([
            0.0,
            0.0,
            estimated_angular_velocity,
        ])


        # ====================================================
        # H. 6D TCP controller
        # ====================================================

        (
            current_probe_position,
            current_rotation,
            actual_tool_axis,
            tcp_tracking_error,
            jacobian_condition,
            solver_name,
            saturation_count,
            motor_usage,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            probe_reference_position,
            desired_rotation,
            desired_reference_velocity,
            reference_acceleration,
            desired_angular_velocity,
        )


        # ====================================================
        # I. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # J. Ground-truth evaluation
        #
        # 注意：
        #
        # 这里才允许读取隐藏真实曲面。
        #
        # 下面的数据绝对没有反馈回控制器。
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            (
                true_s_eval,
                true_normal_eval,
            ) = evaluate_true_normal(
                current_probe_position
            )


            estimated_normal_angle_deg = np.rad2deg(
                wrap_to_pi(
                    np.arctan2(
                        normal_est[
                            1
                        ],
                        normal_est[
                            0
                        ],
                    )

                    -

                    initial_surface_angle
                )
            )


            true_normal_angle_deg = np.rad2deg(
                wrap_to_pi(
                    np.arctan2(
                        true_normal_eval[
                            1
                        ],
                        true_normal_eval[
                            0
                        ],
                    )

                    -

                    initial_surface_angle
                )
            )


            normal_alignment = float(
                np.clip(
                    np.dot(
                        normal_est,
                        true_normal_eval,
                    ),
                    -1.0,
                    1.0,
                )
            )


            normal_estimation_error_deg = np.rad2deg(
                np.arccos(
                    normal_alignment
                )
            )


            tool_alignment = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        normal_est,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_normal_error_deg = np.rad2deg(
                np.arccos(
                    tool_alignment
                )
            )


            (
                tangential_distance_command,
                _,
                _,
            ) = tangential_motion(
                current_time
            )


            print(
                "\n"
                "========== Unknown Surface "
                "未知曲面实验 =========="
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
                f"Estimated normal "
                f"估计法向角: "
                f"{estimated_normal_angle_deg:+.2f} deg"
            )


            print(
                f"True normal "
                f"真实法向角(仅评估): "
                f"{true_normal_angle_deg:+.2f} deg"
            )


            print(
                f"Normal estimate error "
                f"法向估计误差: "
                f"{normal_estimation_error_deg:.2f} deg"
            )


            print(
                f"Tool-normal error "
                f"工具-估计法向误差: "
                f"{tool_normal_error_deg:.2f} deg"
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
                f"Jacobian condition "
                f"TCP雅可比条件数: "
                f"{jacobian_condition:.2f}"
            )


            print(
                f"Motor saturation "
                f"电机饱和数量: "
                f"{saturation_count}"
            )


        step += 1


        time.sleep(
            dt
        )
