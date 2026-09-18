#!/usr/bin/env python3

"""
73_ur3_unknown_surface_slow_pose_tracking.py


73A2
Unknown Surface Normal Estimation
+ Force Tracking
+ Tangential Following
+ Slow Tool-Normal Following

未知曲面：
在线法向估计
+ 5N恒力
+ 未知曲面切向运动
+ 慢速工具法向姿态跟随


============================================================
为什么需要73A2
============================================================

第一次73：

    FT force
        ->
    normal estimate
        ->
    tool orientation immediately

导致：

    工具姿态变化
        ->
    FT300动态力变化
        ->
    normal estimate变化
        ->
    工具继续变化

形成 estimator-controller coupling。


73A：

    固定工具姿态

已经证明：

    法向估计
    力控制
    未知曲面切向运动

可以稳定工作。


73A2：

在73A基础上增加一个独立的
slow orientation reference generator。

不是：

    theta_cmd = theta_est

而是：

    theta_est
        ->
    rate-limited orientation command
        ->
    6D TCP inner loop


============================================================
重要控制层级
============================================================

FT300
    ↓
normal estimator
    ↓
estimated normal angle
    ↓
slow orientation reference generator
    ↓
theta_cmd / omega_cmd / alpha_cmd
    ↓
6D TCP computed torque controller


============================================================
控制器不知道真实曲面
============================================================

真实曲面：

    hidden_surface_xxx()

只能用于：

    1. MuJoCo物理环境
    2. Ground-truth evaluation

绝对不能用于控制器。
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
    "ur3_ft300_unknown_surface_slow_pose.xml"
)


# ============================================================
# 2. 初始粗略接近方向
#
# 控制器知道：
#
# 工件大概位于哪个方向。
#
# 控制器不知道：
#
# 后续曲面的局部法向和曲率。
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
# 以下函数控制器禁止调用。
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


# ============================================================
# 6. Reset base model
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
# 8. Virtual tool
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
# 10. Build MuJoCo environment
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
        "MJCF找不到 worldbody"
    )


if robotiq_body_element is None:

    raise RuntimeError(
        "MJCF找不到 robotiq_85_base_link"
    )


# ============================================================
# 11. Tool rod
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "unknown_slow_pose_tool_rod",

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
# 12. Green probe
#
# 73A2仍使用低摩擦。
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "unknown_slow_pose_probe",

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
# 13. Hidden curved surface
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
        normal,
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
        normal
    )


    yaw = np.arctan2(
        normal[1],
        normal[0],
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
                f"hidden_slow_pose_curve_{index}",

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


# ============================================================
# 15. Joint / actuator mapping
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
            f"MuJoCo找不到关节: {name}"
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
        "Pinocchio找不到 robotiq_ft_frame_id"
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
# 18. Payload bodies
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
# 19. Reset experiment
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


# ============================================================
# 21. TCP offset
# ============================================================

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
# 22. FT helpers
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
# CONTROLLER:
# Surface normal estimator
#
# 禁止调用hidden_surface函数。
# ============================================================

estimated_normal_angle = (
    initial_surface_angle
)


estimated_normal = (
    initial_normal_world.copy()
)


normal_update_force_threshold = (
    3.0
)


normal_estimator_cutoff = (
    2.0
)


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


max_normal_rate = np.deg2rad(
    25.0
)


def update_normal_estimate(
    external_force_world,
    allow_update,
):

    global estimated_normal_angle
    global estimated_normal


    planar_force = np.array([
        external_force_world[0],
        external_force_world[1],
        0.0,
    ])


    planar_force_norm = np.linalg.norm(
        planar_force
    )


    if (
        allow_update
        and
        planar_force_norm
        >
        normal_update_force_threshold
    ):

        raw_normal = (
            -planar_force

            /

            planar_force_norm
        )


        raw_angle = np.arctan2(
            raw_normal[1],
            raw_normal[0],
        )


        angle_error = wrap_to_pi(
            raw_angle

            -

            estimated_normal_angle
        )


        filtered_step = (
            (
                1.0
                -
                normal_estimator_alpha
            )

            *

            angle_error
        )


        max_step = (
            max_normal_rate

            *

            dt
        )


        filtered_step = float(
            np.clip(
                filtered_step,
                -max_step,
                max_step,
            )
        )


        estimated_normal_angle = wrap_to_pi(
            estimated_normal_angle

            +

            filtered_step
        )


        estimated_normal = np.array([
            np.cos(
                estimated_normal_angle
            ),

            np.sin(
                estimated_normal_angle
            ),

            0.0,
        ])


    estimated_tangent = np.array([
        -estimated_normal[1],
        estimated_normal[0],
        0.0,
    ])


    estimated_tangent = normalize(
        estimated_tangent
    )


    if (
        np.dot(
            estimated_tangent,
            initial_tangent_world,
        )
        <
        0.0
    ):

        estimated_tangent = (
            -estimated_tangent
        )


    return (
        estimated_normal.copy(),
        estimated_tangent.copy(),
        planar_force_norm,
    )


# ============================================================
# 25. ========================================================
# Slow tool orientation reference generator
# 慢速工具姿态参考生成器
# ============================================================

tool_command_relative_angle = 0.0

tool_command_angular_velocity = 0.0

tool_command_angular_acceleration = 0.0


# ------------------------------------------------------------
# theta error -> target angular velocity
#
# 单位：
#
# k_theta [1/s]
# ------------------------------------------------------------

tool_orientation_gain = (
    2.0
)


# ------------------------------------------------------------
# 最大工具目标角速度
#
# 比normal estimator最大25deg/s更慢。
# ------------------------------------------------------------

tool_command_max_rate = np.deg2rad(
    15.0
)


# ------------------------------------------------------------
# omega target -> omega command
#
# 相当于一个角速度内层参考滤波。
# ------------------------------------------------------------

tool_rate_gain = (
    10.0
)


# ------------------------------------------------------------
# 最大工具参考角加速度
# ------------------------------------------------------------

tool_command_max_acceleration = np.deg2rad(
    60.0
)


# ------------------------------------------------------------
# 防止小角度噪声导致工具一直动
# ------------------------------------------------------------

tool_angle_deadband = np.deg2rad(
    0.20
)


orientation_follow_start = (
    5.0
)


def update_tool_orientation_reference(
    estimated_relative_angle,
    current_time,
):

    global tool_command_relative_angle
    global tool_command_angular_velocity
    global tool_command_angular_acceleration


    # --------------------------------------------------------
    # 接触力尚未建立时不允许工具姿态动
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Angle error
    # --------------------------------------------------------

    angle_error = wrap_to_pi(
        estimated_relative_angle

        -

        tool_command_relative_angle
    )


    # --------------------------------------------------------
    # Small deadband
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Angular acceleration
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Integrate angular velocity
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Integrate command angle
    # --------------------------------------------------------

    tool_command_relative_angle = (
        tool_command_relative_angle

        +

        tool_command_angular_velocity

        *

        dt
    )


    tool_command_relative_angle = wrap_to_pi(
        tool_command_relative_angle
    )


    return (
        tool_command_relative_angle,
        tool_command_angular_velocity,
        tool_command_angular_acceleration,
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
# 29. TCP reference state
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
# 30. Cartesian controller gains
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
# 31. TCP Cartesian controller
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

    # --------------------------------------------------------
    # State conversion
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
    # Current TCP position
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
    # Position error
    # --------------------------------------------------------

    position_error = (
        desired_probe_position

        -

        current_probe_position
    )


    # --------------------------------------------------------
    # Orientation error
    # --------------------------------------------------------

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
    # Angular acceleration command
    #
    # alpha_cmd
    #
    # =
    #
    # alpha_des
    # +
    # Kp * eR
    # +
    # Kd * (omega_des - omega)
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
    # 6D TCP solve
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

    tcp_error = float(
        np.linalg.norm(
            position_error
        )
    )


    orientation_error_deg = float(
        np.rad2deg(
            np.linalg.norm(
                orientation_error
            )
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
        actual_tool_axis,
        tcp_error,
        orientation_error_deg,
        condition,
        saturation_count,
    )


# ============================================================
# 32. Ground truth evaluation only
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
        normal_true,
        _,
    ) = hidden_surface_frame(
        s_value
    )


    evaluation_normals.append(
        normal_true
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
# 33. Main
# ============================================================

step = 0


print(
    "\n"
    "========== 73A2 Unknown Surface Pose "
    "73A2未知曲面姿态跟踪 =========="
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
    "Tool orientation "
    "工具姿态: slow estimated-normal following "
    "慢速跟随估计法向"
)


print(
    f"Tool max rate "
    f"工具最大目标角速度: "
    f"{np.rad2deg(tool_command_max_rate):.1f} deg/s"
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
        # Bias
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
        # B. Normal estimator
        # ====================================================

        allow_normal_update = (
            current_time >= 4.5
        )


        (
            normal_est,
            tangent_est,
            planar_force_magnitude,
        ) = update_normal_estimate(
            external_world_wrench[
                :3
            ],
            allow_normal_update,
        )


        # ====================================================
        # C. Force
        #
        # 73A2仍用平面合力模长作为第一版力反馈。
        # ====================================================

        measured_normal_force = (
            planar_force_magnitude
        )


        desired_force = (
            desired_force_at_time(
                current_time
            )
        )


        # ====================================================
        # D. Position reference
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
        # E. Reference acceleration
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
        # F. Estimated normal relative angle
        # ====================================================

        estimated_relative_angle = wrap_to_pi(
            estimated_normal_angle

            -

            initial_surface_angle
        )


        # ====================================================
        # G. Slow orientation reference generator
        # ====================================================

        (
            command_relative_angle,
            command_angular_velocity,
            command_angular_acceleration,
        ) = update_tool_orientation_reference(
            estimated_relative_angle,
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
        # H. 6D TCP controller
        # ====================================================

        (
            current_probe_position,
            actual_tool_axis,
            tcp_tracking_error,
            orientation_tracking_error,
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
        # I. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # J. Evaluation output
        # ====================================================

        if (
            step >= 1000

            and

            step % 1000 == 0
        ):

            (
                true_s_eval,
                true_normal,
            ) = evaluate_true_normal(
                current_probe_position
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


            # ------------------------------------------------
            # Normal estimator error
            # ------------------------------------------------

            normal_alignment = float(
                np.clip(
                    np.dot(
                        normal_est,
                        true_normal,
                    ),
                    -1.0,
                    1.0,
                )
            )


            normal_estimation_error = np.rad2deg(
                np.arccos(
                    normal_alignment
                )
            )


            # ------------------------------------------------
            # Actual tool vs estimated normal
            # ------------------------------------------------

            tool_est_alignment = float(
                np.clip(
                    np.dot(
                        actual_tool_axis,
                        normal_est,
                    ),
                    -1.0,
                    1.0,
                )
            )


            tool_est_error = np.rad2deg(
                np.arccos(
                    tool_est_alignment
                )
            )


            # ------------------------------------------------
            # Actual tool vs true normal
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


            tool_true_error = np.rad2deg(
                np.arccos(
                    tool_true_alignment
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
                "========== 73A2 Unknown Surface "
                "73A2未知曲面实验 =========="
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
                f"Normal est/cmd/true "
                f"法向估计/姿态指令/真值: "
                f"{np.rad2deg(estimated_relative_angle):+.2f} / "
                f"{np.rad2deg(command_relative_angle):+.2f} / "
                f"{np.rad2deg(true_relative_angle):+.2f} deg"
            )


            print(
                f"Normal estimate error "
                f"法向估计误差: "
                f"{normal_estimation_error:.2f} deg"
            )


            print(
                f"Tool error est/true "
                f"工具对估计/真实法向误差: "
                f"{tool_est_error:.2f} / "
                f"{tool_true_error:.2f} deg"
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
