#!/usr/bin/env python3

"""
72_ur3_curved_surface_tcp_passive_comp.py


Known Curved Surface
TCP Force + Pose Control
with MuJoCo Passive Force Compensation

已知曲面：
TCP恒力 + 姿态跟踪 + MuJoCo被动力补偿


============================================================
控制目标
============================================================

绿色球头：

    作为真正 TCP

沿曲面移动：

    30 mm

曲面局部法向：

    5 N

工具杆：

    始终跟随局部曲面法向


============================================================
核心结构
============================================================

曲面几何
    ↓
TCP目标位置 p_tcp,d
工具目标姿态 R_d
    ↓
6D Cartesian acceleration controller
    ↓
TCP Jacobian
    ↓
qdd_des
    ↓
Pinocchio RNEA
    ↓
tau_rnea


MuJoCo被动力补偿：

    tau_motor
        =
    tau_rnea
        -
    qfrc_passive


接触外力补偿：

    tau_motor
        -=
    J_sensor^T W_external


============================================================
重要
============================================================

绿色球头：

    radius = 5 mm

工具长度：

    120 mm

姿态加速度诊断限幅：

    80 rad/s²

TCP线加速度诊断限幅：

    30 m/s²


这些大限幅目前是仿真诊断参数，
不是实际UR3工程参数。
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
    "ur3_ft300_robotiq_curved_surface_tcp_passive.xml"
)


# ============================================================
# 2. Base surface frame
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
# 3. Curve
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
# 4. Local surface frame
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
# 5. Surface normal angle
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
# 6. Normal angle derivatives
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
# 7. Math helpers
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


def skew(vector):

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
# 8. Load base model
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
# 9. Reset base model
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
# 10. Initial FT300 position
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
# 11. Robotiq body
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
# 12. Virtual tool
# ============================================================

tool_length = 0.120

tool_rod_radius = 0.003

# ------------------------------------------------------------
# 5 mm radius
# ------------------------------------------------------------

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
# 13. Initial gap
# ============================================================

initial_gap = 0.035


contact_surface_distance = (
    initial_gap
    +
    probe_radius
)


# ============================================================
# 14. Surface point
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
# 15. Ideal probe center at contact
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
# 16. Generate MuJoCo environment
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
        "MJCF找不到 worldbody"
    )


if robotiq_body_element is None:

    raise RuntimeError(
        "MJCF找不到 robotiq_85_base_link"
    )


# ============================================================
# 17. Gray tool rod
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "passive_comp_tool_rod",

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
# 18. Green probe
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "passive_comp_probe",

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
# 19. Curved surface collision geometry
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


    point = (
        surface_point_world(
            s_value
        )
    )


    center = (
        point

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
                f"passive_curve_{index}",

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
                    f"{center[0]:.9f} "
                    f"{center[1]:.9f} "
                    f"{center[2]:.9f}"
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
# 20. Load contact model
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
# 22. Pinocchio task frame
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
        "Pinocchio找不到 robotiq_ft_frame_id"
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
# 25. Reset contact model
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
# 26. Initial Pinocchio pose
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
# 27. TCP offset
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

    base_normal_world
)


tool_axis_task_local /= np.linalg.norm(
    tool_axis_task_local
)


# ============================================================
# 28. FT300 raw
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


# ============================================================
# 29. Payload gravity compensation
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
# 30. Sensor wrench -> world external wrench
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

        wrench_sensor[
            :3
        ]
    )


    torque_world = (
        -R_sensor_world

        @

        wrench_sensor[
            3:
        ]
    )


    return np.concatenate([
        force_world,
        torque_world,
    ])


# ============================================================
# 31. Bias
# ============================================================

bias_sum = np.zeros(
    6
)


bias_count = 0

bias_wrench = None


# ============================================================
# 32. Noise + LPF
# ============================================================

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
# 34. Timeline
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
# 35. Dynamic normal force controller
# ============================================================

force_virtual_mass = 2.0

force_virtual_damping = 120.0


force_velocity_limit = 0.006

force_acceleration_limit = 0.20


normal_correction_min = -0.006

normal_correction_max = +0.006


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
        ) < 0.03
    ):

        force_error = 0.0


    # --------------------------------------------------------
    # Mf * a + Df * v = force error
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


    if new_correction > normal_correction_max:

        normal_correction = (
            normal_correction_max
        )


        if normal_velocity > 0.0:

            normal_velocity = 0.0


    elif new_correction < normal_correction_min:

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
# 36. Desired TCP position
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
# 37. TCP path derivatives
# ============================================================

geometry_epsilon = 1e-4


def probe_geometry_derivatives(
    s,
    correction,
):

    eps = geometry_epsilon


    p_minus = (
        desired_probe_position(
            s - eps,
            correction,
        )
    )


    p_center = (
        desired_probe_position(
            s,
            correction,
        )
    )


    p_plus = (
        desired_probe_position(
            s + eps,
            correction,
        )
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
# 38. Desired orientation
# ============================================================

def desired_orientation(
    s,
):

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
# 39. Cartesian gains
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


linear_acceleration_limit = (
    30.0
)


angular_acceleration_limit = (
    80.0
)


# ============================================================
# 40. Exact / DLS fallback
# ============================================================

exact_sigma_min_limit = 0.05

exact_condition_limit = 40.0

dls_lambda = 0.03


# ============================================================
# 41. TCP Cartesian controller
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

    # ========================================================
    # State
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
    # Kinematics
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
    # TCP
    # ========================================================

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


    # ========================================================
    # Frame Jacobian
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
    # Shift Jacobian to TCP
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
    # TCP Jdot
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
    # Current TCP velocity
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
    # Errors
    # ========================================================

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


    # ========================================================
    # Cartesian acceleration command
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
    # Arm Jacobian
    # ========================================================

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


    # ========================================================
    # Jacobian diagnostics
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
    # Exact / DLS
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
                J_tcp_arm,
                rhs,
            )


            solver_name = (
                "Exact 精确求解"
            )


        except np.linalg.LinAlgError:

            use_exact_solver = False


    if not use_exact_solver:

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


    # ========================================================
    # Full qdd
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
    # RNEA
    # ========================================================

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )


    # ========================================================
    # External wrench compensation
    #
    # FT wrench is measured at sensor/task origin.
    # Therefore use J_task, not J_tcp.
    # ========================================================

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


    # ========================================================
    # Pinocchio -> MuJoCo
    # ========================================================

    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # ========================================================
    # Actuator torque
    #
    # Critical:
    #
    # tau_motor
    #
    # =
    #
    # tau_rnea/external-comp
    #
    # -
    #
    # qfrc_passive
    # ========================================================

    saturation_count = 0

    max_motor_usage = 0.0

    max_passive_torque = 0.0


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


        desired_generalized_torque = float(
            tau_mj[
                dof_id
            ]
        )


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
        # ----------------------------------------------------

        requested_torque = (
            desired_generalized_torque

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
        ] = (
            0.0
        )


    # ========================================================
    # Diagnostics
    # ========================================================

    probe_error = float(
        np.linalg.norm(
            position_error
        )
    )


    actual_tool_axis = (
        R_task

        @

        tool_axis_task_local
    )


    actual_tool_axis /= np.linalg.norm(
        actual_tool_axis
    )


    desired_tool_axis = (
        desired_rotation

        @

        tool_axis_task_local
    )


    desired_tool_axis /= np.linalg.norm(
        desired_tool_axis
    )


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


    tool_normal_error_deg = (
        np.rad2deg(
            np.arccos(
                alignment
            )
        )
    )


    return (
        probe_error,
        tool_normal_error_deg,
        jacobian_condition,
        solver_name,
        saturation_count,
        100.0
        *
        max_motor_usage,
        max_passive_torque,
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
            "TCP force-pose tracking "
            "TCP恒力姿态跟踪"
        )


    return (
        "Force hold 恒力保持"
    )


# ============================================================
# 43. Main
# ============================================================

step = 0


print(
    "\n"
    "========== TCP Force + Pose "
    "TCP恒力姿态曲面控制 =========="
)


print(
    f"Desired force "
    f"目标法向力: "
    f"{maximum_normal_force:.1f} N"
)


print(
    f"Probe radius "
    f"球头半径: "
    f"{probe_radius * 1000.0:.1f} mm"
)


print(
    f"Tool length "
    f"工具长度: "
    f"{tool_length * 1000.0:.0f} mm"
)


print(
    "Passive compensation "
    "被动力补偿: enabled 已启用"
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
        # A. Surface trajectory
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


            bias_count += 1


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


        # ====================================================
        # C. FT signal processing
        # ====================================================

        if bias_wrench is None:

            external_world_wrench = np.zeros(
                6
            )


        else:

            compensated_sensor_wrench = (
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
                    compensated_sensor_wrench

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
                    external_world_wrench[
                        :3
                    ],
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
        # E. Approach / force controller
        # ====================================================

        if current_time < 1.0:

            normal_correction = (
                -initial_gap
            )


            normal_velocity = 0.0

            normal_acceleration = 0.0


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
            desired_probe_position_world,
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


        # ====================================================
        # H. Orientation trajectory
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
            curve_s_dot
            *
            curve_s_dot

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


        # ====================================================
        # I. Controller
        # ====================================================

        (
            probe_tracking_error,
            tool_normal_error,
            jacobian_condition,
            solver_name,
            saturation_count,
            motor_usage,
            max_passive_torque,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            desired_probe_position_world,
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
        # K. Compact bilingual output
        # ====================================================

        if (
            step >= 1000

            and

            step % 1000 == 0
        ):

            normal_angle_deg = (
                np.rad2deg(
                    normal_deflection_angle(
                        curve_s
                    )
                )
            )


            print(
                "\n"
                "========== TCP Surface "
                "TCP曲面实验 =========="
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
                f"{normal_angle_deg:+.2f} deg"
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
                f"Force correction "
                f"力控法向修正: "
                f"{normal_correction * 1000.0:+.2f} mm"
            )


            print(
                f"Passive torque "
                f"最大被动力矩: "
                f"{max_passive_torque:.3f} N·m"
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
                f"{motor_usage:.1f}%"
            )


        step += 1


        time.sleep(
            dt
        )
