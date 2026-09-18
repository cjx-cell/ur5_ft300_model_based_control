#!/usr/bin/env python3

"""
74_ur3_contact_path_learning.py


74A
Contact Path Learning
Low-Force Unknown-Surface Scanning

接触路径学习：
低力扫描未知曲面


============================================================
目标
============================================================

第一遍不进行正式加工。

机器人：

    1. 以低接触力接触未知曲面
    2. 工具姿态固定
    3. 缓慢沿表面移动
    4. 记录实际TCP球心轨迹
    5. 扫描结束以后统一拟合轨迹
    6. 从轨迹导数计算切向和法向


输出：

    74_learned_surface_path.npz
    74_learned_surface_path.csv


============================================================
重要
============================================================

隐藏曲面函数：

    hidden_surface_xxx()

只用于：

    1. MuJoCo创建环境
    2. 最终ground-truth评价

绝不用于：

    扫描控制
    路径拟合
    法向计算


============================================================
为什么工具姿态固定
============================================================

73A已经验证：

    fixed orientation
    + FT normal estimate
    + contact following

可以稳定运行。

所以扫描阶段不再让FT估计直接驱动工具姿态。


============================================================
第二遍74B
============================================================

74B将读取这里生成的：

    learned path
    learned tangent
    learned normal

然后进行：

    5N恒力
    +
    工具法向跟随
    +
    TCP轨迹回放
"""


from pathlib import Path
import time
import csv
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
    "ur3_ft300_path_learning.xml"
)


LEARNED_PATH_NPZ = (
    SCRIPT_DIR
    /
    "74_learned_surface_path.npz"
)


LEARNED_PATH_CSV = (
    SCRIPT_DIR
    /
    "74_learned_surface_path.csv"
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

        return (
            v.copy()
        )

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
# 控制器和路径学习绝对不能调用这些函数。
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
# 7. Initial robot geometry
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
# 8. Virtual scanning tool
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
# 11. Gray tool rod
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "path_learning_tool_rod",

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
# 12. Green spherical probe
# ============================================================

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name":
            "path_learning_probe",

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

        # Low friction scanning
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
                f"path_learning_curve_{index}",

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
# 16. Pinocchio task frame
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
# 19. Reset
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


probe_offset_task_local = (
    R_initial.T

    @

    (
        probe_position_world_initial

        -

        p_task_initial
    )
)


# ============================================================
# 21. FT300 helpers
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
# 22. Bias + FT filtering
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


        wrench_filter_initialized = (
            True
        )


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
# 23. Scan normal estimator
#
# 第一遍只用于低速贴合。
# 不作为最终学习法向。
# ============================================================

estimated_normal_angle = (
    initial_surface_angle
)


estimated_normal = (
    initial_normal_world.copy()
)


normal_update_force_threshold = (
    1.0
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
    20.0
)


def update_scan_normal(
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


    planar_force_norm = (
        np.linalg.norm(
            planar_force
        )
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


        estimated_normal_angle = (
            wrap_to_pi(
                estimated_normal_angle
                +
                filtered_step
            )
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


    tangent_est = np.array([
        -estimated_normal[1],
        estimated_normal[0],
        0.0,
    ])


    tangent_est = normalize(
        tangent_est
    )


    if (
        np.dot(
            tangent_est,
            initial_tangent_world,
        )
        <
        0.0
    ):

        tangent_est = (
            -tangent_est
        )


    return (
        estimated_normal.copy(),
        tangent_est,
        planar_force_norm,
    )


# ============================================================
# 24. Smoothstep
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
# 25. Experiment timeline
# ============================================================

approach_start = 1.0

approach_duration = 2.0

approach_distance = 0.032


force_ramp_start = 3.0

force_ramp_duration = 2.0


# Low-force scan
scan_force = 2.0


scan_start = 6.0

scan_duration = 15.0

scan_end = (
    scan_start
    +
    scan_duration
)


scan_distance = 0.030


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
        scan_force
        *
        progress
    )


def scan_motion(
    t,
):

    (
        progress,
        progress_dot,
        progress_ddot,
    ) = smoothstep(
        t,
        scan_start,
        scan_duration,
    )


    return (
        scan_distance
        *
        progress,

        scan_distance
        *
        progress_dot,

        scan_distance
        *
        progress_ddot,
    )


# ============================================================
# 26. Low-force controller
# ============================================================

force_virtual_mass = 2.0

force_virtual_damping = 120.0


normal_velocity_limit = (
    0.0025
)


normal_acceleration_limit = (
    0.12
)


normal_velocity = 0.0

normal_acceleration = 0.0


def update_force_controller(
    desired_force,
    measured_force,
):

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
        ) < 0.02
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


# ============================================================
# 27. TCP reference state
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


reference_accel_cutoff = (
    4.0
)


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
# 28. Cartesian controller
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


exact_sigma_min_limit = (
    0.05
)


exact_condition_limit = (
    40.0
)


dls_lambda = (
    0.03
)


def tcp_cartesian_controller(
    external_world_wrench,
    desired_probe_position,
    desired_linear_velocity,
    desired_linear_acceleration,
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
    # Actual TCP
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
    # Jacobian
    # ========================================================

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
    # Jdot
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
    # Velocity
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
    # Fixed tool orientation
    # ========================================================

    position_error = (
        desired_probe_position

        -

        current_probe_position
    )


    R_error = (
        R_initial

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


    # ========================================================
    # Exact 6x6 / DLS fallback
    # ========================================================

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

            use_exact = (
                False
            )


    if not use_exact:

        matrix = (
            J_arm

            @

            J_arm.T

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


    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # ========================================================
    # Passive compensation
    # ========================================================

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


    # ========================================================
    # Diagnostics
    # ========================================================

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


    return (
        current_probe_position,
        tcp_error,
        orientation_error_deg,
        condition,
        saturation_count,
    )


# ============================================================
# 29. Scan data storage
# ============================================================

scan_times = []

scan_positions = []

scan_forces = []

scan_normal_estimates = []


# 100 Hz recording
record_period = (
    0.010
)


record_stride = max(
    1,
    int(
        round(
            record_period
            /
            dt
        )
    ),
)


# ============================================================
# 30. Path fitting helpers
# ============================================================

def cumulative_arclength(
    points,
):

    differences = np.diff(
        points,
        axis=0,
    )


    segment_lengths = np.linalg.norm(
        differences,
        axis=1,
    )


    return np.concatenate([
        np.array([
            0.0
        ]),

        np.cumsum(
            segment_lengths
        ),
    ])


def remove_duplicate_points(
    points,
    minimum_distance=1e-6,
):

    if len(
        points
    ) < 2:

        return (
            points.copy()
        )


    keep = [
        True
    ]


    last_point = (
        points[
            0
        ]
    )


    for i in range(
        1,
        len(
            points
        )
    ):

        distance = np.linalg.norm(
            points[
                i
            ]

            -

            last_point
        )


        if distance > minimum_distance:

            keep.append(
                True
            )


            last_point = (
                points[
                    i
                ]
            )


        else:

            keep.append(
                False
            )


    return (
        points[
            np.array(
                keep,
                dtype=bool,
            )
        ]
    )


def local_polynomial_smooth(
    s,
    points,
    half_window=10,
    polynomial_degree=3,
):

    smoothed = np.zeros_like(
        points
    )


    number_of_points = len(
        points
    )


    for i in range(
        number_of_points
    ):

        start = max(
            0,
            i - half_window,
        )


        end = min(
            number_of_points,
            i + half_window + 1,
        )


        local_s = (
            s[
                start:end
            ]

            -

            s[
                i
            ]
        )


        local_points = (
            points[
                start:end
            ]
        )


        degree = min(
            polynomial_degree,
            len(
                local_s
            ) - 1,
        )


        for axis in range(
            3
        ):

            coefficients = np.polyfit(
                local_s,
                local_points[
                    :,
                    axis
                ],
                degree,
            )


            smoothed[
                i,
                axis
            ] = (
                np.polyval(
                    coefficients,
                    0.0,
                )
            )


    return (
        smoothed
    )


def build_learned_path(
    raw_positions,
):

    raw_positions = np.asarray(
        raw_positions,
        dtype=float,
    )


    raw_positions = (
        remove_duplicate_points(
            raw_positions
        )
    )


    if len(
        raw_positions
    ) < 20:

        raise RuntimeError(
            "扫描有效点太少，无法拟合路径"
        )


    raw_s = cumulative_arclength(
        raw_positions
    )


    total_length = float(
        raw_s[
            -1
        ]
    )


    if total_length < 0.005:

        raise RuntimeError(
            "扫描路径长度过短，无法学习曲面"
        )


    # --------------------------------------------------------
    # Uniform resampling
    # --------------------------------------------------------

    learned_point_count = (
        301
    )


    learned_s = np.linspace(
        0.0,
        total_length,
        learned_point_count,
    )


    resampled_points = np.zeros(
        (
            learned_point_count,
            3,
        )
    )


    for axis in range(
        3
    ):

        resampled_points[
            :,
            axis
        ] = np.interp(
            learned_s,
            raw_s,
            raw_positions[
                :,
                axis
            ],
        )


    # --------------------------------------------------------
    # Local polynomial smoothing
    # --------------------------------------------------------

    learned_positions = (
        local_polynomial_smooth(
            learned_s,
            resampled_points,
            half_window=10,
            polynomial_degree=3,
        )
    )


    # --------------------------------------------------------
    # Derivative wrt arc length
    # --------------------------------------------------------

    derivatives = np.gradient(
        learned_positions,
        learned_s,
        axis=0,
        edge_order=2,
    )


    learned_tangents = np.zeros_like(
        derivatives
    )


    learned_normals = np.zeros_like(
        derivatives
    )


    previous_tangent = (
        initial_tangent_world.copy()
    )


    previous_normal = (
        initial_normal_world.copy()
    )


    for i in range(
        learned_point_count
    ):

        tangent = (
            derivatives[
                i
            ].copy()
        )


        # Current course assumes XY surface.
        tangent[
            2
        ] = (
            0.0
        )


        tangent = normalize(
            tangent
        )


        if (
            np.dot(
                tangent,
                previous_tangent,
            )
            <
            0.0
        ):

            tangent = (
                -tangent
            )


        normal = np.array([
            tangent[
                1
            ],
            -tangent[
                0
            ],
            0.0,
        ])


        normal = normalize(
            normal
        )


        if (
            np.dot(
                normal,
                previous_normal,
            )
            <
            0.0
        ):

            normal = (
                -normal
            )


        learned_tangents[
            i
        ] = (
            tangent
        )


        learned_normals[
            i
        ] = (
            normal
        )


        previous_tangent = (
            tangent
        )


        previous_normal = (
            normal
        )


    return (
        learned_s,
        learned_positions,
        learned_tangents,
        learned_normals,
    )


# ============================================================
# 31. Ground-truth evaluation only
# ============================================================

evaluation_s = np.linspace(
    0.0,
    hidden_curve_length,
    2001,
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


evaluation_centers = np.asarray(
    evaluation_centers
)


evaluation_normals = np.asarray(
    evaluation_normals
)


def evaluate_learned_path(
    learned_positions,
    learned_normals,
):

    position_errors = []

    normal_errors_deg = []


    for (
        position,
        normal,
    ) in zip(
        learned_positions,
        learned_normals,
    ):

        distances = np.linalg.norm(
            evaluation_centers

            -

            position[
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


        position_errors.append(
            distances[
                index
            ]
        )


        true_normal = (
            evaluation_normals[
                index
            ]
        )


        alignment = float(
            np.clip(
                np.dot(
                    normal,
                    true_normal,
                ),
                -1.0,
                1.0,
            )
        )


        normal_errors_deg.append(
            np.rad2deg(
                np.arccos(
                    alignment
                )
            )
        )


    return (
        np.asarray(
            position_errors
        ),
        np.asarray(
            normal_errors_deg
        ),
    )


# ============================================================
# 32. Save learned path
# ============================================================

path_saved = False


def save_learned_path():

    global path_saved


    if path_saved:

        return


    (
        learned_s,
        learned_positions,
        learned_tangents,
        learned_normals,
    ) = build_learned_path(
        scan_positions
    )


    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez(
        LEARNED_PATH_NPZ,

        s=learned_s,

        position=learned_positions,

        tangent=learned_tangents,

        normal=learned_normals,

        probe_radius=np.array([
            probe_radius
        ]),

        tool_length=np.array([
            tool_length
        ]),

        initial_surface_angle=np.array([
            initial_surface_angle
        ]),
    )


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        LEARNED_PATH_CSV,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )


        writer.writerow([
            "s_m",
            "x_m",
            "y_m",
            "z_m",
            "tx",
            "ty",
            "tz",
            "nx",
            "ny",
            "nz",
        ])


        for i in range(
            len(
                learned_s
            )
        ):

            writer.writerow([
                learned_s[
                    i
                ],

                *learned_positions[
                    i
                ],

                *learned_tangents[
                    i
                ],

                *learned_normals[
                    i
                ],
            ])


    # --------------------------------------------------------
    # Evaluation only
    # --------------------------------------------------------

    (
        position_errors,
        normal_errors_deg,
    ) = evaluate_learned_path(
        learned_positions,
        learned_normals,
    )


    scan_force_array = np.asarray(
        scan_forces,
        dtype=float,
    )


    force_error = np.abs(
        scan_force_array

        -

        scan_force
    )


    print(
        "\n"
        "========== Path Learning Result "
        "路径学习结果 =========="
    )


    print(
        f"Raw samples 原始扫描点: "
        f"{len(scan_positions)}"
    )


    print(
        f"Learned points 拟合路径点: "
        f"{len(learned_s)}"
    )


    print(
        f"Learned path length "
        f"学习路径长度: "
        f"{learned_s[-1] * 1000.0:.2f} mm"
    )


    print(
        f"Scan force mean error "
        f"扫描力平均误差: "
        f"{np.mean(force_error):.3f} N"
    )


    print(
        f"Path mean/max error "
        f"路径平均/最大误差(仅评估): "
        f"{np.mean(position_errors) * 1000.0:.3f} / "
        f"{np.max(position_errors) * 1000.0:.3f} mm"
    )


    print(
        f"Normal mean/max error "
        f"法向平均/最大误差(仅评估): "
        f"{np.mean(normal_errors_deg):.2f} / "
        f"{np.max(normal_errors_deg):.2f} deg"
    )


    print(
        "Learned path save "
        "学习路径保存: completed 完成"
    )


    print(
        f"NPZ: "
        f"{LEARNED_PATH_NPZ}"
    )


    print(
        f"CSV: "
        f"{LEARNED_PATH_CSV}"
    )


    path_saved = (
        True
    )


# ============================================================
# 33. Main
# ============================================================

step = 0


print(
    "\n"
    "========== 74A Contact Path Learning "
    "74A接触路径学习 =========="
)


print(
    f"Scan force "
    f"扫描接触力: "
    f"{scan_force:.1f} N"
)


print(
    f"Scan distance "
    f"扫描距离: "
    f"{scan_distance * 1000.0:.1f} mm"
)


print(
    f"Scan duration "
    f"扫描时间: "
    f"{scan_duration:.1f} s"
)


print(
    "Tool orientation "
    "工具姿态: fixed 固定"
)


print(
    "Surface model "
    "控制器曲面模型: NONE 无"
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
        # Wrench processing
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
        # B. Scan normal estimate
        # ====================================================

        allow_normal_update = (
            current_time >= 4.5

            and

            current_time <= (
                scan_end
                +
                0.1
            )
        )


        (
            normal_est,
            tangent_est,
            planar_force_magnitude,
        ) = update_scan_normal(
            external_world_wrench[
                :3
            ],
            allow_normal_update,
        )


        measured_force = (
            planar_force_magnitude
        )


        desired_force = (
            desired_force_at_time(
                current_time
            )
        )


        # ====================================================
        # C. Reference generation
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
                measured_force,
            )


            (
                scan_progress_distance,
                tangential_speed,
                _,
            ) = scan_motion(
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
        # D. Reference acceleration
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
        # E. TCP controller
        # ====================================================

        (
            current_probe_position,
            tcp_tracking_error,
            orientation_tracking_error,
            jacobian_condition,
            saturation_count,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            probe_reference_position,
            desired_reference_velocity,
            reference_acceleration,
        )


        # ====================================================
        # F. Record ACTUAL TCP trajectory
        # ====================================================

        if (
            scan_start
            <= current_time
            <= scan_end

            and

            step
            %
            record_stride
            ==
            0
        ):

            scan_times.append(
                current_time
            )


            scan_positions.append(
                current_probe_position.copy()
            )


            scan_forces.append(
                measured_force
            )


            scan_normal_estimates.append(
                normal_est.copy()
            )


        # ====================================================
        # G. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # H. Save once after scan
        # ====================================================

        if (
            not path_saved

            and

            current_time
            >=
            (
                scan_end
                +
                0.05
            )
        ):

            save_learned_path()


        # ====================================================
        # I. Compact output
        # ====================================================

        if (
            step >= 1000

            and

            step % 1000 == 0
        ):

            (
                scan_progress_distance,
                _,
                _,
            ) = scan_motion(
                current_time
            )


            estimated_relative_angle_deg = (
                np.rad2deg(
                    wrap_to_pi(
                        estimated_normal_angle

                        -

                        initial_surface_angle
                    )
                )
            )


            print(
                "\n"
                "========== 74A Scan "
                "74A扫描 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Force 目标/实际接触力: "
                f"{desired_force:.2f} / "
                f"{measured_force:.2f} N"
            )


            print(
                f"Scan travel "
                f"扫描指令距离: "
                f"{scan_progress_distance * 1000.0:.1f} mm"
            )


            print(
                f"Scan normal "
                f"扫描阶段估计法向: "
                f"{estimated_relative_angle_deg:+.2f} deg"
            )


            print(
                f"TCP error "
                f"TCP跟踪误差: "
                f"{tcp_tracking_error * 1000.0:.3f} mm"
            )


            print(
                f"Orientation error "
                f"固定姿态误差: "
                f"{orientation_tracking_error:.3f} deg"
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
