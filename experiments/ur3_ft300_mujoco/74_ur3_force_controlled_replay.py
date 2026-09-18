#!/usr/bin/env python3
"""
74_ur3_force_controlled_replay.py

74B：学习路径恒力回放
- 读取 74A 生成的 74_learned_surface_path.npz
- 使用学习到的 TCP 路径与法向
- FT300 只负责 5 N 法向力修正
- 控制器不使用隐藏曲面真值
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

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_XML = SCRIPT_DIR / "ur3_ft300_robotiq_force_control.xml"
CONTACT_XML = SCRIPT_DIR / "ur3_ft300_force_replay.xml"
LEARNED_PATH_FILE = SCRIPT_DIR / "74_learned_surface_path.npz"

if not LEARNED_PATH_FILE.exists():
    raise FileNotFoundError(
        f"找不到学习路径文件: {LEARNED_PATH_FILE}\n请先运行 74A。"
    )

# ============================================================
# 1. Load learned path
# ============================================================

learned_data = np.load(LEARNED_PATH_FILE)

learned_s = np.asarray(learned_data["s"], dtype=float)
learned_positions = np.asarray(learned_data["position"], dtype=float)
learned_tangents = np.asarray(learned_data["tangent"], dtype=float)
learned_normals = np.asarray(learned_data["normal"], dtype=float)

probe_radius = float(learned_data["probe_radius"][0])
tool_length = float(learned_data["tool_length"][0])
initial_surface_angle = float(learned_data["initial_surface_angle"][0])
learned_path_length = float(learned_s[-1])

if learned_s.ndim != 1:
    raise RuntimeError(f"learned_s 维度错误: {learned_s.shape}")

if learned_positions.ndim != 2 or learned_positions.shape[1] != 3:
    raise RuntimeError(f"learned_positions 维度错误: {learned_positions.shape}")

if learned_normals.ndim != 2 or learned_normals.shape[1] != 3:
    raise RuntimeError(f"learned_normals 维度错误: {learned_normals.shape}")

if len(learned_s) != len(learned_positions) or len(learned_s) != len(learned_normals):
    raise RuntimeError("学习路径数组长度不一致")

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
# 2. Math helpers
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
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


# ============================================================
# 3. Learned-path derivatives
#
# 关键修复：
# 整条路径数据一律使用 *_table 名字；
# 当前插值值一律使用 current_* 名字；
# 避免上一版把 (N,3) 数组覆盖成 (3,) 向量。
# ============================================================

learned_position_s_table = np.zeros_like(learned_positions, dtype=float)
learned_position_ss_table = np.zeros_like(learned_positions, dtype=float)
learned_normal_s_table = np.zeros_like(learned_normals, dtype=float)
learned_normal_ss_table = np.zeros_like(learned_normals, dtype=float)

for axis in range(3):
    learned_position_s_table[:, axis] = np.gradient(
        learned_positions[:, axis], learned_s, edge_order=2
    )
    learned_position_ss_table[:, axis] = np.gradient(
        learned_position_s_table[:, axis], learned_s, edge_order=2
    )
    learned_normal_s_table[:, axis] = np.gradient(
        learned_normals[:, axis], learned_s, edge_order=2
    )
    learned_normal_ss_table[:, axis] = np.gradient(
        learned_normal_s_table[:, axis], learned_s, edge_order=2
    )

learned_normal_world_angle_table = np.unwrap(
    np.arctan2(learned_normals[:, 1], learned_normals[:, 0])
)

learned_relative_angle_table = (
    learned_normal_world_angle_table - initial_surface_angle
)

learned_theta_s_table = np.gradient(
    learned_relative_angle_table,
    learned_s,
    edge_order=2,
)

learned_theta_ss_table = np.gradient(
    learned_theta_s_table,
    learned_s,
    edge_order=2,
)

print("Learned data shape 学习数据维度:")
print(f"  position:    {learned_positions.shape}")
print(f"  normal:      {learned_normals.shape}")
print(f"  position_s:  {learned_position_s_table.shape}")
print(f"  normal_s:    {learned_normal_s_table.shape}")


def interpolate_scalar(s_query, values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            f"interpolate_scalar 需要 (N,) 数据，实际得到: {values.shape}"
        )

    s_query = float(np.clip(s_query, learned_s[0], learned_s[-1]))
    return float(np.interp(s_query, learned_s, values))


def interpolate_vector(s_query, values):
    values = np.asarray(values, dtype=float)

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"interpolate_vector 需要 (N,3) 数据，实际得到: {values.shape}"
        )

    s_query = float(np.clip(s_query, learned_s[0], learned_s[-1]))

    return np.array([
        np.interp(s_query, learned_s, values[:, 0]),
        np.interp(s_query, learned_s, values[:, 1]),
        np.interp(s_query, learned_s, values[:, 2]),
    ])


def learned_geometry(s_query):
    nominal_position = interpolate_vector(s_query, learned_positions)
    nominal_position_s = interpolate_vector(s_query, learned_position_s_table)
    nominal_position_ss = interpolate_vector(s_query, learned_position_ss_table)

    current_normal = normalize(
        interpolate_vector(s_query, learned_normals)
    )

    current_normal_s = interpolate_vector(s_query, learned_normal_s_table)
    current_normal_ss = interpolate_vector(s_query, learned_normal_ss_table)

    current_theta = interpolate_scalar(s_query, learned_relative_angle_table)
    current_theta_s = interpolate_scalar(s_query, learned_theta_s_table)
    current_theta_ss = interpolate_scalar(s_query, learned_theta_ss_table)

    return (
        nominal_position,
        nominal_position_s,
        nominal_position_ss,
        current_normal,
        current_normal_s,
        current_normal_ss,
        current_theta,
        current_theta_s,
        current_theta_ss,
    )


# ============================================================
# 4. Hidden environment only
# ============================================================

hidden_curve_amplitude = 0.004
hidden_curve_length = 0.030


def hidden_surface_height(s):
    if s <= 0.0 or s >= hidden_curve_length:
        return 0.0, 0.0

    phase = 2.0 * np.pi * s / hidden_curve_length

    height = (
        0.5
        * hidden_curve_amplitude
        * (1.0 - np.cos(phase))
    )

    slope = (
        hidden_curve_amplitude
        * np.pi
        / hidden_curve_length
        * np.sin(phase)
    )

    return height, slope


def hidden_surface_frame(s):
    height, slope = hidden_surface_height(s)

    tangent = normalize(
        initial_tangent_world
        +
        slope
        *
        initial_normal_world
    )

    normal = normalize(
        initial_normal_world
        -
        slope
        *
        initial_tangent_world
    )

    return height, normal, tangent


# ============================================================
# 5. Load base models
# ============================================================

base_mj_model, pin_model = load_models()
base_data = mujoco.MjData(base_mj_model)
pin_data = pin_model.createData()

base_home_key_id = mujoco.mj_name2id(
    base_mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)

if base_home_key_id < 0:
    raise RuntimeError("原始模型找不到 home keyframe")

mujoco.mj_resetDataKeyframe(
    base_mj_model,
    base_data,
    base_home_key_id,
)

mujoco.mj_forward(base_mj_model, base_data)

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

if base_ft300_site_id < 0 or base_robotiq_body_id < 0:
    raise RuntimeError("找不到 FT300 或 Robotiq body")

sensor_position_world_initial = (
    base_data.site_xpos[base_ft300_site_id].copy()
)

body_position_world_initial = (
    base_data.xpos[base_robotiq_body_id].copy()
)

R_body_world_initial = (
    base_data.xmat[base_robotiq_body_id]
    .reshape(3, 3)
    .copy()
)

tool_rod_radius = 0.003

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

initial_gap = 0.035

hidden_contact_surface_distance = (
    initial_gap + probe_radius
)


def hidden_surface_point_world(s):
    height, _ = hidden_surface_height(s)

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
    _, normal, _ = hidden_surface_frame(s)

    return (
        hidden_surface_point_world(s)
        -
        probe_radius
        *
        normal
    )


# ============================================================
# 6. Build MuJoCo environment
# ============================================================

tree = ET.parse(SOURCE_XML)
root = tree.getroot()

worldbody_element = root.find("worldbody")

robotiq_body_element = root.find(
    ".//body[@name='robotiq_85_base_link']"
)

if worldbody_element is None:
    raise RuntimeError("MJCF 找不到 worldbody")

if robotiq_body_element is None:
    raise RuntimeError("MJCF 找不到 robotiq_85_base_link")

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name": "force_replay_tool_rod",
        "type": "cylinder",
        "size": f"{tool_rod_radius:.9f}",
        "fromto": (
            f"{sensor_position_local_body[0]:.9f} "
            f"{sensor_position_local_body[1]:.9f} "
            f"{sensor_position_local_body[2]:.9f} "
            f"{probe_position_local_body[0]:.9f} "
            f"{probe_position_local_body[1]:.9f} "
            f"{probe_position_local_body[2]:.9f}"
        ),
        "mass": "0",
        "contype": "0",
        "conaffinity": "0",
        "rgba": "0.45 0.45 0.45 1",
    },
)

ET.SubElement(
    robotiq_body_element,
    "geom",
    {
        "name": "force_replay_probe",
        "type": "sphere",
        "size": f"{probe_radius:.9f}",
        "pos": (
            f"{probe_position_local_body[0]:.9f} "
            f"{probe_position_local_body[1]:.9f} "
            f"{probe_position_local_body[2]:.9f}"
        ),
        "mass": "0",
        "rgba": "0.2 0.8 0.2 1",
        "contype": "2",
        "conaffinity": "2",
        "friction": "0.05 0.005 0.0005",
    },
)

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
wall_half_segment_length = 0.52 * segment_spacing
wall_half_height = 0.080

for index, s_value in enumerate(surface_samples):
    _, normal, _ = hidden_surface_frame(s_value)

    surface_point = hidden_surface_point_world(s_value)

    box_center = (
        surface_point
        +
        wall_half_thickness
        *
        normal
    )

    yaw = np.arctan2(normal[1], normal[0])

    quaternion = np.array([
        np.cos(yaw / 2.0),
        0.0,
        0.0,
        np.sin(yaw / 2.0),
    ])

    ET.SubElement(
        worldbody_element,
        "geom",
        {
            "name": f"force_replay_curve_{index}",
            "type": "box",
            "size": (
                f"{wall_half_thickness:.9f} "
                f"{wall_half_segment_length:.9f} "
                f"{wall_half_height:.9f}"
            ),
            "pos": (
                f"{box_center[0]:.9f} "
                f"{box_center[1]:.9f} "
                f"{box_center[2]:.9f}"
            ),
            "quat": (
                f"{quaternion[0]:.9f} "
                f"{quaternion[1]:.9f} "
                f"{quaternion[2]:.9f} "
                f"{quaternion[3]:.9f}"
            ),
            "rgba": "0.70 0.70 0.70 1",
            "contype": "2",
            "conaffinity": "2",
            "friction": "0.05 0.005 0.0005",
        },
    )

tree.write(
    CONTACT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


# ============================================================
# 7. Load experiment model
# ============================================================

mj_model = mujoco.MjModel.from_xml_path(str(CONTACT_XML))
mj_data = mujoco.MjData(mj_model)

dt = float(mj_model.opt.timestep)

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
        raise RuntimeError(f"MuJoCo 找不到关节: {name}")

    arm_dof_indices.append(
        int(mj_model.jnt_dofadr[joint_id])
    )

    pin_joint_id = pin_model.getJointId(name)

    if pin_joint_id == 0:
        raise RuntimeError(f"Pinocchio 找不到关节: {name}")

    arm_pin_v_indices.append(
        int(pin_model.idx_vs[pin_joint_id])
    )


for name in ARM_ACTUATOR_NAMES:
    actuator_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if actuator_id < 0:
        raise RuntimeError(f"找不到电机: {name}")

    arm_actuator_ids.append(actuator_id)


gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)

task_frame_id = pin_model.getFrameId(
    "robotiq_ft_frame_id"
)

if task_frame_id >= pin_model.nframes:
    raise RuntimeError("Pinocchio 找不到 robotiq_ft_frame_id")

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

if force_sensor_id < 0 or torque_sensor_id < 0 or ft300_site_id < 0:
    raise RuntimeError("找不到 FT300")

force_sensor_adr = int(
    mj_model.sensor_adr[force_sensor_id]
)

torque_sensor_adr = int(
    mj_model.sensor_adr[torque_sensor_id]
)

sensor_body_id = int(
    mj_model.site_bodyid[ft300_site_id]
)


def is_descendant(body_id, root_body_id):
    current = body_id

    while current > 0:
        if current == root_body_id:
            return True

        current = int(
            mj_model.body_parentid[current]
        )

    return False


payload_body_ids = [
    body_id
    for body_id in range(mj_model.nbody)
    if is_descendant(body_id, sensor_body_id)
]

payload_mass = sum(
    mj_model.body_mass[body_id]
    for body_id in payload_body_ids
)

home_key_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home",
)

if home_key_id < 0:
    raise RuntimeError("找不到 home keyframe")

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
# 8. Initial Pinocchio TCP geometry
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

initial_pose = pin_data.oMf[task_frame_id]

p_task_initial = initial_pose.translation.copy()
R_initial = initial_pose.rotation.copy()

probe_offset_task_local = (
    R_initial.T
    @
    (
        probe_position_world_initial
        -
        p_task_initial
    )
)

tool_axis_task_local = normalize(
    R_initial.T
    @
    initial_normal_world
)


# ============================================================
# 9. FT300 helpers
# ============================================================

def read_ft300_raw():
    force = mj_data.sensordata[
        force_sensor_adr
        :
        force_sensor_adr + 3
    ].copy()

    torque = mj_data.sensordata[
        torque_sensor_adr
        :
        torque_sensor_adr + 3
    ].copy()

    return np.concatenate([force, torque])


def payload_gravity_wrench():
    weighted_com = np.zeros(3)

    for body_id in payload_body_ids:
        weighted_com += (
            mj_model.body_mass[body_id]
            *
            mj_data.xipos[body_id]
        )

    payload_com = (
        weighted_com
        /
        payload_mass
    )

    sensor_position = (
        mj_data.site_xpos[ft300_site_id].copy()
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
        payload_com - sensor_position,
        force_world,
    )

    R_sensor_world = (
        mj_data.site_xmat[ft300_site_id]
        .reshape(3, 3)
        .copy()
    )

    return np.concatenate([
        R_sensor_world.T @ force_world,
        R_sensor_world.T @ torque_world,
    ])


def sensor_wrench_to_external_world(wrench_sensor):
    R_sensor_world = (
        mj_data.site_xmat[ft300_site_id]
        .reshape(3, 3)
        .copy()
    )

    return np.concatenate([
        -R_sensor_world @ wrench_sensor[:3],
        -R_sensor_world @ wrench_sensor[3:],
    ])


# ============================================================
# 10. Bias + filter
# ============================================================

bias_sum = np.zeros(6)
bias_count = 0
bias_wrench = None

rng = np.random.default_rng(42)

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

filtered_wrench = np.zeros(6)
wrench_filter_initialized = False


def filter_wrench(wrench):
    global filtered_wrench
    global wrench_filter_initialized

    if not wrench_filter_initialized:
        filtered_wrench = wrench.copy()
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

    return filtered_wrench.copy()


# ============================================================
# 11. Timeline / force controller
# ============================================================

def smoothstep(t, start, duration):
    if t <= start:
        return 0.0, 0.0, 0.0

    if t >= start + duration:
        return 1.0, 0.0, 0.0

    r = (t - start) / duration

    s = 3.0 * r**2 - 2.0 * r**3
    sd = (6.0 * r - 6.0 * r**2) / duration
    sdd = (6.0 - 12.0 * r) / duration**2

    return s, sd, sdd


pre_contact_offset = 0.003

learned_start_position = learned_positions[0].copy()

learned_start_normal = normalize(
    learned_normals[0].copy()
)

pre_contact_position = (
    learned_start_position
    -
    pre_contact_offset
    *
    learned_start_normal
)

approach_start = 1.0
approach_duration = 2.0

force_ramp_start = 3.0
force_ramp_duration = 2.0

desired_normal_force = 5.0

replay_start = 6.0
replay_duration = 12.0
replay_end = replay_start + replay_duration


def desired_force_at_time(t):
    progress, _, _ = smoothstep(
        t,
        force_ramp_start,
        force_ramp_duration,
    )

    return (
        desired_normal_force
        *
        progress
    )


def replay_motion(t):
    progress, progress_dot, progress_ddot = smoothstep(
        t,
        replay_start,
        replay_duration,
    )

    return (
        learned_path_length * progress,
        learned_path_length * progress_dot,
        learned_path_length * progress_ddot,
    )


force_virtual_mass = 2.0
force_virtual_damping = 120.0

normal_velocity_limit = 0.004
normal_acceleration_limit = 0.20

normal_correction_min = -0.005
normal_correction_max = +0.006

normal_correction = -pre_contact_offset
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

    if desired_force >= 0.5 and abs(force_error) < 0.03:
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

    proposed = (
        normal_correction
        +
        normal_velocity
        *
        dt
    )

    if proposed > normal_correction_max:
        normal_correction = normal_correction_max

        if normal_velocity > 0.0:
            normal_velocity = 0.0

    elif proposed < normal_correction_min:
        normal_correction = normal_correction_min

        if normal_velocity < 0.0:
            normal_velocity = 0.0

    else:
        normal_correction = proposed


# ============================================================
# 12. Cartesian controller
# ============================================================

Kp_position = np.array([225.0, 225.0, 225.0])
Kd_position = np.array([30.0, 30.0, 30.0])

Kp_orientation = np.array([225.0, 225.0, 225.0])
Kd_orientation = np.array([30.0, 30.0, 30.0])

linear_acceleration_limit = 30.0
angular_acceleration_limit = 80.0

exact_sigma_min_limit = 0.05
exact_condition_limit = 40.0
dls_lambda = 0.03


def tcp_cartesian_controller(
    external_world_wrench,
    desired_probe_position,
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

    task_pose = pin_data.oMf[task_frame_id]

    p_task = task_pose.translation.copy()
    R_task = task_pose.rotation.copy()

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

    J_task = pin.getFrameJacobian(
        pin_model,
        pin_data,
        task_frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )

    dJ_task = pin.getFrameJacobianTimeVariation(
        pin_model,
        pin_data,
        task_frame_id,
        pin.LOCAL_WORLD_ALIGNED,
    )

    Jv_task = J_task[:3, :]
    Jw_task = J_task[3:, :]

    Jv_tcp = (
        Jv_task
        -
        skew(r_world)
        @
        Jw_task
    )

    J_tcp = np.vstack([
        Jv_tcp,
        Jw_task,
    ])

    task_twist = (
        J_task
        @
        v_pin
    )

    omega_world = task_twist[3:]

    r_dot_world = np.cross(
        omega_world,
        r_world,
    )

    dJv_tcp = (
        dJ_task[:3, :]
        -
        skew(r_dot_world)
        @
        Jw_task
        -
        skew(r_world)
        @
        dJ_task[3:, :]
    )

    dJ_tcp = np.vstack([
        dJv_tcp,
        dJ_task[3:, :],
    ])

    tcp_twist = (
        J_tcp
        @
        v_pin
    )

    current_linear_velocity = tcp_twist[:3]
    current_angular_velocity = tcp_twist[3:]

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

    orientation_error = pin.log3(
        R_error
    )

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
        singular_values[-1]
    )

    condition = float(
        singular_values[0]
        /
        singular_values[-1]
    )

    use_exact = (
        sigma_min > exact_sigma_min_limit
        and
        condition < exact_condition_limit
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
            np.eye(6)
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

    for i in range(6):
        qdd_pin[
            arm_pin_v_indices[i]
        ] = qdd_arm[i]

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )

    # FT wrench is referenced at the sensor/task origin.
    # Therefore external-wrench compensation uses J_task.
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
        ] -= tau_external_arm[i]

    tau_mj = pinocchio_to_mujoco_tau(
        tau_pin,
        mj_model,
        pin_model,
    )

    saturation_count = 0
    max_motor_usage = 0.0

    for i in range(6):
        actuator_id = arm_actuator_ids[i]
        dof_id = arm_dof_indices[i]

        desired_torque = float(
            tau_mj[dof_id]
        )

        passive_torque = float(
            mj_data.qfrc_passive[dof_id]
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

        if requested_torque < lower or requested_torque > upper:
            saturation_count += 1

        torque_limit = max(
            abs(lower),
            abs(upper),
        )

        if torque_limit > 1e-12:
            max_motor_usage = max(
                max_motor_usage,
                abs(requested_torque)
                /
                torque_limit,
            )

        mj_data.ctrl[actuator_id] = np.clip(
            requested_torque,
            lower,
            upper,
        )

    if gripper_actuator_id >= 0:
        mj_data.ctrl[gripper_actuator_id] = 0.0

    tcp_error = float(
        np.linalg.norm(
            position_error
        )
    )

    actual_tool_axis = normalize(
        R_task
        @
        tool_axis_task_local
    )

    return (
        current_probe_position,
        actual_tool_axis,
        tcp_error,
        condition,
        saturation_count,
        100.0 * max_motor_usage,
    )


# ============================================================
# 13. Ground-truth evaluation only
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

    _, normal_true, _ = hidden_surface_frame(
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

    return evaluation_normals[index].copy()


# ============================================================
# 14. Statistics
# ============================================================

replay_force_errors = []
replay_tcp_errors = []
replay_learned_normal_errors = []
replay_true_normal_errors = []

summary_printed = False


# ============================================================
# 15. Main
# ============================================================

step = 0

print(
    "\n"
    "========== 74B Force-Controlled Replay "
    "74B恒力学习路径回放 =========="
)

print(
    f"Learned path length 学习路径长度: "
    f"{learned_path_length * 1000.0:.2f} mm"
)

print(
    f"Desired force 目标法向力: "
    f"{desired_normal_force:.1f} N"
)

print(
    "Path / normal source 轨迹/法向来源: "
    "74A learned data 74A学习数据"
)

print(
    "Hidden surface in controller "
    "控制器真实曲面信息: NONE 无"
)


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data,
) as viewer:

    while viewer.is_running():

        current_time = mj_data.time

        # ----------------------------------------------------
        # FT300
        # ----------------------------------------------------

        raw_wrench = read_ft300_raw()
        gravity_wrench = payload_gravity_wrench()

        if 0.2 <= current_time < 1.0:
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

        if bias_wrench is None:
            external_world_wrench = np.zeros(6)
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

            filtered_sensor_wrench = filter_wrench(
                compensated_wrench
                +
                noise
            )

            external_world_wrench = (
                sensor_wrench_to_external_world(
                    filtered_sensor_wrench
                )
            )

        # ----------------------------------------------------
        # Learned path state
        # ----------------------------------------------------

        (
            path_s,
            path_s_dot,
            path_s_ddot,
        ) = replay_motion(
            current_time
        )

        (
            nominal_position,
            nominal_position_s,
            nominal_position_ss,
            current_learned_normal,
            current_learned_normal_s,
            current_learned_normal_ss,
            current_learned_theta,
            current_learned_theta_s,
            current_learned_theta_ss,
        ) = learned_geometry(
            path_s
        )

        # ----------------------------------------------------
        # Force projected onto learned normal
        # ----------------------------------------------------

        measured_normal_force = max(
            0.0,
            -float(
                np.dot(
                    external_world_wrench[:3],
                    current_learned_normal,
                )
            ),
        )

        desired_force = desired_force_at_time(
            current_time
        )

        # ----------------------------------------------------
        # Position reference
        # ----------------------------------------------------

        if current_time < 1.0:

            desired_probe_position = (
                probe_position_world_initial.copy()
            )

            desired_linear_velocity = np.zeros(3)
            desired_linear_acceleration = np.zeros(3)

            normal_correction = -pre_contact_offset
            normal_velocity = 0.0
            normal_acceleration = 0.0


        elif current_time < 3.0:

            (
                approach_progress,
                approach_progress_dot,
                approach_progress_ddot,
            ) = smoothstep(
                current_time,
                approach_start,
                approach_duration,
            )

            approach_delta = (
                pre_contact_position
                -
                probe_position_world_initial
            )

            desired_probe_position = (
                probe_position_world_initial
                +
                approach_progress
                *
                approach_delta
            )

            desired_linear_velocity = (
                approach_progress_dot
                *
                approach_delta
            )

            desired_linear_acceleration = (
                approach_progress_ddot
                *
                approach_delta
            )

            normal_correction = -pre_contact_offset
            normal_velocity = 0.0
            normal_acceleration = 0.0


        else:

            update_force_controller(
                desired_force,
                measured_normal_force,
            )

            desired_probe_position = (
                nominal_position
                +
                normal_correction
                *
                current_learned_normal
            )

            desired_linear_velocity = (
                nominal_position_s
                *
                path_s_dot
                +
                normal_velocity
                *
                current_learned_normal
                +
                normal_correction
                *
                current_learned_normal_s
                *
                path_s_dot
            )

            desired_linear_acceleration = (
                nominal_position_ss
                *
                path_s_dot**2
                +
                nominal_position_s
                *
                path_s_ddot
                +
                normal_acceleration
                *
                current_learned_normal
                +
                2.0
                *
                normal_velocity
                *
                current_learned_normal_s
                *
                path_s_dot
                +
                normal_correction
                *
                (
                    current_learned_normal_ss
                    *
                    path_s_dot**2
                    +
                    current_learned_normal_s
                    *
                    path_s_ddot
                )
            )

        # ----------------------------------------------------
        # Orientation reference from learned normal
        # ----------------------------------------------------

        desired_rotation = (
            rotation_z(
                current_learned_theta
            )
            @
            R_initial
        )

        desired_angular_velocity = np.array([
            0.0,
            0.0,
            current_learned_theta_s
            *
            path_s_dot,
        ])

        desired_angular_acceleration = np.array([
            0.0,
            0.0,
            current_learned_theta_ss
            *
            path_s_dot**2
            +
            current_learned_theta_s
            *
            path_s_ddot,
        ])

        # ----------------------------------------------------
        # 6D TCP controller
        # ----------------------------------------------------

        (
            current_probe_position,
            actual_tool_axis,
            tcp_tracking_error,
            jacobian_condition,
            saturation_count,
            motor_usage,
        ) = tcp_cartesian_controller(
            external_world_wrench,
            desired_probe_position,
            desired_rotation,
            desired_linear_velocity,
            desired_linear_acceleration,
            desired_angular_velocity,
            desired_angular_acceleration,
        )

        # ----------------------------------------------------
        # Evaluation only
        # ----------------------------------------------------

        true_normal = evaluate_true_normal(
            current_probe_position
        )

        learned_alignment = float(
            np.clip(
                np.dot(
                    current_learned_normal,
                    true_normal,
                ),
                -1.0,
                1.0,
            )
        )

        learned_normal_error_deg = np.rad2deg(
            np.arccos(
                learned_alignment
            )
        )

        tool_learned_alignment = float(
            np.clip(
                np.dot(
                    actual_tool_axis,
                    current_learned_normal,
                ),
                -1.0,
                1.0,
            )
        )

        tool_learned_error_deg = np.rad2deg(
            np.arccos(
                tool_learned_alignment
            )
        )

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

        # ----------------------------------------------------
        # Statistics during replay
        # ----------------------------------------------------

        if replay_start <= current_time <= replay_end:

            replay_force_errors.append(
                abs(
                    desired_force
                    -
                    measured_normal_force
                )
            )

            replay_tcp_errors.append(
                tcp_tracking_error
            )

            replay_learned_normal_errors.append(
                tool_learned_error_deg
            )

            replay_true_normal_errors.append(
                tool_true_error_deg
            )

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        mujoco.mj_step(
            mj_model,
            mj_data,
        )

        viewer.sync()

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        if (
            not summary_printed
            and
            current_time >= replay_end + 0.05
            and
            len(replay_force_errors) > 0
        ):

            print(
                "\n"
                "========== 74B Replay Result "
                "74B回放结果 =========="
            )

            print(
                f"Force mean/max error "
                f"力平均/最大误差: "
                f"{np.mean(replay_force_errors):.3f} / "
                f"{np.max(replay_force_errors):.3f} N"
            )

            print(
                f"TCP mean/max error "
                f"TCP平均/最大跟踪误差: "
                f"{np.mean(replay_tcp_errors) * 1000.0:.3f} / "
                f"{np.max(replay_tcp_errors) * 1000.0:.3f} mm"
            )

            print(
                f"Tool learned-normal mean/max "
                f"工具对学习法向平均/最大误差: "
                f"{np.mean(replay_learned_normal_errors):.3f} / "
                f"{np.max(replay_learned_normal_errors):.3f} deg"
            )

            print(
                f"Tool true-normal mean/max "
                f"工具对真实法向平均/最大误差(仅评估): "
                f"{np.mean(replay_true_normal_errors):.3f} / "
                f"{np.max(replay_true_normal_errors):.3f} deg"
            )

            summary_printed = True

        # ----------------------------------------------------
        # Compact 1 s output
        # ----------------------------------------------------

        if step >= 1000 and step % 1000 == 0:

            path_progress_percent = (
                100.0
                *
                path_s
                /
                learned_path_length
            )

            print(
                "\n"
                "========== 74B Replay "
                "74B学习路径回放 =========="
            )

            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )

            print(
                f"Replay progress 路径进度: "
                f"{path_progress_percent:.1f}%"
            )

            print(
                f"Force 目标/实际法向力: "
                f"{desired_force:.2f} / "
                f"{measured_normal_force:.2f} N"
            )

            print(
                f"Force correction 法向力控修正: "
                f"{normal_correction * 1000.0:+.3f} mm"
            )

            print(
                f"Learned normal error "
                f"学习法向误差(仅评估): "
                f"{learned_normal_error_deg:.3f} deg"
            )

            print(
                f"Tool error learned/true "
                f"工具对学习/真实法向误差: "
                f"{tool_learned_error_deg:.3f} / "
                f"{tool_true_error_deg:.3f} deg"
            )

            print(
                f"TCP tracking error TCP跟踪误差: "
                f"{tcp_tracking_error * 1000.0:.3f} mm"
            )

            print(
                f"Jacobian / saturation "
                f"雅可比条件数/电机饱和: "
                f"{jacobian_condition:.2f} / "
                f"{saturation_count}"
            )

            print(
                f"Motor usage 最大电机使用率: "
                f"{motor_usage:.1f}%"
            )

        step += 1

        time.sleep(dt)
