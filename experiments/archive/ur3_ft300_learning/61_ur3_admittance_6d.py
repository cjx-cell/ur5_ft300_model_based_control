#!/usr/bin/env python3

"""
61_ur3_admittance_6d.py

目标
============================================================

完整六维导纳：

    Fx Fy Fz -> XYZ translation
    Mx My Mz -> XYZ rotation

控制链：

FT300 Raw
    ↓
Payload gravity compensation
    ↓
Bias compensation
    ↓
6D low-pass filter
    ↓
WORLD external wrench
    ↓
6D admittance
    ↓
Desired Cartesian pose
    ↓
Resolved-acceleration controller
    ↓
Pinocchio RNEA
    ↓
-J^T Wext disturbance compensation
    ↓
UR3 motors


实验
============================================================

2 ~ 5 s：

    在 FT300 原点等效施加：

        Fx = +2 N

    希望 FT300 测到：

        [2, 0, 0, 0, 0, 0]

    理论稳态位移：

        x = 2 / 50
          = 0.04 m
          = 40 mm


6 ~ 9 s：

    在 FT300 原点等效施加：

        Mz = +0.2 N*m

    理论稳态转角：

        theta = 0.2 / 2
              = 0.1 rad
              = 5.73 deg
"""

import time

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
# 1. Models
# ============================================================

mj_model, pin_model = load_models()

mj_data = mujoco.MjData(
    mj_model
)

pin_data = pin_model.createData()

dt = float(
    mj_model.opt.timestep
)


# ============================================================
# 2. UR3
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

    # MuJoCo joint

    mj_joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if mj_joint_id < 0:
        raise RuntimeError(
            f"找不到 MuJoCo joint: {name}"
        )

    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                mj_joint_id
            ]
        )
    )


    # Pinocchio joint

    pin_joint_id = (
        pin_model.getJointId(
            name
        )
    )

    if pin_joint_id == 0:
        raise RuntimeError(
            f"找不到 Pinocchio joint: {name}"
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
# 3. Task frame
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
        f"找不到 frame: {TASK_FRAME_NAME}"
    )


# ============================================================
# 4. FT300
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
        "找不到 FT300 sensor/site"
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
# 5. Downstream payload
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
# 6. External load body
# ============================================================
#
# xfrc_applied 的 Cartesian wrench
# 施加在这个 body 上。
#
# force 等效作用于 body COM。
#
# 因此下面会计算额外 torque，
# 把它转换成关于 FT300 原点的目标 wrench。
# ============================================================

push_body_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robotiq_85_base_link",
)


if push_body_id < 0:

    raise RuntimeError(
        "找不到 robotiq_85_base_link"
    )


# ============================================================
# 7. Reset
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
# 8. Initial task pose
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
# 9. Read FT300
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
# 10. Payload gravity wrench
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
# 11. Bias calibration
# ============================================================

bias_start_time = 0.2

bias_end_time = 1.0


bias_sum = np.zeros(
    6
)

bias_count = 0

bias_wrench = None


# ============================================================
# 12. Measurement noise
# ============================================================

rng = np.random.default_rng(
    42
)


force_noise_std = 0.05

torque_noise_std = 0.003


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
# 13. Six-dimensional low-pass filter
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
# 14. Sensor frame -> WORLD external wrench
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
# 15. Desired external wrench
# ============================================================

def simulated_external_wrench(
    t,
):

    force = np.zeros(
        3
    )

    torque_at_sensor = np.zeros(
        3
    )


    # 2 ~ 5 s:
    #
    # pure +X force

    if (
        2.0
        <= t
        <
        5.0
    ):

        force[0] = 2.0


    # 6 ~ 9 s:
    #
    # pure +Z torque

    if (
        6.0
        <= t
        <
        9.0
    ):

        torque_at_sensor[2] = 0.2


    return (
        force,
        torque_at_sensor,
    )


# ============================================================
# 16. Apply a desired wrench ABOUT the FT300 origin
# ============================================================
#
# MuJoCo xfrc_applied：
#
# force 作用于 body COM，
# torque 也是关于 body COM 的 torque。
#
#
# 我们真正希望规定的是：
#
# FT300原点处的：
#
#       F
#       tau_sensor
#
#
# 设：
#
# r = p_COM - p_sensor
#
#
# 关于 FT300 原点：
#
# tau_sensor
#
# =
#
# tau_COM
#
# +
#
# r × F
#
#
# 所以：
#
# tau_COM
#
# =
#
# tau_sensor
#
# -
#
# r × F
#
#
# 这样就可以使用 xfrc_applied，
# 同时又实现“纯 Fx”或“纯 Mz”。
# ============================================================

def apply_wrench_at_ft300(
    force_world,
    torque_sensor_world,
):

    # 清除上一周期外力

    mj_data.xfrc_applied[:] = 0.0


    # FT300 measurement origin

    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    # Robotiq body COM
    #
    # xipos 是 body inertial frame COM
    # 在 WORLD 下的位置。

    body_com_world = (
        mj_data.xipos[
            push_body_id
        ]
        .copy()
    )


    # FT300 -> body COM

    r_world = (
        body_com_world
        -
        sensor_position_world
    )


    # 把希望作用在 sensor origin 的 wrench
    # 转换成等效的 COM wrench

    torque_at_com_world = (
        torque_sensor_world
        -
        np.cross(
            r_world,
            force_world,
        )
    )


    # xfrc_applied:
    #
    # first 3  -> force
    # last 3   -> torque

    mj_data.xfrc_applied[
        push_body_id,
        :3
    ] = force_world


    mj_data.xfrc_applied[
        push_body_id,
        3:
    ] = torque_at_com_world


# ============================================================
# 17. 6D admittance parameters
# ============================================================

M_translation = np.array([
    2.0,
    2.0,
    2.0,
])


D_translation = np.array([
    20.0,
    20.0,
    20.0,
])


K_translation = np.array([
    50.0,
    50.0,
    50.0,
])


I_rotation = np.array([
    0.2,
    0.2,
    0.2,
])


D_rotation = np.array([
    1.3,
    1.3,
    1.3,
])


K_rotation = np.array([
    2.0,
    2.0,
    2.0,
])


# ============================================================
# 18. Admittance states
# ============================================================

position_offset = np.zeros(
    3
)


linear_velocity_adm = np.zeros(
    3
)


linear_acceleration_adm = np.zeros(
    3
)


rotation_offset = np.zeros(
    3
)


angular_velocity_adm = np.zeros(
    3
)


angular_acceleration_adm = np.zeros(
    3
)


# ============================================================
# 19. Safety limits
# ============================================================

position_limit = 0.060

linear_velocity_limit = 0.150

linear_acceleration_limit = 2.0


rotation_limit = np.deg2rad(
    10.0
)


angular_velocity_limit = np.deg2rad(
    30.0
)


angular_acceleration_limit = np.deg2rad(
    200.0
)


# ============================================================
# 20. Admittance update
# ============================================================

def update_admittance(
    external_world_wrench,
):

    global position_offset
    global linear_velocity_adm
    global linear_acceleration_adm

    global rotation_offset
    global angular_velocity_adm
    global angular_acceleration_adm


    force = (
        external_world_wrench[:3]
    )


    torque = (
        external_world_wrench[3:]
    )


    # --------------------------------------------------------
    # Translation
    #
    # M a + D v + K x = F
    # --------------------------------------------------------

    linear_acceleration_adm = (
        (
            force
            -
            D_translation
            *
            linear_velocity_adm
            -
            K_translation
            *
            position_offset
        )
        /
        M_translation
    )


    linear_acceleration_adm = np.clip(
        linear_acceleration_adm,
        -linear_acceleration_limit,
        linear_acceleration_limit,
    )


    linear_velocity_adm += (
        linear_acceleration_adm
        *
        dt
    )


    linear_velocity_adm = np.clip(
        linear_velocity_adm,
        -linear_velocity_limit,
        linear_velocity_limit,
    )


    position_offset += (
        linear_velocity_adm
        *
        dt
    )


    position_offset = np.clip(
        position_offset,
        -position_limit,
        position_limit,
    )


    # --------------------------------------------------------
    # Rotation
    #
    # I alpha + D omega + K theta = M
    # --------------------------------------------------------

    angular_acceleration_adm = (
        (
            torque
            -
            D_rotation
            *
            angular_velocity_adm
            -
            K_rotation
            *
            rotation_offset
        )
        /
        I_rotation
    )


    angular_acceleration_adm = np.clip(
        angular_acceleration_adm,
        -angular_acceleration_limit,
        angular_acceleration_limit,
    )


    angular_velocity_adm += (
        angular_acceleration_adm
        *
        dt
    )


    angular_velocity_adm = np.clip(
        angular_velocity_adm,
        -angular_velocity_limit,
        angular_velocity_limit,
    )


    rotation_offset += (
        angular_velocity_adm
        *
        dt
    )


    rotation_offset = np.clip(
        rotation_offset,
        -rotation_limit,
        rotation_limit,
    )


# ============================================================
# 21. Inner Cartesian gains
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


# 提高姿态内环带宽

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


dls_lambda = 0.03


# ============================================================
# 22. Cartesian inner controller
# ============================================================

def cartesian_tracking_controller(
    external_world_wrench,
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


    p_current = (
        current_pose.translation.copy()
    )


    R_current = (
        current_pose.rotation.copy()
    )


    # --------------------------------------------------------
    # Desired pose
    # --------------------------------------------------------

    p_desired = (
        p_initial
        +
        position_offset
    )


    R_offset = pin.exp3(
        rotation_offset
    )


    R_desired = (
        R_offset
        @
        R_initial
    )


    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    e_position = (
        p_desired
        -
        p_current
    )


    R_error = (
        R_desired
        @
        R_current.T
    )


    e_orientation = (
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


    linear_velocity = (
        twist[:3]
    )


    angular_velocity = (
        twist[3:]
    )


    # --------------------------------------------------------
    # Task acceleration command
    # --------------------------------------------------------

    linear_acceleration_command = (
        linear_acceleration_adm

        +

        Kp_position
        *
        e_position

        +

        Kd_position
        *
        (
            linear_velocity_adm
            -
            linear_velocity
        )
    )


    angular_acceleration_command = (
        angular_acceleration_adm

        +

        Kp_orientation
        *
        e_orientation

        +

        Kd_orientation
        *
        (
            angular_velocity_adm
            -
            angular_velocity
        )
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
    # DLS inverse
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
    # Full Pinocchio qdd
    # --------------------------------------------------------

    qdd_pin = np.zeros(
        pin_model.nv
    )


    for i in range(6):

        qdd_pin[
            arm_pin_v_indices[i]
        ] = qdd_arm[i]


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
    # tau_external = J^T W_external
    #
    # tau_motor =
    #
    # tau_inverse_dynamics
    #
    # -
    #
    # tau_external
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
        ] = 0.0


    # --------------------------------------------------------
    # Tracking errors
    # --------------------------------------------------------

    position_error_norm = (
        np.linalg.norm(
            e_position
        )
    )


    orientation_error_deg = np.rad2deg(
        np.linalg.norm(
            e_orientation
        )
    )


    return (
        position_error_norm,
        orientation_error_deg,
    )


# ============================================================
# 23. Simulation
# ============================================================

step = 0


print(
    "\n"
    "========== 6D Admittance "
    "6D导纳控制 =========="
)


print(
    "Translation test 平移测试: "
    "+2 N X -> 40.0 mm"
)


print(
    "Rotation test 旋转测试: "
    "+0.2 N·m Z -> 5.73 deg"
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
        # A. Desired external wrench
        # ====================================================

        (
            applied_force_world,
            applied_torque_world,
        ) = simulated_external_wrench(
            current_time
        )


        # ====================================================
        # B. Apply physical wrench
        # ====================================================
        #
        # 这里重新使用 xfrc_applied。
        #
        # 与最初版本不同的是：
        #
        # 我们加入了 torque shifting，
        # 确保关于 FT300 原点得到想要的 wrench。
        # ====================================================

        apply_wrench_at_ft300(
            applied_force_world,
            applied_torque_world,
        )


        # ====================================================
        # C. Raw FT300
        # ====================================================

        raw_wrench = (
            read_ft300_raw()
        )


        gravity_wrench = (
            payload_gravity_wrench()
        )


        # ====================================================
        # D. Bias calibration
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


            bias_count += 1


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
        # E. Signal processing
        # ====================================================

        if bias_wrench is None:

            compensated_sensor_wrench = (
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
        # F. 6D admittance
        # ====================================================

        if bias_wrench is not None:

            update_admittance(
                external_world_wrench
            )


        # ====================================================
        # G. Inner Cartesian controller
        # ====================================================

        (
            position_tracking_error,
            orientation_tracking_error,
        ) = cartesian_tracking_controller(
            external_world_wrench
        )


        # ====================================================
        # H. Physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # I. Concise output
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            measured_torque_norm = (
                np.linalg.norm(
                    external_world_wrench[
                        3:
                    ]
                )
            )


            print(
                "\n"
                "========== 6D Admittance "
                "6D导纳实验 =========="
            )


            print(
                f"Time 时间: "
                f"{mj_data.time:.2f} s"
            )


            print(
                f"Applied Fx 施加X力: "
                f"{applied_force_world[0]:+.3f} N"
            )


            print(
                f"Measured Fx FT300外力X: "
                f"{external_world_wrench[0]:+.3f} N"
            )


            print(
                f"X displacement X导纳位移: "
                f"{position_offset[0] * 1000.0:+.2f} mm"
            )


            print(
                f"Applied Mz 施加Z力矩: "
                f"{applied_torque_world[2]:+.3f} N·m"
            )


            print(
                f"Measured Mz FT300外力矩Z: "
                f"{external_world_wrench[5]:+.3f} N·m"
            )


            print(
                f"Measured torque magnitude "
                f"FT300外力矩大小: "
                f"{measured_torque_norm:.4f} N·m"
            )


            print(
                f"Z rotation Z导纳转角: "
                f"{np.rad2deg(rotation_offset[2]):+.2f} deg"
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