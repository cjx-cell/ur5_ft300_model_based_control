#!/usr/bin/env python3

"""
60_ur3_admittance_1d.py

实验目标
============================================================

第一次使用 FT300 真实参与机器人闭环控制：

External force
    ↓
FT300
    ↓
Gravity/Bias compensation
    ↓
Low-pass filter
    ↓
1D Admittance
    ↓
Desired X position
    ↓
Cartesian tracking controller
    ↓
UR3 motion


实验设置
------------------------------------------------------------

0 ~ 2 s
    无外力
    机器人保持初始位置

2 ~ 6 s
    对 Robotiq 施加 +2 N WORLD X 外力

6 s以后
    撤去外力


导纳参数：

    Md = 2 kg
    Dd = 20 N*s/m
    Kd = 50 N/m


因此持续 2 N 静态外力时：

    x = F/K = 2/50 = 0.04 m

理论稳态位移：

    40 mm


撤去外力后：

    x -> 0
"""

import time

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 导入已经验证好的模型与映射工具
# ============================================================

from ur3_ft300_robotiq_pinocchio import (
    load_models,
    mujoco_to_pinocchio_q,
    mujoco_to_pinocchio_v,
    pinocchio_to_mujoco_tau,
)


# ============================================================
# 2. 加载模型
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
# 3. UR3 joint names
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


# ============================================================
# 4. 获取 MuJoCo arm joint / DOF indices
# ============================================================

arm_dof_indices = []

arm_actuator_ids = []


for name in ARM_JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"找不到关节: {name}"
        )

    arm_dof_indices.append(
        int(
            mj_model.jnt_dofadr[
                joint_id
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


# ============================================================
# 5. Pinocchio arm velocity indices
# ============================================================
#
# Pinocchio 当前 nv 不再只是6。
#
# 因为模型还包含 Robotiq joints。
#
# 所以必须按名字找到：
#
# UR3六个关节在 Pinocchio nv 中的位置。
# ============================================================

arm_pin_v_indices = []


for name in ARM_JOINT_NAMES:

    pin_joint_id = (
        pin_model.getJointId(
            name
        )
    )

    if pin_joint_id == 0:
        raise RuntimeError(
            f"Pinocchio 找不到关节: {name}"
        )

    arm_pin_v_indices.append(
        int(
            pin_model.idx_vs[
                pin_joint_id
            ]
        )
    )


# ============================================================
# 6. Gripper actuator
# ============================================================

gripper_actuator_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_position",
)


# ============================================================
# 7. Task frame
# ============================================================
#
# 本次用：
#
# robotiq_ft_frame_id
#
# 作为机器人末端控制点。
# ============================================================

TASK_FRAME_NAME = (
    "robotiq_ft_frame_id"
)


task_frame_id = pin_model.getFrameId(
    TASK_FRAME_NAME
)


if task_frame_id >= pin_model.nframes:

    raise RuntimeError(
        f"Pinocchio 找不到 frame: "
        f"{TASK_FRAME_NAME}"
    )


# ============================================================
# 8. FT300 sensors
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


# ============================================================
# 9. FT300 measurement body
# ============================================================

sensor_body_id = int(
    mj_model.site_bodyid[
        ft300_site_id
    ]
)


# ============================================================
# 10. 找到 FT300 downstream bodies
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
# 11. 模拟“人推夹爪”的 body
# ============================================================
#
# 将外力直接作用在：
#
# robotiq_85_base_link
#
# 上。
#
# 它位于 FT300 下游，
# 因此这个力会通过 FT300。
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
# 12. Reset 到 home
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
# 13. 获取初始 task pose
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
# 14. FT300 raw wrench
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
# 15. Payload gravity wrench
# ============================================================

def payload_gravity_wrench():

    # --------------------------------------------------------
    # Payload total COM in WORLD
    # --------------------------------------------------------

    weighted_com = np.zeros(
        3
    )


    for body_id in payload_body_ids:

        mass = (
            mj_model.body_mass[
                body_id
            ]
        )


        weighted_com += (
            mass
            *
            mj_data.xipos[
                body_id
            ]
        )


    payload_com_world = (
        weighted_com
        /
        payload_mass
    )


    # --------------------------------------------------------
    # Sensor origin
    # --------------------------------------------------------

    sensor_position_world = (
        mj_data.site_xpos[
            ft300_site_id
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Gravity support force
    # --------------------------------------------------------

    gravity_world = np.array(
        mj_model.opt.gravity
    )


    force_world = (
        -payload_mass
        *
        gravity_world
    )


    # --------------------------------------------------------
    # Gravity torque
    # --------------------------------------------------------

    r_world = (
        payload_com_world
        -
        sensor_position_world
    )


    torque_world = np.cross(
        r_world,
        force_world,
    )


    # --------------------------------------------------------
    # WORLD -> SENSOR
    # --------------------------------------------------------

    R_sensor_world = (
        mj_data.site_xmat[
            ft300_site_id
        ]
        .reshape(3, 3)
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
# 16. Bias calibration
# ============================================================

bias_start_time = 0.2

bias_end_time = 1.0


bias_sum = np.zeros(
    6
)

bias_count = 0

bias_wrench = None


# ============================================================
# 17. 模拟小量测量噪声
# ============================================================
#
# 54已经验证过滤波器。
#
# 这里保留少量噪声，
# 让整个控制链更接近实际传感器。
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
# 18. 6D Low-pass filter
# ============================================================
#
# 注意：
#
# 这里是完整6维：
#
# Fx Fy Fz Mx My Mz
#
# 全部滤波。
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


alpha = (

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
    input_wrench,
):

    global filtered_wrench
    global filter_initialized


    if not filter_initialized:

        filtered_wrench = (
            input_wrench.copy()
        )

        filter_initialized = True


    else:

        filtered_wrench = (

            alpha
            *
            filtered_wrench

            +

            (1.0 - alpha)
            *
            input_wrench

        )


    return filtered_wrench.copy()


# ============================================================
# 19. SENSOR wrench -> WORLD external force
# ============================================================
#
# FT300 force sensor 当前测得的是：
#
# parent 对 downstream payload 的作用力。
#
# 我们真正想要用于导纳的是：
#
# environment 对 robot 的外力。
#
# 两者方向相反。
#
#
# 因此：
#
# F_external_world
#
# =
#
# - R_world_sensor
#
#   *
#
#   F_sensor_compensated
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
# 20. 外力实验
# ============================================================
#
# 2 ~ 6 s：
#
# +2 N WORLD X
#
#
# 其他时间：
#
# 0 N
# ============================================================

def simulated_push_force(t):

    if (
        2.0
        <= t
        <
        6.0
    ):

        return np.array([
            2.0,
            0.0,
            0.0,
        ])


    return np.zeros(
        3
    )


# ============================================================
# 21. 1D Admittance parameters
# ============================================================
#
# Md xdd
# +
# Dd xd
# +
# Kd x
# =
# Fext
#
#
# 这里 x 是：
#
# 相对于初始位置的 X displacement。
# ============================================================

M_adm = 2.0       # kg

D_adm = 20.0      # N*s/m

K_adm = 50.0      # N/m


# ============================================================
# 22. Admittance states
# ============================================================

x_adm = 0.0

v_adm = 0.0

a_adm = 0.0


# 安全限制

x_adm_limit = 0.060    # 60 mm

v_adm_limit = 0.150    # 150 mm/s

a_adm_limit = 2.0      # m/s^2


# ============================================================
# 23. Admittance update
# ============================================================
#
# 方程：
#
# Md * xdd
# +
# Dd * xd
# +
# Kd * x
# =
# Fext
#
#
# 解出：
#
# xdd =
#
# (Fext - Dd*xd - Kd*x)
#
# / Md
# ============================================================

def update_admittance(
    force_x,
):

    global x_adm
    global v_adm
    global a_adm


    a_adm = (

        force_x

        -

        D_adm
        *
        v_adm

        -

        K_adm
        *
        x_adm

    ) / M_adm


    a_adm = np.clip(
        a_adm,
        -a_adm_limit,
        a_adm_limit,
    )


    # --------------------------------------------------------
    # Semi-implicit Euler
    # --------------------------------------------------------
    #
    # 先更新速度：
    #
    # v[k+1] = v[k] + a*dt
    #
    # 再更新位置：
    #
    # x[k+1] = x[k] + v[k+1]*dt
    #
    # 比普通显式Euler稍微稳定一些。
    # --------------------------------------------------------

    v_adm += (
        a_adm
        *
        dt
    )


    v_adm = np.clip(
        v_adm,
        -v_adm_limit,
        v_adm_limit,
    )


    x_adm += (
        v_adm
        *
        dt
    )


    x_adm = np.clip(
        x_adm,
        -x_adm_limit,
        x_adm_limit,
    )


# ============================================================
# 24. Inner Cartesian tracking gains
# ============================================================
#
# 外环：
#
# Force -> desired position
#
#
# 内环：
#
# desired Cartesian pose
#       ->
# joint acceleration
#       ->
# inverse dynamics torque
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


Kp_orientation = np.array([
    25.0,
    25.0,
    25.0,
])


Kd_orientation = np.array([
    10.0,
    10.0,
    10.0,
])


dls_lambda = 0.03


# ============================================================
# 25. Cartesian inner controller
# ============================================================

def cartesian_tracking_controller(
    external_world_wrench,
):

    # --------------------------------------------------------
    # Current Pinocchio state
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
    # FK + Jacobian + Jdot
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
    # Desired pose from admittance
    # --------------------------------------------------------

    p_desired = (
        p_initial.copy()
    )


    # 只改变 WORLD X

    p_desired[0] += (
        x_adm
    )


    R_desired = (
        R_initial
    )


    # --------------------------------------------------------
    # Desired Cartesian velocity
    # --------------------------------------------------------

    v_desired = np.array([
        v_adm,
        0.0,
        0.0,
    ])


    omega_desired = np.zeros(
        3
    )


    # --------------------------------------------------------
    # Desired Cartesian acceleration
    # --------------------------------------------------------

    a_desired = np.array([
        a_adm,
        0.0,
        0.0,
    ])


    alpha_desired = np.zeros(
        3
    )


    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    e_position = (
        p_desired
        -
        p_current
    )


    # --------------------------------------------------------
    # Orientation error
    # --------------------------------------------------------

    R_error = (

        R_desired

        @

        R_current.T

    )


    e_orientation = pin.log3(
        R_error
    )


    # --------------------------------------------------------
    # Full Jacobian
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


    # --------------------------------------------------------
    # 只取 UR3 六个关节的列
    # --------------------------------------------------------
    #
    # Robotiq joints 不用于控制 task frame。
    # --------------------------------------------------------

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
    # Resolved acceleration command
    # --------------------------------------------------------

    linear_acceleration_command = (

        a_desired

        +

        Kp_position
        *
        e_position

        +

        Kd_position
        *
        (
            v_desired
            -
            linear_velocity
        )

    )


    angular_acceleration_command = (

        alpha_desired

        +

        Kp_orientation
        *
        e_orientation

        +

        Kd_orientation
        *
        (
            omega_desired
            -
            angular_velocity
        )

    )


    task_acceleration_command = np.concatenate([
        linear_acceleration_command,
        angular_acceleration_command,
    ])


    # --------------------------------------------------------
    # Jdot * qdot
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
    # Damped Least Squares
    # --------------------------------------------------------
    #
    # qdd =
    #
    # J^T
    #
    # (J J^T + lambda^2 I)^-1
    #
    # rhs
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
    # Pinocchio inverse dynamics
    # --------------------------------------------------------
    #
    # tau =
    #
    # M(q) qdd
    #
    # +
    #
    # h(q, qdot)
    #
    #
    # 完整模型已经包含：
    #
    # UR3
    # FT300
    # Robotiq
    # --------------------------------------------------------

    tau_pin = pin.rnea(
        pin_model,
        pin_data,
        q_pin,
        v_pin,
        qdd_pin,
    )


        # ========================================================
    # External wrench disturbance compensation
    # 外部六维力扰动补偿
    # ========================================================
    #
    # 真实机器人动力学：
    #
    # M qdd + h
    #
    # =
    #
    # tau_motor
    #
    # +
    #
    # J^T W_external
    #
    #
    # 所以为了让机器人真正执行：
    #
    # qdd_command
    #
    # 电机应该输出：
    #
    # tau_motor
    #
    # =
    #
    # M qdd_command + h
    #
    # -
    #
    # J^T W_external
    #
    #
    # 注意：
    #
    # external_world_wrench 已经经过：
    #
    # FT300 raw
    # -> payload gravity compensation
    # -> bias compensation
    # -> low-pass filtering
    # -> sensor frame to world frame
    #
    # 所以这里减掉的不是夹爪自身重量，
    # 而是真正估计出来的外部 wrench。
    # ========================================================

    tau_external_arm = (
        J_arm.T
        @
        external_world_wrench
    )


    # 只修改 UR3 六个关节。
    #
    # Robotiq 自己的关节不由这里控制。

    for i in range(6):

        tau_pin[
            arm_pin_v_indices[i]
        ] -= (
            tau_external_arm[i]
        )
    

    tau_mj = (
        pinocchio_to_mujoco_tau(
            tau_pin,
            mj_model,
            pin_model,
        )
    )


    # --------------------------------------------------------
    # 只写 UR3 六个 motor
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


    # --------------------------------------------------------
    # Gripper保持当前位置
    # --------------------------------------------------------

    if gripper_actuator_id >= 0:

        mj_data.ctrl[
            gripper_actuator_id
        ] = 0.0


    position_tracking_error = (
        np.linalg.norm(
            e_position
        )
    )


    return (
        p_current,
        position_tracking_error,
    )


# ============================================================
# 26. Simulation
# ============================================================

step = 0


print(
    "\n========== 1D Admittance 1D导纳控制 =========="
)


print(
    f"Virtual mass 虚拟质量: "
    f"{M_adm:.1f} kg"
)


print(
    f"Virtual damping 虚拟阻尼: "
    f"{D_adm:.1f} N·s/m"
)


print(
    f"Virtual stiffness 虚拟刚度: "
    f"{K_adm:.1f} N/m"
)


print(
    "Expected displacement 理论稳态位移: "
    "40.0 mm at 2 N"
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
        # A. 模拟外部推力
        # ====================================================

        applied_force_world = (
            simulated_push_force(
                current_time
            )
        )


        # 清空上一周期 external forces

        mj_data.xfrc_applied[:] = 0.0


        # 对 Robotiq body 施加 WORLD force

        mj_data.xfrc_applied[
            push_body_id,
            :3
        ] = applied_force_world


        # ====================================================
        # B. 读取 FT300
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
        # D. Wrench processing
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


        # 模拟测量噪声

        noisy_sensor_wrench = (
            add_measurement_noise(
                compensated_sensor_wrench
            )
        )


        # 六维低通滤波

        filtered_sensor_wrench = (
            low_pass_filter(
                noisy_sensor_wrench
            )
        )


        # SENSOR frame
        #
        # ->
        #
        # WORLD external wrench

        external_world_wrench = (
            sensor_wrench_to_external_world(
                filtered_sensor_wrench
            )
        )


        external_force_x = (
            external_world_wrench[0]
        )


        # ====================================================
        # E. Outer admittance
        # ====================================================

        if bias_wrench is not None:

            update_admittance(
                external_force_x
            )


        # ====================================================
        # F. Inner Cartesian controller
        # ====================================================

        (
        current_position,
        tracking_error,
    ) = (
        cartesian_tracking_controller(
            external_world_wrench
        )
    )


        # ====================================================
        # G. MuJoCo physics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data,
        )


        viewer.sync()


        # ====================================================
        # H. 精简打印
        # ====================================================

        if (
            step >= 1000
            and
            step % 1000 == 0
        ):

            print(
                "\n"
                "========== Admittance "
                "导纳实验 =========="
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
                f"{external_force_x:+.3f} N"
            )


            print(
                f"Admittance displacement "
                f"导纳位移: "
                f"{x_adm * 1000.0:+.2f} mm"
            )


            print(
                f"Tracking error "
                f"末端跟踪误差: "
                f"{tracking_error * 1000.0:.3f} mm"
            )


        step += 1


        time.sleep(
            dt
        )