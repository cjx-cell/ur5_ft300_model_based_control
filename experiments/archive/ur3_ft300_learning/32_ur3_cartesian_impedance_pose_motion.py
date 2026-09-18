import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 加载 MuJoCo 模型
# ============================================================

xml_path = Path(__file__).parent / "ur3_converted.xml"

mj_model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

mj_data = mujoco.MjData(
    mj_model
)


# ============================================================
# 2. 加载 Pinocchio 模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

pin_model = pin.buildModelFromUrdf(
    urdf_path
)

pin_data = pin_model.createData()


# ============================================================
# 3. 获取 tool0 frame
# ============================================================

ee_name = "tool0"

ee_id = pin_model.getFrameId(
    ee_name
)

print(
    "Pinocchio tool0 frame id =",
    ee_id
)


# ============================================================
# 4. MuJoCo q -> Pinocchio q
# ============================================================
#
# MuJoCo：
#
# q = [q1 q2 q3 q4 q5 q6]
#
# Pinocchio：
#
# wrist3 是 JointModelRUBZ，
# 所以 configuration 为：
#
# [
#   q1,
#   q2,
#   q3,
#   q4,
#   q5,
#   cos(q6),
#   sin(q6)
# ]
#
# Pinocchio:
#
# nq = 7
# nv = 6
# ============================================================

def mujoco_to_pinocchio(q_mj):

    q_pin = pin.neutral(
        pin_model
    )

    # 前五轴直接复制

    q_pin[:5] = q_mj[:5]

    # wrist3

    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 5. 设置初始关节姿态
# ============================================================

mj_data.qpos[:] = np.deg2rad([
    30.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0
])

mj_data.qvel[:] = 0.0


mujoco.mj_forward(
    mj_model,
    mj_data
)


# ============================================================
# 6. 获取初始 TCP 位姿
# ============================================================

q_pin_initial = mujoco_to_pinocchio(
    mj_data.qpos
)


pin.forwardKinematics(
    pin_model,
    pin_data,
    q_pin_initial
)

pin.updateFramePlacements(
    pin_model,
    pin_data
)


T_initial = pin_data.oMf[ee_id]


p_initial = (
    T_initial.translation.copy()
)

R_initial = (
    T_initial.rotation.copy()
)


print("\n========== Initial Pose ==========")

print("position:")
print(p_initial)

print("\nrotation:")
print(R_initial)


# ============================================================
# 7. 最终目标
# ============================================================
#
# 和前面的 Cartesian Computed Torque 实验保持一致：
#
# 位置：
#
#     WORLD Z + 3 cm
#
# 姿态：
#
#     绕 WORLD Z + 10°
# ============================================================

delta_position = np.array([
    0.0,
    0.0,
    0.03
])


theta_final = np.deg2rad(
    10.0
)


# 最终位置，仅用于打印检查

p_final = (
    p_initial
    +
    delta_position
)


print("\n========== Final Desired Position ==========")
print(p_final)


# ============================================================
# 8. 平滑轨迹时间
# ============================================================
#
# 0 ~ 1 秒：
#
#     保持初始位姿
#
#
# 1 ~ 4 秒：
#
#     用 smoothstep 平滑移动
#
#
# 4 秒以后：
#
#     保持最终位姿
#
#
# 这样可以避免直接给：
#
# 3 cm + 10°
#
# 的瞬间阶跃。
# ============================================================

motion_start_time = 1.0

motion_duration = 3.0


# ============================================================
# 9. Smoothstep
# ============================================================
#
# r 属于 [0, 1]
#
# s(r) =
#
#     3r² - 2r³
#
#
# 特点：
#
# s(0) = 0
# s(1) = 1
#
# s_dot(0) = 0
# s_dot(1) = 0
#
#
# 所以轨迹开始和结束时速度都为0。
# ============================================================

def smoothstep_trajectory(t):

    # --------------------------------------------------------
    # 轨迹还没开始
    # --------------------------------------------------------

    if t <= motion_start_time:

        s = 0.0
        s_dot = 0.0

        return s, s_dot


    # --------------------------------------------------------
    # 轨迹已经结束
    # --------------------------------------------------------

    if t >= (
        motion_start_time
        +
        motion_duration
    ):

        s = 1.0
        s_dot = 0.0

        return s, s_dot


    # --------------------------------------------------------
    # 当前位于运动过程
    # --------------------------------------------------------

    r = (
        (t - motion_start_time)
        /
        motion_duration
    )


    # smoothstep position

    s = (
        3.0 * r**2
        -
        2.0 * r**3
    )


    # ds/dr =
    #
    # 6r - 6r²
    #
    # 而：
    #
    # dr/dt = 1/T
    #
    # 所以：
    #
    # ds/dt =
    #
    # (6r - 6r²) / T

    s_dot = (
        (6.0 * r - 6.0 * r**2)
        /
        motion_duration
    )


    return s, s_dot


# ============================================================
# 10. 根据时间产生 desired 6D pose + velocity
# ============================================================

def desired_trajectory(t):

    # smoothstep进度

    s, s_dot = smoothstep_trajectory(
        t
    )


    # ========================================================
    # A. desired position
    # ========================================================

    p_des = (
        p_initial
        +
        s
        *
        delta_position
    )


    # ========================================================
    # B. desired linear velocity
    # ========================================================
    #
    # p_des =
    #
    # p0 + s * delta_p
    #
    # 因此：
    #
    # v_des =
    #
    # s_dot * delta_p
    # ========================================================

    v_des = (
        s_dot
        *
        delta_position
    )


    # ========================================================
    # C. desired orientation
    # ========================================================
    #
    # 旋转角：
    #
    # theta(t) =
    #
    # s(t) * theta_final
    # ========================================================

    theta = (
        s
        *
        theta_final
    )


    R_delta = np.array([
        [
            np.cos(theta),
            -np.sin(theta),
            0.0
        ],
        [
            np.sin(theta),
            np.cos(theta),
            0.0
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])


    # 左乘：
    #
    # 表示绕 WORLD Z 轴旋转

    R_des = (
        R_delta
        @
        R_initial
    )


    # ========================================================
    # D. desired angular velocity
    # ========================================================
    #
    # 因为整个目标姿态始终绕：
    #
    # WORLD Z
    #
    # 旋转，
    #
    # 所以目标角速度就是：
    #
    # [0, 0, theta_dot]
    #
    #
    # theta_dot =
    #
    # s_dot * theta_final
    # ========================================================

    theta_dot = (
        s_dot
        *
        theta_final
    )


    omega_des = np.array([
        0.0,
        0.0,
        theta_dot
    ])


    return (
        p_des,
        R_des,
        v_des,
        omega_des,
        s
    )


# ============================================================
# 11. Cartesian stiffness
# ============================================================
#
# 使用已经验证稳定的参数。
# ============================================================

K_position = np.array([
    200.0,
    200.0,
    200.0
])


K_orientation = np.array([
    1.0,
    1.0,
    1.0
])


# ============================================================
# 12. 计算初始 Operational-space inertia
# ============================================================
#
# Lambda =
#
#       (J M^-1 J^T)^-1
#
#
# 这里 Lambda 不直接参与控制律。
#
# 只用于计算合理的 damping。
# ============================================================

J_initial = pin.computeFrameJacobian(
    pin_model,
    pin_data,
    q_pin_initial,
    ee_id,
    pin.LOCAL_WORLD_ALIGNED
)


M_initial = pin.crba(
    pin_model,
    pin_data,
    q_pin_initial
)


M_initial = (
    M_initial
    +
    M_initial.T
) / 2.0


# M^-1 J^T

M_inv_JT = np.linalg.solve(
    M_initial,
    J_initial.T
)


Lambda_inv = (
    J_initial
    @
    M_inv_JT
)


Lambda = np.linalg.inv(

    Lambda_inv

    +

    1e-8
    *
    np.eye(6)

)


print(
    "\n========== Initial Operational Inertia =========="
)

print(
    "Lambda diagonal ="
)

print(
    np.diag(Lambda)
)


# ============================================================
# 13. 根据 Lambda 计算 Cartesian damping
# ============================================================
#
# 对每一个 Cartesian 方向近似：
#
# lambda_i xdd
# +
# D_i xd
# +
# K_i x
#
# = 0
#
#
# 临界阻尼：
#
# D_i =
#
# 2 * sqrt(lambda_i * K_i)
# ============================================================

K_6d = np.concatenate([
    K_position,
    K_orientation
])


zeta = 1.0


D_6d = (

    2.0

    *

    zeta

    *

    np.sqrt(

        np.maximum(
            np.diag(Lambda),
            1e-8
        )

        *

        K_6d
    )

)


D_position = D_6d[:3]

D_orientation = D_6d[3:]


print(
    "\n========== Cartesian Damping =========="
)

print(
    "D_position ="
)

print(
    D_position
)

print(
    "D_orientation ="
)

print(
    D_orientation
)


# ============================================================
# 14. Torque limits
# ============================================================

tau_limit = np.array([
    50.0,
    50.0,
    28.0,
    12.0,
    12.0,
    3.0
])


# ============================================================
# 15. 6D Cartesian Impedance Controller
# ============================================================

def controller():

    # ========================================================
    # A. 当前 joint state
    # ========================================================

    q_mj = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()


    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # ========================================================
    # B. 当前 TCP 位姿
    # ========================================================

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin
    )

    pin.updateFramePlacements(
        pin_model,
        pin_data
    )


    T = pin_data.oMf[ee_id]


    p = T.translation.copy()

    R = T.rotation.copy()


    # ========================================================
    # C. 当前 desired trajectory
    # ========================================================

    (
        p_des,
        R_des,
        v_des,
        omega_des,
        trajectory_progress
    ) = desired_trajectory(
        mj_data.time
    )


    # ========================================================
    # D. Position error
    # ========================================================

    e_position = (
        p_des
        -
        p
    )


    # ========================================================
    # E. Orientation error
    # ========================================================
    #
    # 使用 WORLD 表达：
    #
    # R_error =
    #
    # R_des R^T
    #
    #
    # 和 LOCAL_WORLD_ALIGNED Jacobian 对应。
    # ========================================================

    R_error = (
        R_des
        @
        R.T
    )


    e_orientation = pin.log3(
        R_error
    )


    # ========================================================
    # F. Frame Jacobian
    # ========================================================

    J = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q_pin,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # ========================================================
    # G. 当前 TCP velocity
    # ========================================================

    twist = (
        J
        @
        dq
    )


    v = twist[:3]

    omega = twist[3:]


    # ========================================================
    # H. Translation impedance
    # ========================================================
    #
    # 注意：
    #
    # 现在目标在运动。
    #
    # 所以 damping 应该使用：
    #
    # v_des - v
    #
    # 而不是简单：
    #
    # -v
    #
    #
    # 控制律：
    #
    # F =
    #
    # K (p_des - p)
    #
    # +
    #
    # D (v_des - v)
    # ========================================================

    force_control = (

        K_position
        *
        e_position

        +

        D_position
        *
        (
            v_des
            -
            v
        )

    )


    # ========================================================
    # I. Orientation impedance
    # ========================================================
    #
    # M =
    #
    # K_R e_R
    #
    # +
    #
    # D_R
    # (
    # omega_des - omega
    # )
    # ========================================================

    moment_control = (

        K_orientation
        *
        e_orientation

        +

        D_orientation
        *
        (
            omega_des
            -
            omega
        )

    )


    # ========================================================
    # J. 组成 6D wrench
    # ========================================================

    wrench = np.concatenate([
        force_control,
        moment_control
    ])


    # ========================================================
    # K. Cartesian wrench -> Joint torque
    # ========================================================

    tau_task = (
        J.T
        @
        wrench
    )


    # ========================================================
    # L. Gravity compensation
    # ========================================================

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # ========================================================
    # M. 最终 impedance torque
    # ========================================================
    #
    # 注意：
    #
    # 没有：
    #
    # M(q) qdd_cmd
    #
    # 没有：
    #
    # Jdot
    #
    # 没有：
    #
    # inverse dynamics tracking
    #
    #
    # 仍然是标准的：
    #
    # Cartesian spring/damper
    #
    # +
    #
    # gravity compensation
    # ========================================================

    tau = (
        gravity
        +
        tau_task
    )


    # ========================================================
    # N. Torque saturation
    # ========================================================

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    # ========================================================
    # O. 输出给 MuJoCo
    # ========================================================

    mj_data.ctrl[:] = tau


    return (
        p,
        R,
        p_des,
        R_des,
        e_position,
        e_orientation,
        v,
        omega,
        v_des,
        omega_des,
        force_control,
        moment_control,
        gravity,
        tau_task,
        tau,
        trajectory_progress,
        J
    )


# ============================================================
# 16. Simulation
# ============================================================

step = 0


# 8 秒以后结束实验
#
# 时间安排：
#
# 0~1：
#     初始保持
#
# 1~4：
#     移动
#
# 4~8：
#     最终保持 / 收敛

simulation_end_time = 8.0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:


    while viewer.is_running():

        (
            p,
            R,
            p_des,
            R_des,
            e_position,
            e_orientation,
            v,
            omega,
            v_des,
            omega_des,
            force_control,
            moment_control,
            gravity,
            tau_task,
            tau,
            trajectory_progress,
            J
        ) = controller()


        # ====================================================
        # MuJoCo 动力学积分
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        viewer.sync()


        # ====================================================
        # 每500步打印一次
        #
        # 当前 timestep 为2ms时：
        #
        # 500步约1秒
        # ====================================================

        if step % 500 == 0:

            print(
                "\n"
                "========== Moving 6D Cartesian Impedance =========="
            )


            print(
                "time =",
                mj_data.time
            )


            print(
                "trajectory progress =",
                trajectory_progress
            )


            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            print(
                "\nCurrent position:"
            )

            print(
                p
            )


            print(
                "\nDesired position:"
            )

            print(
                p_des
            )


            print(
                "\nPosition error:"
            )

            print(
                e_position
            )


            print(
                "Position error norm [mm] =",
                np.linalg.norm(
                    e_position
                )
                *
                1000.0
            )


            # ------------------------------------------------
            # Orientation
            # ------------------------------------------------

            print(
                "\nOrientation error:"
            )

            print(
                e_orientation
            )


            print(
                "Orientation error [deg] =",
                np.rad2deg(
                    np.linalg.norm(
                        e_orientation
                    )
                )
            )


            # ------------------------------------------------
            # Velocities
            # ------------------------------------------------

            print(
                "\nDesired linear velocity:"
            )

            print(
                v_des
            )


            print(
                "\nCurrent linear velocity:"
            )

            print(
                v
            )


            print(
                "\nDesired angular velocity:"
            )

            print(
                omega_des
            )


            print(
                "\nCurrent angular velocity:"
            )

            print(
                omega
            )


            # ------------------------------------------------
            # Cartesian wrench
            # ------------------------------------------------

            print(
                "\nControl force:"
            )

            print(
                force_control
            )


            print(
                "\nControl moment:"
            )

            print(
                moment_control
            )


            # ------------------------------------------------
            # Joint data
            # ------------------------------------------------

            print(
                "\nJoint velocity:"
            )

            print(
                mj_data.qvel
            )


            print(
                "\nGravity torque:"
            )

            print(
                gravity
            )


            print(
                "\nTask torque:"
            )

            print(
                tau_task
            )


            print(
                "\nFinal joint torque:"
            )

            print(
                tau
            )


            # ------------------------------------------------
            # Numerical / contact diagnosis
            # ------------------------------------------------

            print(
                "\nJacobian condition number =",
                np.linalg.cond(J)
            )


            print(
                "Contacts =",
                mj_data.ncon
            )


            print(
                "Constraint generalized force:"
            )

            print(
                mj_data.qfrc_constraint
            )


        step += 1


        # ====================================================
        # 自动结束
        # ====================================================

        if mj_data.time >= simulation_end_time:

            break


        # 仅控制现实播放速度

        time.sleep(
            mj_model.opt.timestep
        )