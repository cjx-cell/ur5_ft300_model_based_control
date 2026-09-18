import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 加载 MuJoCo
# ============================================================

xml_path = Path(__file__).parent / "ur3_converted.xml"

mj_model = mujoco.MjModel.from_xml_path(
    str(xml_path)
)

mj_data = mujoco.MjData(
    mj_model
)


# ============================================================
# 2. 加载 Pinocchio
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

pin_model = pin.buildModelFromUrdf(
    urdf_path
)

pin_data = pin_model.createData()


# ============================================================
# 3. tool0
# ============================================================

ee_name = "tool0"

ee_id = pin_model.getFrameId(
    ee_name
)

print("tool0 frame id =", ee_id)


# ============================================================
# 4. MuJoCo q -> Pinocchio q
# ============================================================

def mujoco_to_pinocchio(q_mj):

    q_pin = pin.neutral(
        pin_model
    )

    # 前5轴直接对应

    q_pin[:5] = q_mj[:5]


    # 第6轴是 Pinocchio JointModelRUBZ
    #
    # q =
    # [cos(theta6), sin(theta6)]

    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 5. 初始姿态
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
# 6. 计算初始 TCP 位姿
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

p_initial = T_initial.translation.copy()

R_initial = T_initial.rotation.copy()


print("\n========== Initial TCP ==========")

print("position =")
print(p_initial)

print("\nrotation =")
print(R_initial)


# ============================================================
# 7. 最终目标
# ============================================================

# ------------------------------------------------------------
# 平移：
#
# 世界 Z 方向 +3 cm
# ------------------------------------------------------------

delta_p = np.array([
    0.0,
    0.0,
    0.03
])


# ------------------------------------------------------------
# 姿态：
#
# 绕 WORLD Z 方向 +10°
# ------------------------------------------------------------

theta_final = np.deg2rad(
    10.0
)


# ============================================================
# 8. 轨迹过渡时间
# ============================================================

# 上一个程序的问题之一：
#
# t = 0 时目标直接跳：
#
#   position +3cm
#   orientation +10°
#
# 这叫 step reference。
#
# 对真实机器人尤其不友好。
#
# 这里让目标在3秒内平滑变化。

trajectory_time = 3.0


# ============================================================
# 9. Cartesian gains
# ============================================================

# ------------------------------------------------------------
# 这里的增益不再直接产生 Newton / Nm。
#
# 它们产生的是“期望笛卡尔加速度”。
#
# position:
#
# m -> m/s²
#
# orientation:
#
# rad -> rad/s²
# ------------------------------------------------------------

Kp_position = np.array([
    25.0,
    25.0,
    25.0
])

Kd_position = np.array([
    10.0,
    10.0,
    10.0
])


Kp_orientation = np.array([
    16.0,
    16.0,
    16.0
])

Kd_orientation = np.array([
    8.0,
    8.0,
    8.0
])


# ============================================================
# 10. Damped Least Squares 参数
# ============================================================

# J 在某些姿态附近可能接近奇异。
#
# 如果直接：
#
# inv(J)
#
# 数值可能非常大。
#
# 所以使用：
#
# J# = J^T (J J^T + lambda² I)^-1

damping_lambda = 0.03


# ============================================================
# 11. 力矩限制
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
# 12. 生成平滑 Cartesian 目标
# ============================================================

def get_desired_pose(sim_time):

    # --------------------------------------------------------
    # r:
    #
    # 从0逐渐变化到1。
    # --------------------------------------------------------

    r = np.clip(
        sim_time / trajectory_time,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # smoothstep:
    #
    # s = 3r² - 2r³
    #
    # 它满足：
    #
    # s(0)=0
    # s(1)=1
    #
    # 并且两端速度为0。
    # --------------------------------------------------------

    s = (
        3.0 * r**2
        -
        2.0 * r**3
    )


    # ========================================================
    # Desired position
    # ========================================================

    p_des = (
        p_initial
        +
        s * delta_p
    )


    # ========================================================
    # Desired orientation
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
    # 表示绕 WORLD Z 旋转

    R_des = (
        R_delta
        @
        R_initial
    )


    return (
        p_des,
        R_des
    )


# ============================================================
# 13. Cartesian Computed Torque Controller
# ============================================================

def controller():

    # ========================================================
    # A. 当前状态
    # ========================================================

    q_mj = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()


    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # ========================================================
    # B. 当前仿真时间
    # ========================================================

    sim_time = mj_data.time


    # ========================================================
    # C. 获取平滑目标
    # ========================================================

    p_des, R_des = get_desired_pose(
        sim_time
    )


    # ========================================================
    # D. Forward Kinematics
    # ========================================================

    # 注意：
    #
    # 这里给 forwardKinematics 传入 q 和 dq。
    #
    # 因为后面我们不仅需要位置，
    # 还需要速度/Jdot。

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin,
        dq
    )


    pin.updateFramePlacements(
        pin_model,
        pin_data
    )


    T_ee = pin_data.oMf[ee_id]

    p = T_ee.translation.copy()

    R = T_ee.rotation.copy()


    # ========================================================
    # E. 位置误差
    # ========================================================

    e_position = (
        p_des
        -
        p
    )


    # ========================================================
    # F. 姿态误差
    # ========================================================

    # 当前姿态到目标姿态的 world-frame rotation error

    R_error = (
        R_des
        @
        R.T
    )


    # SO(3) -> rotation vector

    e_orientation = pin.log3(
        R_error
    )


    # ========================================================
    # G. 计算 J 和 Jdot
    # ========================================================

    # 这个函数同时计算：
    #
    # data.J
    # data.dJ

    pin.computeJointJacobiansTimeVariation(
        pin_model,
        pin_data,
        q_pin,
        dq
    )


    # 当前 frame Jacobian
    #
    # LOCAL_WORLD_ALIGNED：
    #
    # frame原点
    # +
    # world方向

    J = pin.getFrameJacobian(
        pin_model,
        pin_data,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # Jacobian 时间导数

    dJ = pin.getFrameJacobianTimeVariation(
        pin_model,
        pin_data,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # ========================================================
    # H. 当前末端 twist
    # ========================================================

    # [vx vy vz wx wy wz]

    twist = (
        J
        @
        dq
    )


    v = twist[:3]

    omega = twist[3:]


    # ========================================================
    # I. Cartesian desired acceleration
    # ========================================================

    # --------------------------------------------------------
    # 平移：
    #
    # a_cmd =
    #
    # Kp * position_error
    #
    # -
    #
    # Kd * current_velocity
    #
    # --------------------------------------------------------

    a_linear_cmd = (

        Kp_position
        *
        e_position

        -

        Kd_position
        *
        v
    )


    # --------------------------------------------------------
    # 旋转：
    #
    # alpha_cmd =
    #
    # Kp * orientation_error
    #
    # -
    #
    # Kd * angular_velocity
    # --------------------------------------------------------

    a_angular_cmd = (

        Kp_orientation
        *
        e_orientation

        -

        Kd_orientation
        *
        omega
    )


    # --------------------------------------------------------
    # 合成6D期望空间加速度
    # --------------------------------------------------------

    a_task_cmd = np.concatenate([
        a_linear_cmd,
        a_angular_cmd
    ])


    # ========================================================
    # J. 考虑 Jdot * dq
    # ========================================================

    # 末端真实加速度：
    #
    # xdd =
    #
    # J qdd
    #
    # +
    #
    # Jdot dq
    #
    #
    # 所以：
    #
    # J qdd =
    #
    # xdd_cmd - Jdot dq

    rhs = (

        a_task_cmd

        -

        dJ @ dq
    )


    # ========================================================
    # K. Damped Least Squares
    # ========================================================

    # J:
    #
    # 6 x 6
    #
    # 但是我们不用直接 inv(J)。
    #
    # 使用阻尼伪逆：
    #
    # J# =
    #
    # J.T @
    #
    # inv(
    #     J J.T
    #     +
    #     lambda² I
    # )


    A = (

        J @ J.T

        +

        damping_lambda**2
        *
        np.eye(6)
    )


    # 不显式求 inverse。
    #
    # solve(A, rhs)
    #
    # 比：
    #
    # inv(A) @ rhs
    #
    # 数值更稳定。

    qdd_cmd = (

        J.T

        @

        np.linalg.solve(
            A,
            rhs
        )
    )


    # ========================================================
    # L. Pinocchio Mass Matrix
    # ========================================================

    M = pin.crba(
        pin_model,
        pin_data,
        q_pin
    )


    # CRBA返回的理论质量矩阵是对称的。
    #
    # 数值上再显式对称一次。

    M = (
        M
        +
        M.T
    ) / 2.0


    # ========================================================
    # M. Nonlinear Effects
    # ========================================================

    # h(q,dq) =
    #
    # C(q,dq)dq
    #
    # +
    #
    # g(q)

    h = pin.nonLinearEffects(
        pin_model,
        pin_data,
        q_pin,
        dq
    )


    # ========================================================
    # N. Computed Torque
    # ========================================================

    # 逆动力学：
    #
    # tau =
    #
    # M qdd_cmd
    #
    # +
    #
    # h

    tau = (

        M
        @
        qdd_cmd

        +

        h
    )


    # ========================================================
    # O. Torque limit
    # ========================================================

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    # ========================================================
    # P. 发给 MuJoCo
    # ========================================================

    mj_data.ctrl[:] = tau


    return (
        p,
        p_des,
        e_position,
        e_orientation,
        v,
        omega,
        qdd_cmd,
        tau
    )


# ============================================================
# 14. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:

    while viewer.is_running():

        (
            p,
            p_des,
            e_position,
            e_orientation,
            v,
            omega,
            qdd_cmd,
            tau
        ) = controller()


        # ====================================================
        # MuJoCo推进一步
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        viewer.sync()


        # ====================================================
        # Debug
        # ====================================================

        if step % 1000 == 0:

            print(
                "\n========== Cartesian Computed Torque =========="
            )


            print("\ntime =", mj_data.time)


            print("\nCurrent position:")
            print(p)


            print("\nDesired position:")
            print(p_des)


            print("\nPosition error:")
            print(e_position)


            print(
                "Position error norm [m] =",
                np.linalg.norm(
                    e_position
                )
            )


            print("\nOrientation error:")
            print(e_orientation)


            print(
                "Orientation error [deg] =",
                np.rad2deg(
                    np.linalg.norm(
                        e_orientation
                    )
                )
            )


            print("\nLinear velocity:")
            print(v)


            print("\nAngular velocity:")
            print(omega)


            print("\nqdd_cmd:")
            print(qdd_cmd)


            print("\nJoint torque:")
            print(tau)


            # ------------------------------------------------
            # 额外检查 MuJoCo contact/constraint
            # ------------------------------------------------

            print(
                "\nMuJoCo contacts =",
                mj_data.ncon
            )


            print(
                "qfrc_constraint ="
            )

            print(
                mj_data.qfrc_constraint
            )


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )