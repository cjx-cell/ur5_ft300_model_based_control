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

mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
mj_data = mujoco.MjData(mj_model)


# ============================================================
# 2. 加载 Pinocchio
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

pin_model = pin.buildModelFromUrdf(urdf_path)
pin_data = pin_model.createData()


# ============================================================
# 3. 获取末端 frame
# ============================================================

ee_id = pin_model.getFrameId("tool0")

print("Pinocchio tool0 frame id =", ee_id)


# ============================================================
# 4. MuJoCo 中用于施加外力的末端 body
# ============================================================

mj_ee_body_id = mujoco.mj_name2id(
    mj_model,
    mujoco.mjtObj.mjOBJ_BODY,
    "tool0"
)

if mj_ee_body_id == -1:

    mj_ee_body_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        "wrist_3_link"
    )

if mj_ee_body_id == -1:
    raise RuntimeError(
        "MuJoCo 中找不到 tool0 或 wrist_3_link"
    )


print(
    "MuJoCo end-effector body =",
    mujoco.mj_id2name(
        mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        mj_ee_body_id
    )
)


# ============================================================
# 5. MuJoCo q -> Pinocchio q
# ============================================================

def mujoco_to_pinocchio(q_mj):

    q_pin = pin.neutral(pin_model)

    # 前5个普通旋转关节
    q_pin[:5] = q_mj[:5]

    # 第6轴是 JointModelRUBZ
    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 6. 初始姿态
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
# 7. 获取初始 TCP 位姿
#
# 我们把初始位姿作为 impedance equilibrium pose。
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

p_des = T_initial.translation.copy()
R_des = T_initial.rotation.copy()


print("\nEquilibrium position:")
print(p_des)

print("\nEquilibrium rotation:")
print(R_des)


# ============================================================
# 8. Cartesian stiffness
# ============================================================
#
# 平移：
#
# K = 200 N/m
#
# 因此 +X 方向施加 2N 后，
#
# 理论稳态位移：
#
# dx = 2 / 200 = 0.01 m = 10 mm
#
#
# 姿态：
#
# 先使用非常柔软的 1 Nm/rad。
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
# 9. 根据 Operational-space inertia 估算阻尼
# ============================================================
#
# 前面我们一直人工写：
#
# D_position = 30
# D_orientation = 0.5
#
# 但真正合理的阻尼应该和机器人在当前姿态下的
# 等效惯性有关。
#
#
# 末端等效惯性：
#
# Lambda =
#
#       ( J M^-1 J^T )^-1
#
#
# 如果暂时把每个笛卡尔方向近似成独立二阶系统：
#
# lambda_i * xdd
# +
# D_i * xd
# +
# K_i * x
# = 0
#
#
# 临界阻尼近似：
#
# D_i =
#
#     2 * sqrt(lambda_i * K_i)
#
#
# 这里 Lambda 只用于“算阻尼参数”。
#
# 不参与后面的反馈线性化。
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


# ------------------------------------------------------------
# M^-1 J^T
#
# 不显式写 inv(M)
# ------------------------------------------------------------

M_inv_JT = np.linalg.solve(
    M_initial,
    J_initial.T
)


Lambda_inv = (
    J_initial
    @
    M_inv_JT
)


# 数值正则化
#
# 当前 Jacobian condition number 约16，
# 不算奇异，但仍加一个很小的 regularization。

lambda_regularization = 1e-8


Lambda = np.linalg.inv(
    Lambda_inv
    +
    lambda_regularization * np.eye(6)
)


print("\n========== Initial Operational Inertia ==========")
print("Lambda =")
print(Lambda)

print("\nLambda diagonal =")
print(np.diag(Lambda))


# ============================================================
# 10. 根据 Lambda 自动计算 damping
# ============================================================

K_6d = np.concatenate([
    K_position,
    K_orientation
])


# damping ratio
#
# zeta = 1
#
# 对应近似 critical damping

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


print("\n========== Automatically Tuned Damping ==========")

print("D_position =")
print(D_position)

print("D_orientation =")
print(D_orientation)


# ============================================================
# 11. 外部力实验开关
# ============================================================
#
# !!! 第一次运行先保持 False !!!
#
# 我们首先验证：
#
# 无外力时6D impedance自身是否稳定。
#
# 如果0~5秒完全稳定，
# 再改成True测试2N外力。
# ============================================================

ENABLE_EXTERNAL_FORCE = True


external_force = np.array([
    2.0,
    0.0,
    0.0
])

external_torque = np.zeros(3)


force_start_time = 2.0
force_end_time = 6.0


# ============================================================
# 12. Torque limits
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
# 13. 6D Cartesian impedance controller
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
    # B. Forward Kinematics
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
    # C. Position error
    # ========================================================

    e_position = (
        p_des
        -
        p
    )


    # ========================================================
    # D. Orientation error
    # ========================================================
    #
    # R_error =
    #
    #       R_des * R^T
    #
    #
    # 这个误差是在 WORLD 坐标方向表达的。
    #
   # 与 LOCAL_WORLD_ALIGNED Jacobian 配套。
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
    # E. Frame Jacobian
    # ========================================================

    J = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q_pin,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # ========================================================
    # F. TCP twist
    # ========================================================
    #
    # Pinocchio：
    #
    # 前3维 linear
    # 后3维 angular
    #
    # 官方 Pinocchio spatial vector 的 LINEAR index = 0，
    # ANGULAR index = 3。
    # ========================================================

    twist = (
        J
        @
        dq
    )


    v = twist[:3]
    omega = twist[3:]


    # ========================================================
    # G. Translational impedance
    # ========================================================

    force_control = (

        K_position
        *
        e_position

        -

        D_position
        *
        v
    )


    # ========================================================
    # H. Rotational impedance
    # ========================================================

    moment_control = (

        K_orientation
        *
        e_orientation

        -

        D_orientation
        *
        omega
    )


    # ========================================================
    # I. 6D Cartesian wrench
    # ========================================================
    #
    # 顺序：
    #
    # [Fx, Fy, Fz, Mx, My, Mz]
    # ========================================================

    wrench = np.concatenate([
        force_control,
        moment_control
    ])


    # ========================================================
    # J. Cartesian wrench -> joint torque
    # ========================================================

    tau_task = (
        J.T
        @
        wrench
    )


    # ========================================================
    # K. Gravity compensation
    # ========================================================
    #
    # 注意：
    #
    # gravity 必须从第一个仿真周期完整输出。
    # ========================================================

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # ========================================================
    # L. 最终 torque
    # ========================================================
    #
    # 这里已经没有任何 torque-rate limiter。
    #
    #
    # 标准基础 Cartesian impedance：
    #
    # tau =
    #
    #       J^T W
    #
    #       +
    #
    #       g(q)
    # ========================================================

    tau = (
        gravity
        +
        tau_task
    )


    # ========================================================
    # M. 执行器 torque magnitude limit
    # ========================================================

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    mj_data.ctrl[:] = tau


    return (
        p,
        e_position,
        e_orientation,
        v,
        omega,
        force_control,
        moment_control,
        gravity,
        tau_task,
        tau,
        J
    )


# ============================================================
# 14. 模拟外部力
# ============================================================

def apply_external_force():

    # 清除上一周期外部 generalized force

    mj_data.qfrc_applied[:] = 0.0


    # 第一次实验关闭外力

    if not ENABLE_EXTERNAL_FORCE:
        return np.zeros(3)


    t = mj_data.time


    if (
        force_start_time
        <= t
        <
        force_end_time
    ):

        # 获取当前 TCP 世界坐标

        q_pin = mujoco_to_pinocchio(
            mj_data.qpos
        )


        pin.forwardKinematics(
            pin_model,
            pin_data,
            q_pin
        )

        pin.updateFramePlacements(
            pin_model,
            pin_data
        )


        tcp_point = (
            pin_data
            .oMf[ee_id]
            .translation
            .copy()
        )


        # 把 WORLD X 方向2N
        # 施加到TCP点

        mujoco.mj_applyFT(
            mj_model,
            mj_data,

            external_force,
            external_torque,

            tcp_point,

            mj_ee_body_id,

            mj_data.qfrc_applied
        )


        return external_force.copy()


    return np.zeros(3)


# ============================================================
# 15. Simulation
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:


    while viewer.is_running():

        (
            p,
            e_position,
            e_orientation,
            v,
            omega,
            force_control,
            moment_control,
            gravity,
            tau_task,
            tau,
            J
        ) = controller()


        F_external_now = (
            apply_external_force()
        )


        mujoco.mj_step(
            mj_model,
            mj_data
        )


        viewer.sync()


        # ====================================================
        # Debug output
        # ====================================================

        if step % 500 == 0:

            displacement = (
                p
                -
                p_des
            )


            print(
                "\n"
                "========== 6D Cartesian Impedance =========="
            )


            print(
                "time =",
                mj_data.time
            )


            print(
                "\nPosition displacement [m]:"
            )

            print(
                displacement
            )


            print(
                "Position error norm [mm] =",
                np.linalg.norm(
                    displacement
                )
                *
                1000.0
            )


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


            print(
                "\nExternal force:"
            )

            print(
                F_external_now
            )


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


            print(
                "\nLinear velocity:"
            )

            print(
                v
            )


            print(
                "\nAngular velocity:"
            )

            print(
                omega
            )


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


            print(
                "\nJacobian condition number =",
                np.linalg.cond(J)
            )


            print(
                "\nContacts =",
                mj_data.ncon
            )


            print(
                "Constraint generalized force:"
            )

            print(
                mj_data.qfrc_constraint
            )


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )