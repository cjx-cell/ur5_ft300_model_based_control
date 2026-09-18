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
# 3. 获取 tool0 frame
# ============================================================

ee_name = "tool0"

ee_id = pin_model.getFrameId(
    ee_name
)

print("Pinocchio tool0 frame id =", ee_id)


# ============================================================
# 4. 找到 MuJoCo 末端 body
# ============================================================

# Pinocchio 的 frame id 和 MuJoCo body id 完全不是一回事。
#
# URDF -> MuJoCo 后，tool0 这种 fixed link 有可能被融合掉。
#
# 所以优先找 tool0；
# 如果 MuJoCo 里不存在，则使用 wrist_3_link。

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
        "MuJoCo 中没有找到 tool0 或 wrist_3_link body"
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

    q_pin = pin.neutral(
        pin_model
    )

    # 前五轴直接复制

    q_pin[:5] = q_mj[:5]

    # 第六轴为 JointModelRUBZ

    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 6. 设置初始姿态
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
# 7. 得到初始 TCP 位置
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


print("\nInitial / equilibrium TCP position:")
print(p_des)


# ============================================================
# 8. Cartesian stiffness
# ============================================================

# 单位：
#
# N / m
#
# K越大：
#
# 机器人越“硬”
#
# K越小：
#
# 机器人越“软”

K_cart = np.array([
    200.0,
    200.0,
    200.0
])


# ============================================================
# 9. Cartesian damping
# ============================================================

# 单位：
#
# N*s/m
#
# D用于抑制弹簧振荡。

D_cart = np.array([
    30.0,
    30.0,
    30.0
])


# ============================================================
# 10. Joint damping
# ============================================================

# 当前只控制 TCP 的 x/y/z。
#
# UR3 有6自由度，
# 因此还有未被平移任务直接约束的方向。
#
# 加少量关节阻尼防止这些方向自由漂移。

joint_damping = np.array([
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.3
])


# ============================================================
# 11. Torque limit
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
# 12. 外部测试力
# ============================================================

# 我们人为模拟：
#
# 有人沿 WORLD X 方向推 TCP。
#
#
# F_external =
#
# [2N, 0, 0]

external_force = np.array([
    2.0,
    0.0,
    0.0
])


# 不施加额外力矩

external_torque = np.zeros(3)


# ============================================================
# 13. 外力作用时间
# ============================================================

# 0~2秒：
#
# 不推机器人
#
#
# 2~6秒：
#
# +X方向施加2N
#
#
# 6秒以后：
#
# 撤掉外力

force_start_time = 2.0

force_end_time = 6.0


# ============================================================
# 14. Impedance Controller
# ============================================================

def controller():

    # --------------------------------------------------------
    # A. 获取当前状态
    # --------------------------------------------------------

    q_mj = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()


    # --------------------------------------------------------
    # B. 转 Pinocchio configuration
    # --------------------------------------------------------

    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # --------------------------------------------------------
    # C. Forward Kinematics
    # --------------------------------------------------------

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin
    )

    pin.updateFramePlacements(
        pin_model,
        pin_data
    )


    T_ee = pin_data.oMf[ee_id]


    # 当前TCP位置

    p = T_ee.translation.copy()


    # --------------------------------------------------------
    # D. Cartesian position error
    # --------------------------------------------------------

    # p_des 是机械臂没有受到外力时的“平衡位置”。
    #
    # 当外力把机械臂推开时：
    #
    # error != 0
    #
    # 虚拟弹簧就会产生恢复力。

    error = (
        p_des
        -
        p
    )


    # --------------------------------------------------------
    # E. Jacobian
    # --------------------------------------------------------

    J = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q_pin,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # 我们当前只研究平移阻抗，
    # 所以只取线速度Jacobian。

    J_linear = J[:3, :]


    # --------------------------------------------------------
    # F. 当前TCP线速度
    # --------------------------------------------------------

    velocity = (
        J_linear
        @
        dq
    )


    # --------------------------------------------------------
    # G. 虚拟弹簧 + 虚拟阻尼
    # --------------------------------------------------------

    # 控制器内部产生：
    #
    # F_control =
    #
    # K * (p_des - p)
    #
    # -
    #
    # D * velocity
    #
    #
    # 当TCP被外力推离平衡位置：
    #
    # spring force会把机器人拉回来。

    F_control = (

        K_cart
        *
        error

        -

        D_cart
        *
        velocity
    )


    # --------------------------------------------------------
    # H. Cartesian Force -> Joint Torque
    # --------------------------------------------------------

    tau_task = (
        J_linear.T
        @
        F_control
    )


    # --------------------------------------------------------
    # I. Gravity Compensation
    # --------------------------------------------------------

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # --------------------------------------------------------
    # J. 最终控制力矩
    # --------------------------------------------------------

    tau = (

        tau_task

        +

        gravity

        -

        joint_damping
        *
        dq
    )


    # --------------------------------------------------------
    # K. Torque limit
    # --------------------------------------------------------

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    # --------------------------------------------------------
    # L. 发给MuJoCo
    # --------------------------------------------------------

    mj_data.ctrl[:] = tau


    return (
        p,
        error,
        velocity,
        F_control,
        tau
    )


# ============================================================
# 15. 外力函数
# ============================================================

def apply_external_force():

    # --------------------------------------------------------
    # 每一仿真步先把上一周期外力清零。
    #
    # qfrc_applied 是 MuJoCo generalized external force。
    # --------------------------------------------------------

    mj_data.qfrc_applied[:] = 0.0


    current_time = mj_data.time


    # --------------------------------------------------------
    # 只有 2s ~ 6s 之间施加外力。
    # --------------------------------------------------------

    if (
        force_start_time
        <= current_time
        <
        force_end_time
    ):

        # ----------------------------------------------------
        # 当前 TCP 世界坐标。
        #
        # 我们希望力作用在 TCP 点，
        # 而不是随便作用在 wrist body 的质心。
        # ----------------------------------------------------

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
            pin_data.oMf[ee_id]
            .translation
            .copy()
        )


        # ----------------------------------------------------
        # MuJoCo mj_applyFT：
        #
        # 将：
        #
        # Cartesian force
        # Cartesian torque
        #
        # 作用到指定 body 上的指定 WORLD point，
        #
        # 并映射到：
        #
        # qfrc_applied
        #
        # generalized force。
        # ----------------------------------------------------

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
# 16. Simulation loop
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:

    while viewer.is_running():

        # ====================================================
        # Impedance controller
        # ====================================================

        (
            p,
            error,
            velocity,
            F_control,
            tau
        ) = controller()


        # ====================================================
        # External perturbation
        # ====================================================

        F_external_now = apply_external_force()


        # ====================================================
        # MuJoCo dynamics
        # ====================================================

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        viewer.sync()


        # ====================================================
        # 每500步打印
        # ====================================================

        if step % 500 == 0:

            displacement = (
                p
                -
                p_des
            )


            print(
                "\n========== Cartesian Impedance =========="
            )


            print(
                "time =",
                mj_data.time
            )


            print("\nTCP position:")
            print(p)


            print("\nEquilibrium position:")
            print(p_des)


            print("\nDisplacement:")
            print(displacement)


            print(
                "\nX displacement [m] =",
                displacement[0]
            )


            print(
                "X displacement [mm] =",
                displacement[0] * 1000.0
            )


            print("\nExternal force:")
            print(F_external_now)


            print("\nControl spring/damping force:")
            print(F_control)


            print("\nTCP velocity:")
            print(velocity)


            print("\nJoint torque:")
            print(tau)


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )