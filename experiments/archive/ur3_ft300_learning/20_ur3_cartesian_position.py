import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 加载 MuJoCo 模型
# ============================================================

# MuJoCo：
#
# 负责真实的物理仿真：
#
# - 刚体动力学
# - 重力
# - 接触
# - 约束
# - actuator
# - 数值积分
#
# 控制器最终算出来的 tau 会送给 MuJoCo。

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

# Pinocchio：
#
# 不负责物理仿真。
#
# 我们用它计算：
#
# - Forward Kinematics
# - tool0位置
# - Jacobian
# - gravity
#
# 后面阻抗控制、导纳控制同样会大量使用。

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

pin_model = pin.buildModelFromUrdf(
    urdf_path
)

pin_data = pin_model.createData()


# ============================================================
# 3. 获取 tool0 frame
# ============================================================

# 我们现在不再控制：
#
# q1 q2 q3 ...
#
# 而是控制机器人末端 tool0 的空间位置。

ee_name = "tool0"

ee_id = pin_model.getFrameId(
    ee_name
)

print("tool0 frame id =", ee_id)


# ============================================================
# 4. MuJoCo configuration -> Pinocchio configuration
# ============================================================

def mujoco_to_pinocchio(q_mj):

    # --------------------------------------------------------
    # MuJoCo:
    #
    # q_mj =
    #
    # [q1 q2 q3 q4 q5 q6]
    #
    # shape = (6,)
    #
    #
    # Pinocchio：
    #
    # 前5轴普通 revolute：
    #
    # q1 q2 q3 q4 q5
    #
    # 第6轴为 JointModelRUBZ，
    # configuration 用两个数表达。
    #
    # 所以：
    #
    # nq = 7
    # nv = 6
    # --------------------------------------------------------

    q_pin = pin.neutral(
        pin_model
    )

    # 前5个普通转动关节直接复制

    q_pin[:5] = q_mj[:5]

    # 第6轴
    #
    # MuJoCo只保存实际角度 theta6。

    theta6 = q_mj[5]

    # Pinocchio unbounded revolute joint
    # configuration表示：

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 5. 设置机器人初始关节姿态
# ============================================================

# 使用之前已经验证稳定的姿态。
#
# 这里不是我们的“控制目标”。
#
# 它只是 Cartesian 控制开始时机器人的初始状态。

mj_data.qpos[:] = np.deg2rad([
    30.0,
    -60.0,
    90.0,
    -30.0,
    45.0,
    0.0
])

# 初始速度为0

mj_data.qvel[:] = 0.0


# 根据刚刚设置的 qpos
# 更新 MuJoCo 内部所有运动学量。

mujoco.mj_forward(
    mj_model,
    mj_data
)


# ============================================================
# 6. 获取初始 TCP 位置
# ============================================================

# MuJoCo当前角度

q_mj_initial = mj_data.qpos.copy()

# 转成Pinocchio configuration

q_pin_initial = mujoco_to_pinocchio(
    q_mj_initial
)


# Forward Kinematics
#
# 输入：
# q
#
# 得到：
# 各个joint的空间位姿

pin.forwardKinematics(
    pin_model,
    pin_data,
    q_pin_initial
)


# tool0属于frame，
# 所以更新frame placements。

pin.updateFramePlacements(
    pin_model,
    pin_data
)


# data.oMf[ee_id]
#
# 表示：
#
# world -> tool0
#
# 的SE(3)位姿。

T_initial = pin_data.oMf[ee_id]


# tool0初始位置

p_initial = T_initial.translation.copy()


print("\n========== Initial TCP ==========")

print("p_initial [m] =")
print(p_initial)


# ============================================================
# 7. 设置 Cartesian 目标
# ============================================================

# 我们不指定目标joint angle。
#
# 而是：
#
# 从当前TCP位置出发，
# 沿世界Z方向上升 5cm。
#
#
# 5cm = 0.05m

p_des = (
    p_initial
    +
    np.array([
        0.0,
        0.0,
        0.05
    ])
)


print("\n========== Desired TCP ==========")

print("p_des [m] =")
print(p_des)


# ============================================================
# 8. Cartesian PD 参数
# ============================================================

# 这里的Kp已经不是：
#
# N*m/rad
#
# 而是：
#
# N/m
#
# 可以把TCP想象成由三个虚拟弹簧拉向目标点。


Kp_cart = np.array([
    200.0,
    200.0,
    200.0
])


# Cartesian damping
#
# 单位近似：
#
# N*s/m

Kd_cart = np.array([
    30.0,
    30.0,
    30.0
])


# ============================================================
# 9. Joint damping
# ============================================================

# 一个重要问题：
#
# 我们当前只控制：
#
# x y z
#
# 这是3个任务。
#
# 但是UR3有6个自由度。
#
# 所以还有一些没有被Cartesian位置任务约束的自由度。
#
# 给关节加少量阻尼，
# 防止这些方向自由漂动。

joint_damping = np.array([
    1.0,
    1.0,
    1.0,
    0.5,
    0.5,
    0.0
])


# ============================================================
# 10. Torque limit
# ============================================================

# 和XML里面的actuator能力保持在一个合理范围。
#
# 第6轴暂时仍不参与控制。

tau_limit = np.array([
    50.0,
    50.0,
    28.0,
    12.0,
    12.0,
    0.0
])


# ============================================================
# 11. Cartesian Controller
# ============================================================

def controller():

    # ========================================================
    # A. 读取 MuJoCo 状态
    # ========================================================

    q_mj = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()


    # ========================================================
    # B. 转换给 Pinocchio
    # ========================================================

    q_pin = mujoco_to_pinocchio(
        q_mj
    )


    # ========================================================
    # C. Forward Kinematics
    # ========================================================

    # 根据当前q计算机器人运动学。

    pin.forwardKinematics(
        pin_model,
        pin_data,
        q_pin
    )

    pin.updateFramePlacements(
        pin_model,
        pin_data
    )


    # 当前tool0位姿

    T_ee = pin_data.oMf[ee_id]


    # 当前TCP位置

    p = T_ee.translation.copy()


    # ========================================================
    # D. 计算TCP位置误差
    # ========================================================

    # e =
    #
    # desired - current

    position_error = (
        p_des
        -
        p
    )


    # ========================================================
    # E. 计算 Frame Jacobian
    # ========================================================

    # LOCAL_WORLD_ALIGNED：
    #
    # Jacobian作用点：
    # tool0
    #
    # Jacobian XYZ方向：
    # 与世界坐标系对齐
    #
    # 所以后面的：
    #
    # X方向力
    # Y方向力
    # Z方向力
    #
    # 都很好理解。

    J = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q_pin,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # ========================================================
    # F. 提取线速度 Jacobian
    # ========================================================

    # J:
    #
    # 6 x 6
    #
    # 前三行：
    #
    # vx vy vz
    #
    # 后三行：
    #
    # wx wy wz

    J_linear = J[:3, :]


    # ========================================================
    # G. 计算当前TCP线速度
    # ========================================================

    # 根据：
    #
    # v_tcp = Jv(q) * dq

    v_tcp = (
        J_linear
        @
        dq
    )


    # ========================================================
    # H. Cartesian PD
    # ========================================================

    # 我们在任务空间构造一个“虚拟力”：
    #
    #
    # F =
    #
    # Kp * position_error
    #
    # -
    #
    # Kd * tcp_velocity
    #
    #
    # 第一项：
    #
    # 虚拟弹簧
    #
    #
    # 第二项：
    #
    # 虚拟阻尼

    force_task = (
        Kp_cart * position_error
        -
        Kd_cart * v_tcp
    )


    # ========================================================
    # I. Cartesian Force -> Joint Torque
    # ========================================================

    # 之前已经推导过：
    #
    # tau = J^T F
    #
    #
    # 这里因为只控制位置，
    # 所以使用：
    #
    # J_linear.T
    #
    # shape：
    #
    # J_linear.T = 6 x 3
    #
    # F            = 3 x 1
    #
    # tau_task     = 6 x 1

    tau_task = (
        J_linear.T
        @
        force_task
    )


    # ========================================================
    # J. Gravity Compensation
    # ========================================================

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # ========================================================
    # K. 最终控制力矩
    # ========================================================

    # Cartesian task：
    #
    # J^T F
    #
    # +
    #
    # 重力补偿
    #
    # -
    #
    # 少量joint damping

    tau = (
        tau_task
        +
        gravity
        -
        joint_damping * dq
    )


    # ========================================================
    # L. 暂时关闭 wrist3
    # ========================================================

    # 之前已经验证：
    #
    # wrist3 continuous joint
    # 需要以后单独处理。
    #
    # 当前我们的任务只是：
    #
    # TCP位置控制。

    tau[5] = 0.0


    # ========================================================
    # M. Torque saturation
    # ========================================================

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    # ========================================================
    # N. 输出给MuJoCo
    # ========================================================

    mj_data.ctrl[:] = tau


    # 返回一些数据用于打印

    return (
        p,
        position_error,
        v_tcp,
        force_task,
        tau
    )


# ============================================================
# 12. Simulation Loop
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:

    while viewer.is_running():

        # 运行一次控制器

        (
            p,
            position_error,
            v_tcp,
            force_task,
            tau
        ) = controller()


        # MuJoCo推进一步物理仿真

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        # 更新Viewer

        viewer.sync()


        # ====================================================
        # 每1000步打印一次
        # ====================================================

        if step % 1000 == 0:

            print("\n========== Cartesian State ==========")

            print("\nCurrent TCP position:")
            print(p)

            print("\nDesired TCP position:")
            print(p_des)

            print("\nPosition error:")
            print(position_error)

            print(
                "\nError norm [m]:",
                np.linalg.norm(position_error)
            )

            print("\nTCP velocity:")
            print(v_tcp)

            print("\nTask force:")
            print(force_task)

            print("\nJoint torque:")
            print(tau)


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )