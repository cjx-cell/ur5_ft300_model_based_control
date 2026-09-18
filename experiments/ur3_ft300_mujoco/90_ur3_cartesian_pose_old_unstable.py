import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

import pinocchio as pin


# ============================================================
# 1. 加载 MuJoCo 模型
# ============================================================

# MuJoCo 负责真正的机器人动力学：
#
# - 刚体动力学
# - 接触
# - 重力
# - actuator
# - 数值积分

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

# Pinocchio 用来计算：
#
# - FK
# - tool0 位姿
# - Jacobian
# - Gravity

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

print("tool0 frame id =", ee_id)


# ============================================================
# 4. MuJoCo q -> Pinocchio q
# ============================================================

def mujoco_to_pinocchio(q_mj):

    # MuJoCo:
    #
    # q =
    # [q1 q2 q3 q4 q5 q6]
    #
    # nq = 6
    #
    #
    # Pinocchio：
    #
    # 前5轴直接存角度。
    #
    # wrist3 是 JointModelRUBZ，
    # configuration 使用：
    #
    # [cos(theta6), sin(theta6)]
    #
    # 所以 Pinocchio：
    #
    # nq = 7
    # nv = 6

    q_pin = pin.neutral(
        pin_model
    )

    # 前五个关节

    q_pin[:5] = q_mj[:5]

    # 第六轴

    theta6 = q_mj[5]

    q_pin[5] = np.cos(theta6)
    q_pin[6] = np.sin(theta6)

    return q_pin


# ============================================================
# 5. 设置机器人初始姿态
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
# 6. 获取初始 tool0 位姿
# ============================================================

q_initial = mujoco_to_pinocchio(
    mj_data.qpos
)


pin.forwardKinematics(
    pin_model,
    pin_data,
    q_initial
)

pin.updateFramePlacements(
    pin_model,
    pin_data
)


T_initial = pin_data.oMf[ee_id]


# 初始位置

p_initial = T_initial.translation.copy()


# 初始姿态

R_initial = T_initial.rotation.copy()


print("\n========== Initial Pose ==========")

print("position:")
print(p_initial)

print("\nrotation:")
print(R_initial)


# ============================================================
# 7. 设置目标位置
# ============================================================

# 这次 TCP 沿世界 Z 方向抬高 3cm。
#
# 上一课用了 5cm。
#
# 这次因为还要同时旋转姿态，
# 所以先用比较温和的 3cm。

p_des = (
    p_initial
    +
    np.array([
        0.0,
        0.0,
        0.03
    ])
)


# ============================================================
# 8. 设置目标姿态
# ============================================================

# 我们让 TCP：
#
# 绕 WORLD Z 轴旋转 +10°
#
# 注意：
#
# 这里不是让某个机器人关节转10°。
#
# 而是让 tool0 的整个空间姿态绕世界Z轴变化10°。


theta = np.deg2rad(
    10.0
)


# 绕世界Z轴的旋转矩阵

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


# 为什么是：
#
# R_des = R_delta @ R_initial
#
# 而不是：
#
# R_initial @ R_delta
#
# 因为我们要求的是：
#
# "绕 WORLD Z轴旋转"
#
# 左乘代表在世界坐标系中施加这个旋转。

R_des = (
    R_delta
    @
    R_initial
)


print("\n========== Desired Pose ==========")

print("position:")
print(p_des)

print("\nrotation:")
print(R_des)


# ============================================================
# 9. Position gains
# ============================================================

# 单位：
#
# Kp_position:
# N / m
#
# Kd_position:
# N*s / m

Kp_position = np.array([
    200.0,
    200.0,
    200.0
])

Kd_position = np.array([
    30.0,
    30.0,
    30.0
])


# ============================================================
# 10. Orientation gains
# ============================================================

# 姿态误差单位是 rad。
#
# 所以：
#
# Kp_orientation
#
# 可以理解成：
#
# N*m / rad
#
#
# 初次实验不要设置太大，
# 尤其 wrist 关节惯量比较小。

Kp_orientation = np.array([
    15.0,
    15.0,
    15.0
])


# 姿态阻尼：
#
# N*m*s / rad

Kd_orientation = np.array([
    3.0,
    3.0,
    3.0
])


# ============================================================
# 11. Joint damping
# ============================================================

# Cartesian 控制之外增加少量 joint damping。
#
# 它不是主控制器。
#
# 主要用于抑制高频关节运动。

joint_damping = np.array([
    0.5,
    0.5,
    0.5,
    0.3,
    0.3,
    0.2
])


# ============================================================
# 12. Torque limits
# ============================================================

# 这次需要控制完整6D姿态，
# 因此 wrist3 不能像上一课一样完全关闭。
#
# 但是先把第6轴最大力矩限制得比较小，
# 防止小惯量 wrist3 瞬间被很大的力矩加速。

tau_limit = np.array([
    50.0,
    50.0,
    28.0,
    12.0,
    12.0,
    3.0
])


# ============================================================
# 13. Controller
# ============================================================

def controller():

    # --------------------------------------------------------
    # A. 获取机器人当前 joint state
    # --------------------------------------------------------

    q_mj = mj_data.qpos.copy()

    dq = mj_data.qvel.copy()


    # --------------------------------------------------------
    # B. 转换成 Pinocchio configuration
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


    # 当前 tool0 位姿

    T_ee = pin_data.oMf[ee_id]


    # 当前位置

    p = T_ee.translation.copy()


    # 当前旋转矩阵

    R = T_ee.rotation.copy()


    # --------------------------------------------------------
    # D. Position error
    # --------------------------------------------------------

    # 世界坐标系表达：
    #
    # e_p = p_des - p

    e_position = (
        p_des
        -
        p
    )


    # --------------------------------------------------------
    # E. Orientation error
    # --------------------------------------------------------

    # 这是本课最重要的部分。
    #
    # 当前：
    #
    # R
    #
    # 目标：
    #
    # R_des
    #
    #
    # 计算：
    #
    # R_error = R_des * R^T
    #
    #
    # 如果当前已经等于目标：
    #
    # R_des = R
    #
    # 那么：
    #
    # R_error = I
    #
    #
    # log(I)=0
    #
    # 所以姿态误差正好为0。

    R_error = (
        R_des
        @
        R.T
    )


    # pin.log3()
    #
    # SO(3) -> R^3
    #
    # 把一个旋转矩阵转换成：
    #
    # rotation vector
    #
    # 向量方向：
    # 旋转轴
    #
    # 向量长度：
    # 旋转角度

    e_orientation = pin.log3(
        R_error
    )


    # --------------------------------------------------------
    # F. Frame Jacobian
    # --------------------------------------------------------

    # 使用 LOCAL_WORLD_ALIGNED：
    #
    # - 原点在 tool0
    # - XYZ轴与WORLD平行
    #
    # 所以我们可以直接把：
    #
    # 世界系位置误差
    # 世界系姿态误差
    #
    # 与这个Jacobain配套使用。

    J = pin.computeFrameJacobian(
        pin_model,
        pin_data,
        q_pin,
        ee_id,
        pin.LOCAL_WORLD_ALIGNED
    )


    # --------------------------------------------------------
    # G. TCP spatial velocity
    # --------------------------------------------------------

    # Pinocchio 这里：
    #
    # J前三行：
    # linear velocity
    #
    # J后三行：
    # angular velocity

    spatial_velocity = (
        J
        @
        dq
    )


    # TCP线速度

    v = spatial_velocity[:3]


    # TCP角速度

    omega = spatial_velocity[3:]


    # --------------------------------------------------------
    # H. Cartesian position PD
    # --------------------------------------------------------

    force = (

        Kp_position
        *
        e_position

        -

        Kd_position
        *
        v
    )


    # --------------------------------------------------------
    # I. Cartesian orientation PD
    # --------------------------------------------------------

    # 姿态控制产生的是：
    #
    # Moment / Torque
    #
    # 而不是Force。

    moment = (

        Kp_orientation
        *
        e_orientation

        -

        Kd_orientation
        *
        omega
    )


    # --------------------------------------------------------
    # J. 构造6D wrench
    # --------------------------------------------------------

    # Pinocchio Jacobian顺序：
    #
    # [linear]
    # [angular]
    #
    # 所以 wrench 对应：
    #
    # [Fx]
    # [Fy]
    # [Fz]
    # [Mx]
    # [My]
    # [Mz]

    wrench = np.concatenate([
        force,
        moment
    ])


    # --------------------------------------------------------
    # K. Cartesian wrench -> Joint torque
    # --------------------------------------------------------

    # 虚功关系：
    #
    # tau_task = J^T * wrench

    tau_task = (
        J.T
        @
        wrench
    )


    # --------------------------------------------------------
    # L. Gravity compensation
    # --------------------------------------------------------

    gravity = pin.computeGeneralizedGravity(
        pin_model,
        pin_data,
        q_pin
    )


    # --------------------------------------------------------
    # M. 最终控制力矩
    # --------------------------------------------------------

    tau = (

        tau_task

        +

        gravity

        -

        joint_damping * dq
    )


    # --------------------------------------------------------
    # N. Torque saturation
    # --------------------------------------------------------

    tau = np.clip(
        tau,
        -tau_limit,
        tau_limit
    )


    # --------------------------------------------------------
    # O. 输出给 MuJoCo
    # --------------------------------------------------------

    mj_data.ctrl[:] = tau


    return (
        p,
        R,
        e_position,
        e_orientation,
        v,
        omega,
        force,
        moment,
        tau
    )


# ============================================================
# 14. Simulation loop
# ============================================================

step = 0


with mujoco.viewer.launch_passive(
    mj_model,
    mj_data
) as viewer:

    while viewer.is_running():

        (
            p,
            R,
            e_position,
            e_orientation,
            v,
            omega,
            force,
            moment,
            tau
        ) = controller()


        # MuJoCo物理积分

        mujoco.mj_step(
            mj_model,
            mj_data
        )


        viewer.sync()


        # ----------------------------------------------------
        # 每1000步打印一次状态
        # ----------------------------------------------------

        if step % 1000 == 0:

            print(
                "\n"
                "========== Cartesian Pose State =========="
            )


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
                "Orientation error norm [rad] =",
                np.linalg.norm(
                    e_orientation
                )
            )


            print(
                "Orientation error norm [deg] =",
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


            print("\nCartesian force:")
            print(force)


            print("\nCartesian moment:")
            print(moment)


            print("\nJoint torque:")
            print(tau)


        step += 1


        time.sleep(
            mj_model.opt.timestep
        )