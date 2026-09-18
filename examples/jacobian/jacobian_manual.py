import numpy as np
import pinocchio as pin


# ============================================================
# 1. 读取机器人模型
# ============================================================

urdf_path = "/home/ubuntu/ur3_ft300_ws/ur3_generated.urdf"

# model：
# 保存机器人结构、关节、惯量等固定信息。
model = pin.buildModelFromUrdf(urdf_path)

# data：
# 保存某个 q 下运动学/动力学的计算结果。
data = model.createData()


# ============================================================
# 2. 设置当前机器人姿态 q
# ============================================================

# neutral configuration：
# 获取一个合法的机器人初始 configuration。
q0 = pin.neutral(model)

# 实际希望机器人六个关节处于下面这些角度。
theta = np.deg2rad([
    30.0,
    -45.0,
    60.0,
    0.0,
    20.0,
    90.0
])

# 因为你的 nq=7、nv=6，
# 不能简单直接 q = theta。
#
# integrate 会正确处理第六轴 continuous joint。
q = pin.integrate(model, q0, theta)


# ============================================================
# 3. Forward Kinematics
# ============================================================

# 计算每个 joint 在世界坐标系下的位置和姿态。
pin.forwardKinematics(
    model,
    data,
    q
)

# forwardKinematics 主要计算 joint placement。
#
# tool0 是一个 frame，因此还需要更新 frame placement。
pin.updateFramePlacements(
    model,
    data
)


# ============================================================
# 4. 获取 tool0 的位置
# ============================================================

ee_name = "tool0"

# 名字 -> frame ID
ee_id = model.getFrameId(ee_name)

# data.oMf[ee_id]
#
# 表示：
#
# 世界坐标系 o
# 到 frame f
# 的 SE3 位姿。
T_ee = data.oMf[ee_id]

# tool0 原点在世界坐标系中的位置
p_e = T_ee.translation.copy()

print("Tool0 position:")
print(p_e)


# ============================================================
# 5. 创建我们自己的 Jacobian
# ============================================================

# UR3 nv = 6
#
# 所以 Jacobian：
#
# 6 行：
# vx vy vz wx wy wz
#
# 6 列：
# joint1 ... joint6
J_manual = np.zeros((6, model.nv))


# ============================================================
# 6. 对六个关节逐个计算 Jacobian column
# ============================================================

for jid in range(1, model.njoints):

    # --------------------------------------------------------
    # A. 当前 joint 在世界坐标系下的位姿
    # --------------------------------------------------------

    # data.oMi[jid]
    #
    # o = world
    # M = SE3 placement
    # i = joint i
    #
    # 即：
    # 世界 -> 第 i 个 joint 的变换。
    T_joint = data.oMi[jid]


    # --------------------------------------------------------
    # B. 获取 joint 原点位置 p_i
    # --------------------------------------------------------

    p_i = T_joint.translation.copy()


    # --------------------------------------------------------
    # C. 获取 joint 的旋转轴 z_i
    # --------------------------------------------------------

    # UR3 的这些 revolute joints
    # 在各自局部坐标系里都是绕 Z 轴旋转。
    #
    # 局部轴：
    z_local = np.array([
        0.0,
        0.0,
        1.0
    ])

    # 但是 Jacobian 现在使用：
    #
    # LOCAL_WORLD_ALIGNED
    #
    # 因此需要把 joint 的局部 Z 轴
    # 转换到世界坐标系。
    #
    # R_world_joint：
    # joint 坐标系相对于世界的旋转矩阵。
    R_world_joint = T_joint.rotation

    # 世界坐标系下的关节轴。
    z_world = R_world_joint @ z_local


    # --------------------------------------------------------
    # D. joint -> TCP 的位置向量
    # --------------------------------------------------------

    # r = p_e - p_i
    #
    # 表示从关节原点指向末端的位置向量。
    r = p_e - p_i


    # --------------------------------------------------------
    # E. Jacobian 的线速度部分
    # --------------------------------------------------------

    # 对转动关节：
    #
    # Jv_i = z_i × (p_e - p_i)
    #
    # np.cross(a, b)
    # 就是计算 a × b。
    Jv = np.cross(
        z_world,
        r
    )


    # --------------------------------------------------------
    # F. Jacobian 的角速度部分
    # --------------------------------------------------------

    # 对转动关节：
    #
    # Jw_i = z_i
    Jw = z_world


    # --------------------------------------------------------
    # G. 填到 Jacobian 对应的一列
    # --------------------------------------------------------

    # jid 从 1 开始：
    #
    # jid=1 -> Jacobian column 0
    # jid=2 -> Jacobian column 1
    # ...
    column = jid - 1

    # 前三行：
    # 线速度部分
    J_manual[:3, column] = Jv

    # 后三行：
    # 角速度部分
    J_manual[3:, column] = Jw


# ============================================================
# 7. 用 Pinocchio 官方函数计算 Jacobian
# ============================================================

J_pin = pin.computeFrameJacobian(
    model,
    data,
    q,
    ee_id,
    pin.LOCAL_WORLD_ALIGNED
)


# ============================================================
# 8. 对比
# ============================================================

np.set_printoptions(
    precision=5,
    suppress=True
)

print("\n========== Manual Jacobian ==========")
print(J_manual)

print("\n========== Pinocchio Jacobian ==========")
print(J_pin)

print("\n========== Difference ==========")

error = J_manual - J_pin

print(error)

print(
    "\nError norm =",
    np.linalg.norm(error)
)